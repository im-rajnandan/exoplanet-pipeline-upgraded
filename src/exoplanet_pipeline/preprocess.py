from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from .config import PipelineConfig
from .schema import RawLightCurve, CleanLightCurve
from .quality import quality_mask_from_flags, summarize_quality_flags, QUALITY_MASK_PRESETS
from .utils import robust_sigma, safe_median, write_json
from .ingest import load_tess_fits


def _is_valid_flux(flux: np.ndarray | None, min_valid_fraction: float) -> bool:
    if flux is None:
        return False
    flux = np.asarray(flux, dtype=float)
    if flux.size == 0:
        return False
    finite = np.isfinite(flux) & (flux > 0)
    if finite.mean() < min_valid_fraction:
        return False
    med = np.nanmedian(flux[finite])
    if not np.isfinite(med) or med <= 0:
        return False
    sig = robust_sigma(flux[finite])
    return bool(np.isfinite(sig) and sig > 0)


def choose_flux(raw: RawLightCurve, config: PipelineConfig):
    """Choose PDCSAP or SAP without relabeling the source."""
    pdcsap_ok = _is_valid_flux(raw.pdcsap_flux, config.min_valid_flux_fraction)
    sap_ok = _is_valid_flux(raw.sap_flux, config.min_valid_flux_fraction)

    if config.preferred_flux == "PDCSAP" and pdcsap_ok:
        return raw.pdcsap_flux, raw.pdcsap_flux_err, "PDCSAP", []
    if config.preferred_flux == "SAP" and sap_ok:
        return raw.sap_flux, raw.sap_flux_err, "SAP", []
    if config.allow_sap_fallback and sap_ok:
        return raw.sap_flux, raw.sap_flux_err, "SAP", ["PDCSAP_UNAVAILABLE_OR_INVALID_USED_SAP"]
    if pdcsap_ok:
        return raw.pdcsap_flux, raw.pdcsap_flux_err, "PDCSAP", ["SAP_INVALID_USED_PDCSAP"]
    return None, None, "NONE", ["NO_VALID_FLUX"]


def rolling_median_trend(time: np.ndarray, flux: np.ndarray, window_days: float) -> np.ndarray:
    """Robust rolling-median trend with interpolation over edge NaNs.

    This fallback avoids requiring specialized packages. For production, wotan's
    biweight filter is often better, but rolling median is transparent and stable.
    """
    time = np.asarray(time, dtype=float)
    flux = np.asarray(flux, dtype=float)
    if len(time) < 5:
        return np.ones_like(flux) * np.nanmedian(flux)

    dt = np.nanmedian(np.diff(np.sort(time)))
    if not np.isfinite(dt) or dt <= 0:
        dt = window_days / 50
    win = max(5, int(round(window_days / dt)))
    if win % 2 == 0:
        win += 1

    s = pd.Series(flux)
    trend = s.rolling(win, center=True, min_periods=max(3, win // 5)).median().to_numpy()
    good = np.isfinite(trend)
    if good.sum() >= 2:
        trend = np.interp(np.arange(len(trend)), np.where(good)[0], trend[good])
    else:
        trend = np.ones_like(flux) * np.nanmedian(flux)
    return trend


def _append_unique_warning(warnings: list[str] | None, warning: str) -> None:
    if warnings is not None and warning not in warnings:
        warnings.append(warning)


def detrend_flux(
    time: np.ndarray,
    flux_norm: np.ndarray,
    config: PipelineConfig,
    window_days: float | None = None,
    warnings: list[str] | None = None,
    warning_context: str = "default",
    mask: np.ndarray | None = None,
):
    window = config.detrend_window_days if window_days is None else float(window_days)
    if config.detrend_method == "none":
        trend = np.ones_like(flux_norm)
        return flux_norm.copy(), trend

    # Determine subset of data to fit trend to (unmasked data)
    if mask is not None:
        fit_indices = ~mask
    else:
        fit_indices = np.ones_like(time, dtype=bool)

    time_fit = time[fit_indices]
    flux_fit = flux_norm[fit_indices]

    # If too few points are left unmasked, fall back to fitting on all data
    if fit_indices.sum() < 10:
        time_fit = time
        flux_fit = flux_norm

    if config.detrend_method == "wotan_biweight":
        try:
            from wotan import flatten
            _, trend_fit = flatten(
                time_fit,
                flux_fit,
                method="biweight",
                window_length=window,
                return_trend=True,
                break_tolerance=0.5,
                cval=5.0,
            )
            trend = np.interp(time, time_fit, trend_fit)
            flat = flux_norm / trend
            return np.asarray(flat, dtype=float), np.asarray(trend, dtype=float)
        except Exception as exc:
            _append_unique_warning(
                warnings,
                f"WOTAN_BIWEIGHT_FAILED_USED_ROLLING_MEDIAN[{warning_context}]:{type(exc).__name__}",
            )

    trend_fit = rolling_median_trend(time_fit, flux_fit, window)
    trend_fit = np.where(np.isfinite(trend_fit) & (trend_fit > 0), trend_fit, np.nanmedian(trend_fit[np.isfinite(trend_fit)]))
    trend = np.interp(time, time_fit, trend_fit)
    flat = flux_norm / trend
    flat = flat / np.nanmedian(flat[np.isfinite(flat)])
    return flat, trend


def compute_gap_metrics(time: np.ndarray) -> dict[str, float | int]:
    if len(time) < 2:
        return {
            "baseline_days": 0.0,
            "median_cadence_days": float("nan"),
            "median_cadence_minutes": float("nan"),
            "n_large_gaps": 0,
            "largest_gap_days": 0.0,
        }
    t = np.sort(np.asarray(time, dtype=float))
    dt = np.diff(t)
    median_dt = float(np.nanmedian(dt))
    threshold = 5.0 * median_dt if np.isfinite(median_dt) and median_dt > 0 else float("inf")
    return {
        "baseline_days": float(np.nanmax(t) - np.nanmin(t)),
        "median_cadence_days": median_dt,
        "median_cadence_minutes": float(median_dt * 24 * 60),
        "n_large_gaps": int(np.sum(dt > threshold)),
        "largest_gap_days": float(np.nanmax(dt)) if dt.size else 0.0,
    }


def preprocess_raw_lightcurve(raw: RawLightCurve, config: PipelineConfig | None = None) -> CleanLightCurve:
    config = config or PipelineConfig()
    warnings: list[str] = []
    qc: dict = {"raw_status": raw.status}

    if raw.status != "RAW_LOADED":
        return CleanLightCurve(
            raw.tic_id, raw.sector, np.array([]), np.array([]), np.array([]), np.array([]), None, None, "NONE",
            np.array([], dtype=bool), np.array([], dtype=bool), np.array([], dtype=bool), np.array([], dtype=bool),
            np.array([], dtype=bool), None, None, raw.metadata, qc, status=raw.status, warnings=[raw.error or raw.status],
        )

    flux, flux_err, source, source_warnings = choose_flux(raw, config)
    warnings.extend(source_warnings)
    if flux is None:
        return CleanLightCurve(
            raw.tic_id, raw.sector, raw.time, np.array([]), np.array([]), np.array([]), None, None, source,
            np.zeros_like(raw.time, dtype=bool), np.zeros_like(raw.time, dtype=bool), np.zeros_like(raw.time, dtype=bool), np.zeros_like(raw.time, dtype=bool),
            np.zeros_like(raw.time, dtype=bool), raw.centroid_col, raw.centroid_row, raw.metadata, qc, status="NO_VALID_FLUX", warnings=warnings,
        )

    time = np.asarray(raw.time, dtype=float)
    flux = np.asarray(flux, dtype=float)
    if flux_err is not None:
        flux_err = np.asarray(flux_err, dtype=float)

    n_raw = len(time)
    finite_mask = np.isfinite(time) & np.isfinite(flux) & (flux > 0)
    if raw.quality is not None and len(raw.quality) == n_raw:
        qmask = quality_mask_from_flags(raw.quality, config.quality_mask_mode)
    else:
        qmask = np.ones(n_raw, dtype=bool)
        if raw.quality is None:
            warnings.append("QUALITY_COLUMN_MISSING")
        else:
            warnings.append("QUALITY_LENGTH_MISMATCH")

    pre_mask = finite_mask & qmask
    if pre_mask.sum() == 0:
        return CleanLightCurve(
            raw.tic_id, raw.sector, np.array([]), np.array([]), np.array([]), np.array([]), None, None, source,
            finite_mask, qmask, np.zeros(n_raw, dtype=bool), pre_mask, np.zeros(n_raw, dtype=bool),
            raw.centroid_col, raw.centroid_row, raw.metadata, qc, status="NO_VALID_POINTS_AFTER_MASKING", warnings=warnings,
        )

    median_flux = np.nanmedian(flux[pre_mask])
    flux_norm_all = flux / median_flux
    flux_err_norm_all = flux_err / median_flux if flux_err is not None and len(flux_err) == n_raw else None

    sigma = robust_sigma(flux_norm_all[pre_mask])
    med = safe_median(flux_norm_all[pre_mask])
    positive_outlier = flux_norm_all > med + config.positive_clip_sigma * sigma
    negative_outlier = flux_norm_all < med - config.negative_clip_sigma * sigma
    if config.remove_extreme_negative_outliers:
        outlier_mask = ~(positive_outlier | negative_outlier)
    else:
        outlier_mask = ~positive_outlier

    final_mask = pre_mask & outlier_mask

    time_c = time[final_mask]
    flux_raw_c = flux[final_mask]
    flux_norm_c = flux_norm_all[final_mask]
    flux_err_c = flux_err_norm_all[final_mask] if flux_err_norm_all is not None else None

    sort_idx = np.argsort(time_c)
    time_c = time_c[sort_idx]
    flux_raw_c = flux_raw_c[sort_idx]
    flux_norm_c = flux_norm_c[sort_idx]
    if flux_err_c is not None:
        flux_err_c = flux_err_c[sort_idx]

    if len(time_c) < config.min_clean_points:
        status = "TOO_FEW_CLEAN_POINTS"
    else:
        status = "OK"

    removed_fraction = 1.0 - (final_mask.sum() / n_raw if n_raw else 0.0)
    if removed_fraction > config.max_removed_fraction:
        warnings.append("HIGH_REMOVED_FRACTION")

    flux_detrended, trend = detrend_flux(time_c, flux_norm_c, config, warnings=warnings, warning_context="default")

    variants = {}
    for w in config.detrend_variants_days:
        try:
            variant_flat, _ = detrend_flux(
                time_c,
                flux_norm_c,
                config,
                window_days=w,
                warnings=warnings,
                warning_context=f"{w:.2f}d",
            )
            variants[f"{w:.2f}d"] = variant_flat
        except Exception as exc:
            warnings.append(f"DETREND_VARIANT_{w}_FAILED:{exc!r}")

    centroid_col_c = None
    centroid_row_c = None
    if raw.centroid_col is not None and len(raw.centroid_col) == n_raw:
        centroid_col_c = np.asarray(raw.centroid_col, dtype=float)[final_mask][sort_idx]
    if raw.centroid_row is not None and len(raw.centroid_row) == n_raw:
        centroid_row_c = np.asarray(raw.centroid_row, dtype=float)[final_mask][sort_idx]

    qc.update({
        "n_raw": int(n_raw),
        "n_finite": int(finite_mask.sum()),
        "n_quality_kept": int(qmask.sum()),
        "n_final": int(final_mask.sum()),
        "removed_fraction": float(removed_fraction),
        "finite_removed_fraction": float(1 - finite_mask.sum() / n_raw) if n_raw else float("nan"),
        "quality_removed_fraction": float(1 - qmask.sum() / n_raw) if n_raw else float("nan"),
        "selected_flux_source": source,
        "median_flux_before_normalization": float(median_flux),
        "median_flux_after_normalization": float(np.nanmedian(flux_norm_c)) if len(flux_norm_c) else float("nan"),
        "median_flux_after_detrending": float(np.nanmedian(flux_detrended)) if len(flux_detrended) else float("nan"),
        "robust_noise_ppm": float(robust_sigma(flux_detrended - np.nanmedian(flux_detrended)) * 1e6) if len(flux_detrended) else float("nan"),
        "rms_ppm": float(np.sqrt(np.nanmean((flux_detrended - np.nanmedian(flux_detrended)) ** 2)) * 1e6) if len(flux_detrended) else float("nan"),
        "n_positive_outliers": int(np.sum(positive_outlier & pre_mask)),
        "n_extreme_negative_outliers": int(np.sum(negative_outlier & pre_mask)),
        "quality_mask_mode": config.quality_mask_mode,
        "quality_mask_value": int(QUALITY_MASK_PRESETS[config.quality_mask_mode]),
        "quality_flag_counts": summarize_quality_flags(raw.quality),
        "crowdsap": raw.metadata.get("crowdsap"),
        "flfrcsap": raw.metadata.get("flfrcsap"),
    })
    qc.update(compute_gap_metrics(time_c))

    if raw.metadata.get("crowdsap") is not None:
        try:
            qc["crowding_risk"] = float(1.0 - float(raw.metadata.get("crowdsap")))
        except Exception:
            qc["crowding_risk"] = None

    return CleanLightCurve(
        tic_id=raw.tic_id,
        sector=raw.sector,
        time=time_c,
        flux_raw_selected=flux_raw_c,
        flux_normalized=flux_norm_c,
        flux_detrended=flux_detrended,
        flux_err=flux_err_c,
        trend=trend,
        selected_flux_source=source,
        finite_mask=finite_mask,
        quality_mask=qmask,
        outlier_mask=outlier_mask,
        final_mask=final_mask,
        negative_outlier_flag=negative_outlier,
        centroid_col=centroid_col_c,
        centroid_row=centroid_row_c,
        metadata=raw.metadata,
        qc=qc,
        detrended_variants=variants,
        status=status,
        warnings=warnings,
        flux_detrended_pass1=flux_detrended.copy(),
    )


def preprocess_fits_file(file_path: str | Path, config: PipelineConfig | None = None) -> CleanLightCurve:
    raw = load_tess_fits(file_path)
    return preprocess_raw_lightcurve(raw, config=config)


def redetrend_with_mask(
    clean: CleanLightCurve,
    mask: np.ndarray | None = None,
    period: float | None = None,
    t0: float | None = None,
    duration: float | None = None,
    config: PipelineConfig | None = None,
) -> CleanLightCurve:
    """Re-detrend the light curve by ignoring the in-transit points specified by mask or parameters."""
    if clean.status != "OK" or len(clean.time) == 0:
        return clean

    config = config or PipelineConfig()

    # Construct mask if parameters are provided
    if mask is None and period is not None and t0 is not None and duration is not None:
        from .detect import make_transit_mask
        mask = make_transit_mask(clean.time, period, t0, duration, width_factor=1.5)

    if mask is None:
        mask = np.zeros_like(clean.time, dtype=bool)

    warnings = list(clean.warnings)
    # Re-detrend main flux
    flux_detrended, trend = detrend_flux(
        clean.time, clean.flux_normalized, config, warnings=warnings, warning_context="redetrend", mask=mask
    )

    # Re-detrend variants
    variants = {}
    for w in config.detrend_variants_days:
        try:
            variant_flat, _ = detrend_flux(
                clean.time,
                clean.flux_normalized,
                config,
                window_days=w,
                warnings=warnings,
                warning_context=f"redetrend_{w:.2f}d",
                mask=mask,
            )
            variants[f"{w:.2f}d"] = variant_flat
        except Exception as exc:
            warnings.append(f"REDETREND_VARIANT_{w}_FAILED:{exc!r}")

    # Create a new CleanLightCurve object with updated values
    import dataclasses
    return dataclasses.replace(
        clean,
        flux_detrended=flux_detrended,
        trend=trend,
        detrended_variants=variants,
        warnings=warnings,
    )


def save_clean_lightcurve(clean: CleanLightCurve, output_dir: str | Path = "data/processed") -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"TIC_{clean.tic_id or 'unknown'}_S{clean.sector or 'unknown'}"
    data_path = output_dir / f"{stem}_clean.csv"
    meta_path = output_dir / f"{stem}_metadata.json"

    df = pd.DataFrame({
        "time": clean.time,
        "flux_raw_selected": clean.flux_raw_selected,
        "flux_normalized": clean.flux_normalized,
        "flux_detrended": clean.flux_detrended,
    })
    if clean.flux_err is not None:
        df["flux_err"] = clean.flux_err
    if clean.centroid_col is not None:
        df["centroid_col"] = clean.centroid_col
    if clean.centroid_row is not None:
        df["centroid_row"] = clean.centroid_row
    for name, arr in clean.detrended_variants.items():
        if len(arr) == len(df):
            df[f"flux_detrended_{name}"] = arr
    df.to_csv(data_path, index=False)

    write_json(meta_path, {
        "tic_id": clean.tic_id,
        "sector": clean.sector,
        "status": clean.status,
        "selected_flux_source": clean.selected_flux_source,
        "warnings": clean.warnings,
        "metadata": clean.metadata,
        "qc": clean.qc,
    })
    return data_path, meta_path
