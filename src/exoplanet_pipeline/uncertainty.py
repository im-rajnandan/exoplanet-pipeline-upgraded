from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any
import numpy as np
import pandas as pd

from .schema import CleanLightCurve, CandidateSignal, TransitFitResult, VettingFeatures, ClassificationResult
from .detect import make_transit_mask, estimate_depth_snr
from .fit import refine_period_epoch_grid, estimate_event_depths
from .utils import robust_sigma


@dataclass
class UncertaintyResult:
    """Part 7 uncertainty and confidence summary for one candidate.

    The estimates are intentionally transparent and robust rather than pretending
    to be a full astrophysical posterior. They combine: local photometric noise,
    event-to-event depth scatter, bootstrap resampling, ephemeris-grid curvature,
    multi-detrender stability, and classifier margin.
    """

    tic_id: int | None
    sector: int | None
    candidate_id: int

    period_days: float
    period_err_days: float
    epoch_time: float
    epoch_err_days: float
    duration_days: float
    duration_err_days: float
    depth_ppm: float
    depth_err_ppm: float
    snr: float
    effective_snr: float
    red_noise_beta: float

    detection_confidence: float
    parameter_confidence: float
    classification_confidence: float
    final_confidence: float
    confidence_level: str

    depth_err_sources: dict[str, float]
    stability_metrics: dict[str, float]
    warnings: list[str]
    method: str = "robust_bootstrap_stability_uncertainty"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["warnings"] = ";".join(self.warnings)
        # Flatten compactly for CSV friendliness.
        for k, v in self.depth_err_sources.items():
            d[f"depth_err_source_{k}"] = v
        for k, v in self.stability_metrics.items():
            d[f"stability_{k}"] = v
        d.pop("depth_err_sources", None)
        d.pop("stability_metrics", None)
        return d


def _safe_float(x: Any, default: float = np.nan) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) else default
    except Exception:
        return default


def _sigmoid(x: float, center: float, scale: float) -> float:
    if not np.isfinite(x):
        return 0.0
    return float(1.0 / (1.0 + np.exp(-(x - center) / max(scale, 1e-9))))


def _mad_or_nan(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size <= 1:
        return np.nan
    return float(1.4826 * np.nanmedian(np.abs(values - np.nanmedian(values))))


def estimate_red_noise_beta(time: np.ndarray, flux: np.ndarray, duration_days: float) -> float:
    """Estimate a simple time-correlated-noise inflation factor.

    White noise should bin down approximately as sqrt(N). We compare unbinned
    scatter with scatter after binning on roughly one transit-duration timescale.
    beta > 1 inflates uncertainties and reduces effective SNR.
    """
    time = np.asarray(time, dtype=float)
    flux = np.asarray(flux, dtype=float)
    finite = np.isfinite(time) & np.isfinite(flux)
    time, flux = time[finite], flux[finite]
    if len(time) < 100:
        return 1.0
    sigma1 = robust_sigma(flux - np.nanmedian(flux))
    if not np.isfinite(sigma1) or sigma1 <= 0:
        return 1.0
    dt = np.nanmedian(np.diff(np.sort(time)))
    if not np.isfinite(dt) or dt <= 0 or not np.isfinite(duration_days) or duration_days <= 0:
        return 1.0
    bin_n = int(max(3, round(duration_days / dt)))
    if bin_n >= len(flux) // 5:
        return 1.0
    n_bins = len(flux) // bin_n
    trimmed = flux[: n_bins * bin_n]
    binned = np.nanmean(trimmed.reshape(n_bins, bin_n), axis=1)
    sigma_bin = robust_sigma(binned - np.nanmedian(binned))
    expected = sigma1 / np.sqrt(bin_n)
    beta = sigma_bin / expected if np.isfinite(sigma_bin) and expected > 0 else 1.0
    return float(np.clip(beta, 1.0, 5.0))


def residual_bootstrap_depth_uncertainty(
    time: np.ndarray,
    flux: np.ndarray,
    period: float,
    t0: float,
    duration: float,
    n_bootstrap: int = 300,
    random_seed: int = 42,
) -> dict[str, float]:
    """Bootstrap transit depth by resampling out-of-transit residuals.

    This captures photometric scatter around the local baseline. It does not
    model TTVs or physical limb-darkening; it is a robust first-pass uncertainty
    appropriate for a detection/classification pipeline.
    """
    time = np.asarray(time, dtype=float)
    flux = np.asarray(flux, dtype=float)
    finite = np.isfinite(time) & np.isfinite(flux)
    time, flux = time[finite], flux[finite]
    in_tr = make_transit_mask(time, period, t0, duration, width_factor=1.0)
    near = make_transit_mask(time, period, t0, duration, width_factor=5.0)
    out = ~near
    if in_tr.sum() < 3 or out.sum() < 20:
        return {"depth_bootstrap_err_fraction": np.nan, "depth_bootstrap_err_ppm": np.nan}

    baseline = np.nanmedian(flux[out])
    in_flux = np.nanmedian(flux[in_tr])
    depth = baseline - in_flux
    out_resid = flux[out] - baseline
    in_resid = flux[in_tr] - in_flux
    out_resid = out_resid[np.isfinite(out_resid)]
    in_resid = in_resid[np.isfinite(in_resid)]
    if out_resid.size < 10 or in_resid.size < 3:
        return {"depth_bootstrap_err_fraction": np.nan, "depth_bootstrap_err_ppm": np.nan}
    rng = np.random.default_rng(random_seed)
    vals = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        # resample both baseline and in-transit residuals to preserve the median estimator's variance
        out_sample = baseline + rng.choice(out_resid, size=out_resid.size, replace=True)
        in_sample = in_flux + rng.choice(in_resid, size=in_resid.size, replace=True)
        vals[i] = np.nanmedian(out_sample) - np.nanmedian(in_sample)
    err = float(np.nanstd(vals, ddof=1))
    # Degenerate synthetic cases may produce zero; keep a noise-floor based estimate then.
    if not np.isfinite(err) or err <= 0:
        noise = robust_sigma(flux[out] - np.nanmedian(flux[out]))
        err = noise / np.sqrt(in_tr.sum()) if np.isfinite(noise) and noise > 0 else np.nan
    return {
        "depth_bootstrap_median_fraction": float(np.nanmedian(vals)) if np.isfinite(vals).any() else float(depth),
        "depth_bootstrap_err_fraction": err,
        "depth_bootstrap_err_ppm": err * 1e6 if np.isfinite(err) else np.nan,
    }


def estimate_multidetrender_stability(clean: CleanLightCurve, candidate: CandidateSignal) -> dict[str, float]:
    """Estimate how stable depth/SNR are across Part-1 detrending variants."""
    if not clean.detrended_variants:
        return {
            "n_variants": 1.0,
            "depth_mad_ppm": np.nan,
            "snr_mad": np.nan,
            "period_cv_proxy": np.nan,
            "stability_score": 0.75,
        }
    depths = []
    snrs = []
    for name, flux in clean.detrended_variants.items():
        stats = estimate_depth_snr(clean.time, flux, candidate.period_days, candidate.epoch_time, candidate.duration_days)
        depths.append(_safe_float(stats.get("depth_ppm")))
        snrs.append(_safe_float(stats.get("local_snr", stats.get("snr"))))
    depths = np.asarray(depths, dtype=float)
    snrs = np.asarray(snrs, dtype=float)
    depth_mad = _mad_or_nan(depths)
    snr_mad = _mad_or_nan(snrs)
    median_depth = np.nanmedian(depths) if np.isfinite(depths).any() else np.nan
    rel_depth = abs(depth_mad / median_depth) if np.isfinite(depth_mad) and np.isfinite(median_depth) and median_depth != 0 else np.nan
    rel_snr = abs(snr_mad / np.nanmedian(snrs)) if np.isfinite(snr_mad) and np.isfinite(np.nanmedian(snrs)) and np.nanmedian(snrs) != 0 else np.nan
    penalty = 0.0
    if np.isfinite(rel_depth):
        penalty += min(rel_depth, 1.0) * 0.50
    if np.isfinite(rel_snr):
        penalty += min(rel_snr, 1.0) * 0.25
    stability_score = float(np.clip(1.0 - penalty, 0.0, 1.0))
    return {
        "n_variants": float(len(clean.detrended_variants)),
        "depth_mad_ppm": float(depth_mad) if np.isfinite(depth_mad) else np.nan,
        "snr_mad": float(snr_mad) if np.isfinite(snr_mad) else np.nan,
        "period_cv_proxy": 0.0,  # period is fixed here; Part 8 injection recovery measures true period stability
        "stability_score": stability_score,
    }


def estimate_classification_margin(classification: ClassificationResult | None, ai_row: pd.Series | None = None) -> float:
    """Return probability margin between top two final classes if available."""
    if ai_row is not None:
        cols = [c for c in ai_row.index if str(c).startswith("final_prob_")]
        vals = sorted([_safe_float(ai_row[c], 0.0) for c in cols], reverse=True)
        if len(vals) >= 2:
            return float(max(vals[0] - vals[1], 0.0))
    if classification is None:
        return np.nan
    scores = np.array([
        classification.planet_score,
        classification.eb_score,
        classification.blend_score,
        classification.stellar_variability_score,
        classification.systematic_score,
    ], dtype=float)
    scores = scores[np.isfinite(scores)]
    if scores.size < 2:
        return np.nan
    scores = np.sort(scores)[::-1]
    return float(max(scores[0] - scores[1], 0.0))


def estimate_candidate_uncertainty(
    clean: CleanLightCurve,
    candidate: CandidateSignal,
    fit: TransitFitResult,
    vetting: VettingFeatures | None = None,
    classification: ClassificationResult | None = None,
    ai_row: pd.Series | None = None,
    n_bootstrap: int = 300,
    random_seed: int = 42,
) -> UncertaintyResult:
    """Part 7: estimate detection/parameter/classification uncertainty for one candidate."""
    warnings_list: list[str] = []
    time = np.asarray(clean.time, dtype=float)
    flux = np.asarray(clean.flux_detrended, dtype=float)
    finite = np.isfinite(time) & np.isfinite(flux)
    time, flux = time[finite], flux[finite]
    beta = estimate_red_noise_beta(time, flux, fit.duration_days)
    effective_snr = fit.snr / beta if np.isfinite(fit.snr) and beta > 0 else np.nan

    boot = residual_bootstrap_depth_uncertainty(
        time,
        flux,
        fit.period_days,
        fit.epoch_time,
        fit.duration_days,
        n_bootstrap=n_bootstrap,
        random_seed=random_seed,
    )
    event_depths = pd.DataFrame(fit.event_depths or [])
    event_scatter_ppm = np.nan
    event_err_ppm = np.nan
    if not event_depths.empty and "depth_ppm" in event_depths:
        d = pd.to_numeric(event_depths["depth_ppm"], errors="coerce").to_numpy(dtype=float)
        d = d[np.isfinite(d)]
        event_scatter_ppm = _mad_or_nan(d)
        if d.size > 1 and np.isfinite(event_scatter_ppm):
            event_err_ppm = event_scatter_ppm / np.sqrt(d.size)

    formal_depth_err = _safe_float(fit.depth_err_ppm)
    bootstrap_depth_err = _safe_float(boot.get("depth_bootstrap_err_ppm"))
    depth_sources = {
        "formal_fit_ppm": formal_depth_err,
        "residual_bootstrap_ppm": bootstrap_depth_err,
        "event_scatter_ppm": event_scatter_ppm,
        "event_median_err_ppm": event_err_ppm,
    }
    # Conservative depth uncertainty: max of finite estimates, inflated by red-noise beta.
    finite_depth_errs = [v for v in depth_sources.values() if np.isfinite(v) and v >= 0]
    if finite_depth_errs:
        depth_err_ppm = float(max(finite_depth_errs) * beta)
    else:
        depth_err_ppm = np.nan
        warnings_list.append("DEPTH_UNCERTAINTY_UNAVAILABLE")

    # Use existing fit uncertainties, but inflate by red-noise and stability penalties.
    stability = estimate_multidetrender_stability(clean, candidate)
    stability_score = _safe_float(stability.get("stability_score"), 0.75)
    stability_inflation = 1.0 + max(0.0, 1.0 - stability_score)
    period_err_days = _safe_float(fit.period_err_days) * beta * stability_inflation
    epoch_err_days = _safe_float(fit.epoch_err_days) * beta * stability_inflation
    duration_err_days = _safe_float(fit.duration_err_days) * beta * stability_inflation

    # Confidence pieces: detection, parameters, classification.
    sde = _safe_float(candidate.sde)
    snr_score = _sigmoid(effective_snr, center=7.0, scale=2.0)
    sde_score = _sigmoid(sde, center=6.0, scale=1.5) if np.isfinite(sde) else 0.5
    ntrans_score = min(max(candidate.n_full_transits / 3.0, 0.0), 1.0)
    data_quality = _safe_float(vetting.data_quality_score if vetting is not None else clean.qc.get("data_quality_score"), 0.7)
    detection_conf = float(np.clip(0.45 * snr_score + 0.25 * sde_score + 0.20 * ntrans_score + 0.10 * data_quality, 0.0, 1.0))

    rel_depth_err = abs(depth_err_ppm / fit.depth_ppm) if np.isfinite(depth_err_ppm) and np.isfinite(fit.depth_ppm) and fit.depth_ppm != 0 else np.inf
    parameter_conf = float(np.clip(1.0 - 0.7 * min(rel_depth_err, 1.0) - 0.3 * (1.0 - stability_score), 0.0, 1.0))
    if not np.isfinite(rel_depth_err):
        parameter_conf = min(parameter_conf, 0.5)

    if ai_row is not None and "final_confidence" in ai_row:
        class_conf = _safe_float(ai_row.get("final_confidence"), np.nan)
    elif classification is not None:
        class_conf = _safe_float(classification.confidence, np.nan)
    else:
        class_conf = np.nan
    margin = estimate_classification_margin(classification, ai_row)
    if np.isfinite(class_conf) and np.isfinite(margin):
        classification_conf = float(np.clip(0.75 * class_conf + 0.25 * min(margin / 0.35, 1.0), 0.0, 1.0))
    elif np.isfinite(class_conf):
        classification_conf = float(class_conf)
    else:
        classification_conf = 0.5
        warnings_list.append("CLASSIFICATION_CONFIDENCE_UNAVAILABLE")

    # Epistemic uncertainty penalty from MC Dropout
    epistemic_unc = _safe_float(ai_row.get("cnn_epistemic_uncertainty")) if ai_row is not None else np.nan
    if np.isfinite(epistemic_unc) and epistemic_unc > 0.0:
        penalty = float(np.clip(epistemic_unc, 0.0, 0.5))
        classification_conf = float(np.clip(classification_conf - penalty, 0.0, 1.0))

    final_conf = float(np.clip(0.35 * detection_conf + 0.30 * parameter_conf + 0.35 * classification_conf, 0.0, 1.0))
    if final_conf >= 0.80:
        level = "HIGH"
    elif final_conf >= 0.60:
        level = "MEDIUM"
    elif final_conf >= 0.40:
        level = "LOW"
    else:
        level = "VERY_LOW"

    if beta > 1.5:
        warnings_list.append("RED_NOISE_INFLATED_UNCERTAINTIES")
    if stability_score < 0.6:
        warnings_list.append("DETRENDER_STABILITY_LOW")
    if candidate.n_full_transits < 2:
        warnings_list.append("FEW_FULL_TRANSITS")

    return UncertaintyResult(
        tic_id=clean.tic_id,
        sector=clean.sector,
        candidate_id=candidate.candidate_id,
        period_days=float(fit.period_days),
        period_err_days=float(period_err_days) if np.isfinite(period_err_days) else np.nan,
        epoch_time=float(fit.epoch_time),
        epoch_err_days=float(epoch_err_days) if np.isfinite(epoch_err_days) else np.nan,
        duration_days=float(fit.duration_days),
        duration_err_days=float(duration_err_days) if np.isfinite(duration_err_days) else np.nan,
        depth_ppm=float(fit.depth_ppm),
        depth_err_ppm=float(depth_err_ppm) if np.isfinite(depth_err_ppm) else np.nan,
        snr=float(fit.snr),
        effective_snr=float(effective_snr) if np.isfinite(effective_snr) else np.nan,
        red_noise_beta=float(beta),
        detection_confidence=detection_conf,
        parameter_confidence=parameter_conf,
        classification_confidence=classification_conf,
        final_confidence=final_conf,
        confidence_level=level,
        depth_err_sources=depth_sources,
        stability_metrics=stability,
        warnings=warnings_list,
    )


def add_uncertainty_columns(catalog: pd.DataFrame, prefix: str = "unc_") -> pd.DataFrame:
    """Add lightweight catalog-level uncertainty proxies when raw light curves are unavailable.

    This is useful for organizer-provided catalogs where only candidate features
    exist. The full estimate_candidate_uncertainty function is better when the
    actual light curve is available.
    """
    out = catalog.copy()
    depth = pd.to_numeric(out.get("fit_depth_ppm", out.get("depth_ppm", np.nan)), errors="coerce")
    depth_err = pd.to_numeric(out.get("fit_depth_err_ppm", np.nan), errors="coerce")
    snr = pd.to_numeric(out.get("fit_snr", out.get("local_snr", out.get("snr", np.nan))), errors="coerce")
    if "vet_data_quality_score" in out.columns:
        data_quality = pd.to_numeric(out["vet_data_quality_score"], errors="coerce").fillna(0.7)
    else:
        data_quality = pd.Series(0.7, index=out.index)
    if "vet_red_noise_proxy" in out.columns:
        red_noise = pd.to_numeric(out["vet_red_noise_proxy"], errors="coerce").fillna(0.0)
    else:
        red_noise = pd.Series(0.0, index=out.index)
    beta = (1.0 + red_noise.clip(lower=0.0, upper=1.0)).clip(lower=1.0, upper=2.0)
    fallback_depth_err = (depth.abs() / snr.replace(0, np.nan).abs()).replace([np.inf, -np.inf], np.nan)
    final_depth_err = depth_err.where(depth_err.notna() & (depth_err > 0), fallback_depth_err) * beta
    eff_snr = snr / beta
    out[f"{prefix}depth_err_ppm"] = final_depth_err
    out[f"{prefix}red_noise_beta"] = beta
    out[f"{prefix}effective_snr"] = eff_snr
    out[f"{prefix}relative_depth_err"] = (final_depth_err / depth.abs()).replace([np.inf, -np.inf], np.nan)
    out[f"{prefix}detection_confidence"] = (1.0 / (1.0 + np.exp(-(eff_snr - 7.0) / 2.0))).clip(0, 1)
    out[f"{prefix}parameter_confidence"] = (1.0 - out[f"{prefix}relative_depth_err"].clip(0, 1) * 0.7).clip(0, 1) * data_quality
    return out


def uncertainty_to_dataframe(result: UncertaintyResult) -> pd.DataFrame:
    return pd.DataFrame([result.to_dict()])
