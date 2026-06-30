from __future__ import annotations

from dataclasses import asdict
import warnings as py_warnings
import numpy as np
import pandas as pd

from .config import PipelineConfig
from .schema import CleanLightCurve, CandidateSignal, DetectionResult
from .utils import robust_sigma


def make_transit_mask(time: np.ndarray, period: float, t0: float, duration: float, width_factor: float = 1.0) -> np.ndarray:
    """True for points inside a periodic transit window centered at t0 + nP."""
    time = np.asarray(time, dtype=float)
    if not np.isfinite(period) or period <= 0 or not np.isfinite(duration) or duration <= 0:
        return np.zeros_like(time, dtype=bool)
    phase_time = ((time - t0 + 0.5 * period) % period) - 0.5 * period
    return np.abs(phase_time) < 0.5 * width_factor * duration


def transit_event_numbers(time: np.ndarray, period: float, t0: float) -> np.ndarray:
    return np.floor((np.asarray(time) - t0) / period + 0.5).astype(int)


def count_transits(time: np.ndarray, period: float, t0: float, duration: float) -> tuple[int, int]:
    mask = make_transit_mask(time, period, t0, duration)
    if mask.sum() == 0:
        return 0, 0
    event_ids = transit_event_numbers(time[mask], period, t0)
    unique, counts = np.unique(event_ids, return_counts=True)
    n_transits = len(unique)
    # Approximate full transit: at least 3 in-transit points. Part 3 can improve this.
    n_full = int(np.sum(counts >= 3))
    return int(n_transits), int(n_full)


def estimate_depth_snr(time: np.ndarray, flux: np.ndarray, period: float, t0: float, duration: float) -> dict[str, float | int]:
    in_tr = make_transit_mask(time, period, t0, duration, width_factor=1.0)
    near_tr = make_transit_mask(time, period, t0, duration, width_factor=5.0)
    out_tr = ~near_tr
    local_out = near_tr & (~in_tr)

    if in_tr.sum() < 3 or out_tr.sum() < 20:
        return {
            "depth_fraction": float("nan"),
            "depth_ppm": float("nan"),
            "snr": float("nan"),
            "local_snr": float("nan"),
            "n_in_transit_points": int(in_tr.sum()),
        }

    baseline = np.nanmedian(flux[out_tr])
    in_flux = np.nanmedian(flux[in_tr])
    depth = float(baseline - in_flux)
    global_noise = robust_sigma(flux[out_tr] - np.nanmedian(flux[out_tr]))
    if local_out.sum() >= 20:
        local_noise = robust_sigma(flux[local_out] - np.nanmedian(flux[local_out]))
    else:
        local_noise = global_noise

    n_in = int(in_tr.sum())
    snr = depth / global_noise * np.sqrt(n_in) if global_noise and np.isfinite(global_noise) and global_noise > 0 else float("nan")
    local_snr = depth / local_noise * np.sqrt(n_in) if local_noise and np.isfinite(local_noise) and local_noise > 0 else float("nan")

    return {
        "depth_fraction": depth,
        "depth_ppm": float(depth * 1e6),
        "snr": float(snr),
        "local_snr": float(local_snr),
        "n_in_transit_points": n_in,
    }


def build_period_grid(time: np.ndarray, config: PipelineConfig) -> np.ndarray:
    baseline = float(np.nanmax(time) - np.nanmin(time)) if len(time) else 0.0
    period_max = config.period_max_days
    if period_max is None:
        period_max = min(13.5, baseline / max(config.min_transits, 2))
    period_min = config.period_min_days
    if period_max <= period_min:
        period_max = max(period_min * 1.5, baseline / 2.0)

    mode = getattr(config, "period_grid_mode", "frequency")
    if mode == "frequency":
        freq_min = 1.0 / period_max
        freq_max = 1.0 / period_min
        freqs = np.linspace(freq_min, freq_max, config.n_periods)
        return 1.0 / freqs
    else:
        return np.linspace(period_min, period_max, config.n_periods)


def build_duration_grid(config: PipelineConfig) -> np.ndarray:
    return np.linspace(config.min_duration_days, config.max_duration_days, config.n_durations)


def _status_from_scores(snr: float, sde: float | None, n_transits: int, config: PipelineConfig) -> str:
    sde_value = -np.inf if sde is None or not np.isfinite(sde) else float(sde)
    snr_value = -np.inf if not np.isfinite(snr) else float(snr)
    if n_transits >= 3 and snr_value >= config.strong_snr_threshold and sde_value >= config.strong_sde_threshold:
        return "STRONG_DETECTION"
    if n_transits >= config.min_transits and snr_value >= config.weak_snr_threshold and sde_value >= config.weak_sde_threshold:
        return "WEAK_DETECTION"
    # If BLS is used without calibrated SDE, use SNR and transit count only.
    if n_transits >= 3 and snr_value >= config.strong_snr_threshold and sde is None:
        return "STRONG_DETECTION"
    if n_transits >= config.min_transits and snr_value >= config.weak_snr_threshold and sde is None:
        return "WEAK_DETECTION"
    return "NO_DETECTION"



def run_numpy_box(clean: CleanLightCurve, config: PipelineConfig, flux: np.ndarray | None = None, detrend_variant: str = "default") -> tuple[CandidateSignal | None, dict]:
    """Dependency-free periodic box search fallback.

    This is not as accurate as astropy BLS/TLS, but it makes the project runnable
    in minimal environments and is excellent for synthetic sanity checks. It bins
    phase-folded flux for each trial period, searches for the lowest contiguous
    phase window, and then re-estimates depth/SNR on the original time series.
    """
    time = np.asarray(clean.time, dtype=float)
    y = np.asarray(clean.flux_detrended if flux is None else flux, dtype=float)
    finite = np.isfinite(time) & np.isfinite(y)
    time = time[finite]
    y = y[finite]
    if len(time) < config.min_clean_points:
        return None, {"status": "TOO_FEW_POINTS"}

    periods = build_period_grid(time, config)
    duration_grid = build_duration_grid(config)
    n_phase_bins = 240
    best = {"score": -np.inf, "period": np.nan, "duration": np.nan, "phase_center": np.nan, "power": np.nan}
    y_med = np.nanmedian(y)

    for period in periods:
        phase = ((time - time.min()) / period) % 1.0
        bins = np.floor(phase * n_phase_bins).astype(int)
        bins = np.clip(bins, 0, n_phase_bins - 1)
        sums = np.bincount(bins, weights=y - y_med, minlength=n_phase_bins).astype(float)
        counts = np.bincount(bins, minlength=n_phase_bins).astype(float)
        means = np.full(n_phase_bins, np.nan)
        good = counts > 0
        means[good] = sums[good] / counts[good]
        if np.sum(good) < 0.5 * n_phase_bins:
            continue
        fill = np.nanmedian(means[good])
        means = np.where(np.isfinite(means), means, fill)
        doubled = np.concatenate([means, means])
        for duration in duration_grid:
            width_bins = int(round((duration / period) * n_phase_bins))
            width_bins = max(1, min(width_bins, n_phase_bins // 3))
            kernel = np.ones(width_bins) / width_bins
            roll = np.convolve(doubled, kernel, mode="valid")[:n_phase_bins]
            idx = int(np.nanargmin(roll))
            depth_proxy = -float(roll[idx])
            out_scatter = robust_sigma(means)
            score = depth_proxy / out_scatter * np.sqrt(max(width_bins, 1)) if np.isfinite(out_scatter) and out_scatter > 0 else -np.inf
            if score > best["score"]:
                best.update({"score": float(score), "period": float(period), "duration": float(duration), "phase_center": float((idx + 0.5 * width_bins) / n_phase_bins), "power": float(depth_proxy)})

    if not np.isfinite(best["score"]):
        return None, {"status": "NUMPY_BOX_NO_POWER"}

    period = best["period"]
    duration = best["duration"]
    # Convert phase center relative to time.min() into T0 in same time system.
    t0 = float(time.min() + best["phase_center"] * period)
    # Shift T0 close to the first observed event but preserve phase.
    while t0 - period > time.min():
        t0 -= period
    while t0 < time.min() - 0.5 * period:
        t0 += period

    depth_stats = estimate_depth_snr(time, y, period, t0, duration)
    n_tr, n_full = count_transits(time, period, t0, duration)
    sde = float(best["score"])
    status = _status_from_scores(float(depth_stats["snr"]), sde, n_tr, config)
    warnings_list = ["NUMPY_BOX_FALLBACK_USED"]
    if n_tr < config.min_transits:
        warnings_list.append("TOO_FEW_TRANSITS")
    if depth_stats["depth_fraction"] <= 0:
        warnings_list.append("NON_POSITIVE_DEPTH")

    candidate = CandidateSignal(
        tic_id=clean.tic_id,
        sector=clean.sector,
        candidate_id=1,
        period_days=period,
        epoch_time=t0,
        duration_days=duration,
        depth_fraction=float(depth_stats["depth_fraction"]),
        depth_ppm=float(depth_stats["depth_ppm"]),
        snr=float(depth_stats["snr"]),
        local_snr=float(depth_stats["local_snr"]),
        sde=sde,
        fap=None,
        n_transits=n_tr,
        n_full_transits=n_full,
        n_in_transit_points=int(depth_stats["n_in_transit_points"]),
        detection_method="NUMPY_BOX_FALLBACK",
        flux_source=clean.selected_flux_source,
        detrend_variant=detrend_variant,
        periodogram_peak_power=float(best["power"]),
        period_uncertainty_rough=None,
        status=status,
        warnings=warnings_list,
        extra={"n_phase_bins": n_phase_bins, "period_grid_min": float(periods.min()), "period_grid_max": float(periods.max())},
    )
    diag = {"status": "OK", "method": "NUMPY_BOX_FALLBACK", "best_score": best["score"], "periods": periods, "power": np.array([])}
    return candidate, diag


def run_bls(clean: CleanLightCurve, config: PipelineConfig, flux: np.ndarray | None = None, detrend_variant: str = "default") -> tuple[CandidateSignal | None, dict]:
    try:
        from astropy.timeseries import BoxLeastSquares
    except ImportError:
        return run_numpy_box(clean, config, flux=flux, detrend_variant=detrend_variant)

    time = np.asarray(clean.time, dtype=float)
    y = np.asarray(clean.flux_detrended if flux is None else flux, dtype=float)
    finite = np.isfinite(time) & np.isfinite(y)
    time = time[finite]
    y = y[finite]
    if len(time) < config.min_clean_points:
        return None, {"status": "TOO_FEW_POINTS"}

    periods = build_period_grid(time, config)
    durations = build_duration_grid(config)
    raw_n_durations = int(len(durations))
    min_period = float(np.nanmin(periods)) if len(periods) else float("nan")
    duration_grid_clipped = False
    if np.isfinite(min_period):
        valid_duration = np.isfinite(durations) & (durations > 0) & (durations < min_period)
        duration_grid_clipped = bool(valid_duration.sum() != len(durations))
        durations = durations[valid_duration]
    if len(durations) == 0:
        cand, diag = run_numpy_box(clean, config, flux=flux, detrend_variant=detrend_variant)
        if cand is not None:
            cand.warnings.append("BLS_DURATION_GRID_INVALID_USED_NUMPY_FALLBACK")
        return cand, {
            "status": "BLS_DURATION_GRID_INVALID_USED_NUMPY_FALLBACK",
            "period_grid_min": min_period,
            "raw_n_durations": raw_n_durations,
            "fallback": diag,
        }
    y_centered = y - np.nanmedian(y)

    try:
        bls = BoxLeastSquares(time, y_centered)
        with py_warnings.catch_warnings():
            py_warnings.simplefilter("ignore")
            res = bls.power(periods, durations)
    except Exception as exc:
        cand, diag = run_numpy_box(clean, config, flux=flux, detrend_variant=detrend_variant)
        if cand is not None:
            cand.warnings.append("BLS_FAILED_USED_NUMPY_FALLBACK")
        return cand, {"status": "BLS_FAILED_USED_NUMPY_FALLBACK", "error": repr(exc), "fallback": diag}

    if len(res.power) == 0 or not np.isfinite(res.power).any():
        return None, {"status": "NO_PERIOD_POWER"}

    best_idx = int(np.nanargmax(res.power))
    period = float(res.period[best_idx])
    duration = float(res.duration[best_idx])
    t0 = float(res.transit_time[best_idx])
    peak_power = float(res.power[best_idx])

    depth_stats = estimate_depth_snr(time, y, period, t0, duration)
    n_tr, n_full = count_transits(time, period, t0, duration)

    # A simple BLS SDE-like statistic: peak minus median over robust sigma of periodogram power.
    power_sigma = robust_sigma(res.power)
    sde = float((peak_power - np.nanmedian(res.power)) / power_sigma) if power_sigma and np.isfinite(power_sigma) and power_sigma > 0 else None
    status = _status_from_scores(float(depth_stats["snr"]), sde, n_tr, config)

    warnings_list = []
    if duration_grid_clipped:
        warnings_list.append("BLS_DURATION_GRID_CLIPPED")
    if n_tr < config.min_transits:
        warnings_list.append("TOO_FEW_TRANSITS")
    if depth_stats["depth_fraction"] <= 0:
        warnings_list.append("NON_POSITIVE_DEPTH")

    candidate = CandidateSignal(
        tic_id=clean.tic_id,
        sector=clean.sector,
        candidate_id=1,
        period_days=period,
        epoch_time=t0,
        duration_days=duration,
        depth_fraction=float(depth_stats["depth_fraction"]),
        depth_ppm=float(depth_stats["depth_ppm"]),
        snr=float(depth_stats["snr"]),
        local_snr=float(depth_stats["local_snr"]),
        sde=sde,
        fap=None,
        n_transits=n_tr,
        n_full_transits=n_full,
        n_in_transit_points=int(depth_stats["n_in_transit_points"]),
        detection_method="BLS",
        flux_source=clean.selected_flux_source,
        detrend_variant=detrend_variant,
        periodogram_peak_power=peak_power,
        period_uncertainty_rough=None,
        status=status,
        warnings=warnings_list,
        extra={
            "period_grid_min": float(periods.min()),
            "period_grid_max": float(periods.max()),
            "n_periods": int(len(periods)),
            "n_durations": int(len(durations)),
            "raw_n_durations": raw_n_durations,
        },
    )
    diag = {
        "status": "OK",
        "periods": periods,
        "power": np.asarray(res.power),
        "durations": durations,
        "raw_n_durations": raw_n_durations,
        "duration_grid_clipped": duration_grid_clipped,
    }
    return candidate, diag


def run_tls(clean: CleanLightCurve, config: PipelineConfig, flux: np.ndarray | None = None, detrend_variant: str = "default") -> tuple[CandidateSignal | None, dict]:
    try:
        from transitleastsquares import transitleastsquares
    except ImportError:
        return None, {"status": "MISSING_TRANSITLEASTSQUARES", "error": "Install transitleastsquares for TLS detection."}

    time = np.asarray(clean.time, dtype=float)
    y = np.asarray(clean.flux_detrended if flux is None else flux, dtype=float)
    finite = np.isfinite(time) & np.isfinite(y)
    time = time[finite]
    y = y[finite]
    if len(time) < config.min_clean_points:
        return None, {"status": "TOO_FEW_POINTS"}

    baseline = float(np.nanmax(time) - np.nanmin(time))
    period_max = config.period_max_days or min(13.5, baseline / max(config.min_transits, 2))

    try:
        model = transitleastsquares(time, y)
        res = model.power(
            period_min=config.period_min_days,
            period_max=period_max,
            n_transits_min=config.min_transits,
            show_progress_bar=False,
        )
    except Exception as exc:
        return None, {"status": "TLS_FAILED", "error": repr(exc)}

    period = float(res.period)
    duration = float(res.duration)
    t0 = float(res.T0)
    depth_fraction = float(getattr(res, "depth", np.nan))
    depth_stats = estimate_depth_snr(time, y, period, t0, duration)
    if not np.isfinite(depth_stats["depth_fraction"]):
        depth_stats["depth_fraction"] = depth_fraction
        depth_stats["depth_ppm"] = depth_fraction * 1e6
    n_tr, n_full = count_transits(time, period, t0, duration)

    sde = float(getattr(res, "SDE", np.nan)) if np.isfinite(getattr(res, "SDE", np.nan)) else None
    fap = getattr(res, "FAP", None)
    try:
        fap = float(fap) if fap is not None else None
    except Exception:
        fap = None

    # Harmonic / Subharmonic check
    harmonic_flag = False
    if sde is not None and sde >= 8.0 and hasattr(res, "periods") and hasattr(res, "power"):
        # Check P/2, P/3, 2P, 3P
        harmonics = [period/2, period/3, period*2, period*3]
        for h in harmonics:
            if h < config.period_min_days or h > period_max:
                continue
            idx = np.argmin(np.abs(res.periods - h))
            if res.power[idx] > 0.8 * sde:
                harmonic_flag = True
                break

    status = _status_from_scores(float(depth_stats["snr"]), sde, n_tr, config)
    warnings_list = []
    if n_tr < config.min_transits:
        warnings_list.append("TOO_FEW_TRANSITS")
    if harmonic_flag:
        warnings_list.append("HARMONIC_DETECTED")

    candidate = CandidateSignal(
        tic_id=clean.tic_id,
        sector=clean.sector,
        candidate_id=1,
        period_days=period,
        epoch_time=t0,
        duration_days=duration,
        depth_fraction=float(depth_stats["depth_fraction"]),
        depth_ppm=float(depth_stats["depth_ppm"]),
        snr=float(depth_stats["snr"]),
        local_snr=float(depth_stats["local_snr"]),
        sde=sde,
        fap=fap,
        n_transits=n_tr,
        n_full_transits=n_full,
        n_in_transit_points=int(depth_stats["n_in_transit_points"]),
        detection_method="TLS",
        flux_source=clean.selected_flux_source,
        detrend_variant=detrend_variant,
        periodogram_peak_power=float(np.nanmax(res.power)) if hasattr(res, "power") else None,
        period_uncertainty_rough=None,
        status=status,
        warnings=warnings_list,
        extra={"period_max_days": period_max, "harmonic_flag": int(harmonic_flag)},
    )
    diag = {
        "status": "OK",
        "periods": np.asarray(getattr(res, "periods", [])),
        "power": np.asarray(getattr(res, "power", [])),
    }
    return candidate, diag


def detect_candidates(clean: CleanLightCurve, config: PipelineConfig | None = None, use_variants: bool = True) -> DetectionResult:
    config = config or PipelineConfig()
    if clean.status != "OK":
        return DetectionResult(clean.tic_id, clean.sector, status=f"PREPROCESS_STATUS_{clean.status}", warnings=clean.warnings)
    if len(clean.time) < config.min_clean_points:
        return DetectionResult(clean.tic_id, clean.sector, status="TOO_FEW_POINTS")

    all_candidates: list[CandidateSignal] = []
    diagnostics = {}

    variant_fluxes = {"default": clean.flux_detrended}
    if use_variants:
        variant_fluxes.update(clean.detrended_variants)

    for variant_name, flux in variant_fluxes.items():
        methods = [config.detection_method]
        if config.detection_method == "both":
            methods = ["bls", "tls"]
        for method in methods:
            if method == "bls":
                cand, diag = run_bls(clean, config, flux=flux, detrend_variant=variant_name)
            elif method == "tls":
                cand, diag = run_tls(clean, config, flux=flux, detrend_variant=variant_name)
            else:
                continue
            diagnostics[f"{variant_name}_{method}"] = diag
            if cand is not None:
                cand.candidate_id = len(all_candidates) + 1
                all_candidates.append(cand)

    if not all_candidates:
        return DetectionResult(clean.tic_id, clean.sector, status="NO_DETECTION", diagnostics=diagnostics)

    # Sort by status strength, then SNR/SDE. Keep all candidates because disagreement across variants is informative.
    def rank(c: CandidateSignal):
        status_score = {"STRONG_DETECTION": 3, "WEAK_DETECTION": 2, "NO_DETECTION": 1}.get(c.status, 0)
        return (status_score, np.nan_to_num(c.local_snr, nan=-999), np.nan_to_num(c.sde if c.sde is not None else np.nan, nan=-999))

    all_candidates = sorted(all_candidates, key=rank, reverse=True)
    for i, c in enumerate(all_candidates, 1):
        c.candidate_id = i

    best = all_candidates[0]
    if best.status in ("STRONG_DETECTION", "WEAK_DETECTION"):
        result_status = best.status
    else:
        result_status = "NO_STRONG_DETECTION"

    return DetectionResult(
        clean.tic_id,
        clean.sector,
        status=result_status,
        candidates=all_candidates[: config.max_candidates_per_star * max(1, len(variant_fluxes))],
        best_candidate=best,
        diagnostics=diagnostics,
    )


def candidates_to_dataframe(result: DetectionResult) -> pd.DataFrame:
    return pd.DataFrame(result.candidate_table_rows())
