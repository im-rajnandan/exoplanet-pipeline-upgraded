from __future__ import annotations

from dataclasses import asdict
import numpy as np
import pandas as pd

from .schema import CleanLightCurve, CandidateSignal, TransitFitResult, VettingFeatures
from .detect import make_transit_mask
from .fit import phase_fold_time
from .utils import robust_sigma


def _nanmedian_or_nan(x) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.nanmedian(x)) if x.size else np.nan


def _safe_div(a: float, b: float) -> float:
    return float(a / b) if np.isfinite(a) and np.isfinite(b) and b != 0 else np.nan


def odd_even_test(fit: TransitFitResult) -> dict[str, float]:
    events = fit.event_depths or []
    depths = []
    errs = []
    ids = []
    for row in events:
        d = row.get("depth_fraction", np.nan)
        e = row.get("depth_err_fraction", np.nan)
        i = row.get("event_id", np.nan)
        if np.isfinite(d):
            depths.append(float(d))
            errs.append(float(e) if np.isfinite(e) and e > 0 else np.nan)
            ids.append(int(i) if np.isfinite(i) else len(ids))
    if len(depths) < 2:
        return {
            "odd_depth_ppm": np.nan,
            "even_depth_ppm": np.nan,
            "odd_even_sigma": 0.0,
            "odd_even_depth_diff_ppm": np.nan,
        }
    depths = np.asarray(depths)
    errs = np.asarray(errs)
    ids = np.asarray(ids)
    odd = depths[(ids % 2) != 0]
    even = depths[(ids % 2) == 0]
    odd_errs = errs[(ids % 2) != 0]
    even_errs = errs[(ids % 2) == 0]
    if odd.size == 0 or even.size == 0:
        return {
            "odd_depth_ppm": np.nan,
            "even_depth_ppm": np.nan,
            "odd_even_sigma": 0.0,
            "odd_even_depth_diff_ppm": np.nan,
        }
    odd_depth = float(np.nanmedian(odd))
    even_depth = float(np.nanmedian(even))
    # Robust group errors: combine measurement error if present with group scatter.
    odd_group_err = robust_sigma(odd) / np.sqrt(max(odd.size, 1)) if odd.size > 1 else np.nan
    even_group_err = robust_sigma(even) / np.sqrt(max(even.size, 1)) if even.size > 1 else np.nan
    odd_meas = np.nanmedian(odd_errs[np.isfinite(odd_errs)]) if np.any(np.isfinite(odd_errs)) else np.nan
    even_meas = np.nanmedian(even_errs[np.isfinite(even_errs)]) if np.any(np.isfinite(even_errs)) else np.nan
    odd_total_err = np.nanmax([odd_group_err, odd_meas]) if np.any(np.isfinite([odd_group_err, odd_meas])) else np.nan
    even_total_err = np.nanmax([even_group_err, even_meas]) if np.any(np.isfinite([even_group_err, even_meas])) else np.nan
    denom = np.sqrt(odd_total_err**2 + even_total_err**2) if np.isfinite(odd_total_err) and np.isfinite(even_total_err) else robust_sigma(depths)
    sigma = abs(odd_depth - even_depth) / denom if np.isfinite(denom) and denom > 0 else 0.0
    return {
        "odd_depth_ppm": float(odd_depth * 1e6),
        "even_depth_ppm": float(even_depth * 1e6),
        "odd_even_sigma": float(sigma),
        "odd_even_depth_diff_ppm": float(abs(odd_depth - even_depth) * 1e6),
    }


def secondary_eclipse_test(time: np.ndarray, flux: np.ndarray, period: float, t0: float, duration: float, primary_depth: float) -> dict[str, float]:
    """Search for the strongest non-primary dip, emphasizing phase 0.5.

    This fixes the common bug where phase zero is accidentally reused as the
    secondary window. Phase 0 is primary; phase 0.5 is the circular-orbit
    secondary location. We also scan nearby phases for eccentric binaries.
    """
    time = np.asarray(time, dtype=float)
    flux = np.asarray(flux, dtype=float)
    finite = np.isfinite(time) & np.isfinite(flux)
    time = time[finite]
    flux = flux[finite]
    if len(time) < 50 or not np.isfinite(period) or period <= 0 or not np.isfinite(duration) or duration <= 0:
        return {"secondary_depth_ppm": np.nan, "secondary_sigma": 0.0, "secondary_phase": np.nan, "secondary_to_primary_ratio": np.nan}

    phase = ((time - t0) / period) % 1.0
    width = max(duration / period, 1e-4)
    primary_dist = np.minimum(phase, 1.0 - phase)
    not_primary = primary_dist > 2.0 * width

    best_sigma = -np.inf
    best_depth = np.nan
    best_phase = np.nan
    # Include exact 0.5 and a phase scan for eccentric secondaries.
    search_phases = np.unique(np.concatenate([[0.5], np.linspace(0.15, 0.85, 141)]))
    for ph0 in search_phases:
        dist = np.abs(phase - ph0)
        # Circular wrap does not matter for ph0 within 0.15..0.85.
        in_sec = dist < 0.5 * width
        near_sec = dist < max(3.0 * width, 0.04)
        out_sec = near_sec & (~in_sec) & not_primary
        if in_sec.sum() < 3 or out_sec.sum() < 10:
            continue
        baseline = np.nanmedian(flux[out_sec])
        depth = baseline - np.nanmedian(flux[in_sec])
        noise = robust_sigma(flux[out_sec] - np.nanmedian(flux[out_sec]))
        sigma = depth / noise * np.sqrt(in_sec.sum()) if np.isfinite(depth) and depth > 0 and np.isfinite(noise) and noise > 0 else 0.0
        if sigma > best_sigma:
            best_sigma = sigma
            best_depth = depth
            best_phase = ph0

    if not np.isfinite(best_sigma) or best_sigma < 0:
        best_sigma = 0.0
    return {
        "secondary_depth_ppm": float(best_depth * 1e6) if np.isfinite(best_depth) else np.nan,
        "secondary_sigma": float(best_sigma),
        "secondary_phase": float(best_phase) if np.isfinite(best_phase) else np.nan,
        "secondary_to_primary_ratio": _safe_div(best_depth, primary_depth),
    }


def _rolling_median_residual(time: np.ndarray, values: np.ndarray, window_days: float = 1.0) -> np.ndarray:
    import pandas as pd
    time = np.asarray(time, dtype=float)
    values = np.asarray(values, dtype=float)
    if len(values) < 5:
        return values - np.nanmedian(values)
    dt = np.nanmedian(np.diff(np.sort(time)))
    if not np.isfinite(dt) or dt <= 0:
        win = min(101, max(5, len(values) // 10))
    else:
        win = max(5, int(round(window_days / dt)))
    if win % 2 == 0:
        win += 1
    trend = pd.Series(values).rolling(win, center=True, min_periods=max(3, win // 5)).median().to_numpy()
    good = np.isfinite(trend)
    if good.sum() >= 2:
        trend = np.interp(np.arange(len(trend)), np.where(good)[0], trend[good])
    else:
        trend = np.ones_like(values) * np.nanmedian(values)
    return values - trend


def centroid_shift_test(clean: CleanLightCurve, period: float, t0: float, duration: float) -> dict[str, float]:
    if clean.centroid_col is None or clean.centroid_row is None:
        return {"centroid_shift_pix": np.nan, "centroid_shift_sigma": 0.0}
    time = np.asarray(clean.time, dtype=float)
    col = np.asarray(clean.centroid_col, dtype=float)
    row = np.asarray(clean.centroid_row, dtype=float)
    finite = np.isfinite(time) & np.isfinite(col) & np.isfinite(row)
    time, col, row = time[finite], col[finite], row[finite]
    if len(time) < 50:
        return {"centroid_shift_pix": np.nan, "centroid_shift_sigma": 0.0}
    col_res = _rolling_median_residual(time, col, window_days=1.0)
    row_res = _rolling_median_residual(time, row, window_days=1.0)
    in_tr = make_transit_mask(time, period, t0, duration, width_factor=1.0)
    out_tr = ~make_transit_mask(time, period, t0, duration, width_factor=5.0)
    if in_tr.sum() < 3 or out_tr.sum() < 20:
        return {"centroid_shift_pix": np.nan, "centroid_shift_sigma": 0.0}
    dc = np.nanmedian(col_res[in_tr]) - np.nanmedian(col_res[out_tr])
    dr = np.nanmedian(row_res[in_tr]) - np.nanmedian(row_res[out_tr])
    shift = float(np.sqrt(dc * dc + dr * dr))
    sig_c = robust_sigma(col_res[out_tr]) / np.sqrt(in_tr.sum())
    sig_r = robust_sigma(row_res[out_tr]) / np.sqrt(in_tr.sum())
    zc = dc / sig_c if np.isfinite(sig_c) and sig_c > 0 else 0.0
    zr = dr / sig_r if np.isfinite(sig_r) and sig_r > 0 else 0.0
    shift_sigma = float(np.sqrt(zc * zc + zr * zr))
    return {"centroid_shift_pix": shift, "centroid_shift_sigma": shift_sigma}


def shape_features(time: np.ndarray, flux: np.ndarray, period: float, t0: float, duration: float, depth: float) -> dict[str, float]:
    folded = phase_fold_time(time, period, t0)
    finite = np.isfinite(folded) & np.isfinite(flux)
    folded = folded[finite]
    y = np.asarray(flux, dtype=float)[finite]
    if len(folded) < 50 or not np.isfinite(depth) or depth <= 0:
        return {"v_shape_score": np.nan, "transit_asymmetry": np.nan}

    center = np.abs(folded) < 0.25 * duration
    shoulder = (np.abs(folded) >= 0.25 * duration) & (np.abs(folded) < 0.5 * duration)
    left = (folded >= -0.5 * duration) & (folded < 0)
    right = (folded > 0) & (folded <= 0.5 * duration)
    if center.sum() < 3 or shoulder.sum() < 3:
        v_shape = np.nan
    else:
        center_depth = 1.0 - np.nanmedian(y[center])
        shoulder_depth = 1.0 - np.nanmedian(y[shoulder])
        # This simple score is closer to 1 when the event has a flat bottom
        # across the central/shoulder region, and lower when the shape is more
        # triangular or poorly resolved. It is a rough morphology proxy only.
        v_shape = _safe_div(shoulder_depth, center_depth)
    if left.sum() >= 3 and right.sum() >= 3:
        asym = abs(np.nanmedian(y[left]) - np.nanmedian(y[right])) / depth
    else:
        asym = np.nan
    return {"v_shape_score": float(v_shape), "transit_asymmetry": float(asym)}


def red_noise_proxy(flux: np.ndarray, max_lag: int = 20) -> float:
    y = np.asarray(flux, dtype=float)
    y = y[np.isfinite(y)]
    if len(y) < max_lag + 5:
        return np.nan
    y = y - np.nanmedian(y)
    denom = np.nansum(y * y)
    if denom <= 0 or not np.isfinite(denom):
        return np.nan
    acfs = []
    for lag in range(1, max_lag + 1):
        acfs.append(np.nansum(y[:-lag] * y[lag:]) / denom)
    return float(np.nanmedian(np.abs(acfs)))


def data_quality_score(clean: CleanLightCurve) -> float:
    removed = clean.qc.get("removed_fraction", np.nan)
    noise = clean.qc.get("robust_noise_ppm", np.nan)
    baseline = clean.qc.get("baseline_days", np.nan)
    score = 1.0
    if np.isfinite(removed):
        score -= min(max(removed, 0.0), 1.0) * 0.35
    if np.isfinite(noise):
        # Soft penalty above 1000 ppm, stronger above 3000 ppm.
        score -= min(max((noise - 1000.0) / 5000.0, 0.0), 0.4)
    if np.isfinite(baseline) and baseline < 10:
        score -= 0.2
    return float(np.clip(score, 0.0, 1.0))


def extract_vetting_features(clean: CleanLightCurve, candidate: CandidateSignal, fit: TransitFitResult) -> VettingFeatures:
    time = np.asarray(clean.time, dtype=float)
    flux = np.asarray(clean.flux_detrended, dtype=float)
    odd_even = odd_even_test(fit)
    secondary = secondary_eclipse_test(time, flux, fit.period_days, fit.epoch_time, fit.duration_days, fit.depth_fraction)
    centroid = centroid_shift_test(clean, fit.period_days, fit.epoch_time, fit.duration_days)
    shape = shape_features(time, flux, fit.period_days, fit.epoch_time, fit.duration_days, fit.depth_fraction)

    crowdsap = clean.qc.get("crowdsap", None)
    flfrcsap = clean.qc.get("flfrcsap", None)
    try:
        crowdsap_f = float(crowdsap) if crowdsap is not None else np.nan
    except Exception:
        crowdsap_f = np.nan
    try:
        flfrcsap_f = float(flfrcsap) if flfrcsap is not None else np.nan
    except Exception:
        flfrcsap_f = np.nan
    crowding_risk = float(1.0 - crowdsap_f) if np.isfinite(crowdsap_f) else None
    corrected_depth = fit.depth_fraction / crowdsap_f if np.isfinite(crowdsap_f) and crowdsap_f > 0 else np.nan

    in_tr = make_transit_mask(time, fit.period_days, fit.epoch_time, fit.duration_days, width_factor=5.0)
    out_flux = flux[~in_tr]
    out_rms = float(np.sqrt(np.nanmean((out_flux - np.nanmedian(out_flux)) ** 2)) * 1e6) if out_flux.size else np.nan
    red = red_noise_proxy(out_flux)

    harmonic = 0.0
    if candidate.extra:
        # If future detection module adds alias ratios, they can be used here.
        harmonic = float(candidate.extra.get("harmonic_risk", 0.0) or 0.0)
    warnings = []
    if np.isfinite(crowdsap_f) and crowdsap_f < 0.7:
        warnings.append("LOW_CROWDSAP_HIGH_CONTAMINATION_RISK")
    if centroid["centroid_shift_sigma"] >= 5:
        warnings.append("CENTROID_SHIFT_SIGNIFICANT")
    if secondary["secondary_sigma"] >= 5:
        warnings.append("SECONDARY_ECLIPSE_SIGNIFICANT")
    if odd_even["odd_even_sigma"] >= 3:
        warnings.append("ODD_EVEN_DEPTH_MISMATCH")

    return VettingFeatures(
        tic_id=clean.tic_id,
        sector=clean.sector,
        candidate_id=candidate.candidate_id,
        odd_depth_ppm=odd_even["odd_depth_ppm"],
        even_depth_ppm=odd_even["even_depth_ppm"],
        odd_even_sigma=odd_even["odd_even_sigma"],
        odd_even_depth_diff_ppm=odd_even["odd_even_depth_diff_ppm"],
        secondary_depth_ppm=secondary["secondary_depth_ppm"],
        secondary_sigma=secondary["secondary_sigma"],
        secondary_phase=secondary["secondary_phase"],
        secondary_to_primary_ratio=secondary["secondary_to_primary_ratio"],
        centroid_shift_pix=centroid["centroid_shift_pix"],
        centroid_shift_sigma=centroid["centroid_shift_sigma"],
        crowdsap=float(crowdsap_f) if np.isfinite(crowdsap_f) else None,
        flfrcsap=float(flfrcsap_f) if np.isfinite(flfrcsap_f) else None,
        crowding_risk=crowding_risk,
        corrected_depth_ppm=float(corrected_depth * 1e6) if np.isfinite(corrected_depth) else np.nan,
        v_shape_score=shape["v_shape_score"],
        transit_asymmetry=shape["transit_asymmetry"],
        out_of_transit_rms_ppm=out_rms,
        red_noise_proxy=red,
        harmonic_risk=harmonic,
        data_quality_score=data_quality_score(clean),
        warnings=warnings,
        extra={
            "primary_depth_ppm": fit.depth_ppm,
            "candidate_status": candidate.status,
            "candidate_snr": candidate.local_snr,
            "candidate_sde": candidate.sde,
        },
    )


def vetting_to_dataframe(features: VettingFeatures) -> pd.DataFrame:
    return pd.DataFrame([features.to_dict()])
