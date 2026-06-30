from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .detect import transit_event_numbers
from .schema import CleanLightCurve, CandidateSignal, TransitFitResult, VettingFeatures
from .utils import robust_sigma


GLOBAL_VIEW_BINS = 1001
LOCAL_VIEW_BINS = 401

GLOBAL_VIEW_NAMES: tuple[str, ...] = ("global_flux",)
LOCAL_VIEW_NAMES: tuple[str, ...] = (
    "local_flux",
    "secondary_flux",
    "odd_flux",
    "even_flux",
    "centroid_col_local",
    "centroid_row_local",
)
CNN_VIEW_NAMES: tuple[str, ...] = GLOBAL_VIEW_NAMES + LOCAL_VIEW_NAMES

CNN_SCALAR_FEATURE_COLUMNS: tuple[str, ...] = (
    "period_days",
    "duration_days",
    "depth_fraction",
    "depth_ppm",
    "snr",
    "local_snr",
    "sde",
    "n_transits",
    "n_full_transits",
    "n_in_transit_points",
    "fit_period_days",
    "fit_duration_days",
    "fit_depth_fraction",
    "fit_depth_ppm",
    "fit_snr",
    "fit_n_events",
    "fit_n_good_events",
    "fit_event_depth_scatter_ppm",
    "vet_crowdsap",
    "vet_crowding_risk",
    "vet_odd_even_sigma",
    "vet_secondary_sigma",
    "vet_centroid_shift_sigma",
    "vet_red_noise_proxy",
    "vet_data_quality_score",
)


@dataclass
class CNNCandidateViews:
    """Fixed-size CNN inputs for one detected candidate.

    Each view has shape ``(2, n_bins)``: channel 0 is the normalized value and
    channel 1 is a validity mask. Missing bins and unavailable centroid views
    therefore remain explicit to the model.
    """

    views: dict[str, np.ndarray]
    scalar_features: dict[str, float]
    metadata: dict[str, Any]

    def global_tensor(self) -> np.ndarray:
        return self.views["global_flux"].astype(np.float32, copy=False)

    def local_tensor(self) -> np.ndarray:
        return np.stack([self.views[name] for name in LOCAL_VIEW_NAMES], axis=0).astype(np.float32, copy=False)

    def scalar_vector(self, feature_names: tuple[str, ...] | list[str] = CNN_SCALAR_FEATURE_COLUMNS) -> np.ndarray:
        return np.asarray([self.scalar_features.get(name, np.nan) for name in feature_names], dtype=np.float32)


def build_cnn_candidate_views(
    clean: CleanLightCurve,
    candidate: CandidateSignal,
    fit: TransitFitResult,
    vet: VettingFeatures,
    *,
    global_bins: int = GLOBAL_VIEW_BINS,
    local_bins: int = LOCAL_VIEW_BINS,
) -> CNNCandidateViews:
    """Build deterministic fixed-size CNN views for a fitted candidate."""
    time = np.asarray(clean.time, dtype=float)
    flux = np.asarray(clean.flux_detrended, dtype=float)
    finite = np.isfinite(time) & np.isfinite(flux)
    time = time[finite]
    flux = flux[finite]

    period = _safe_float(fit.period_days, candidate.period_days)
    epoch = _safe_float(fit.epoch_time, candidate.epoch_time)
    duration = _safe_float(fit.duration_days, candidate.duration_days)
    duration_phase = duration / period if np.isfinite(duration) and np.isfinite(period) and period > 0 else np.nan
    local_half_width = _local_half_width(duration_phase)

    phase_centered = _phase_delta(time, period, epoch, center_phase=0.0)
    oot = np.abs(phase_centered) > max(2.0 * local_half_width, 3.0 * _finite_or(duration_phase, 0.01))
    baseline = _nanmedian_or_default(flux[oot], _nanmedian_or_default(flux, 1.0))
    scatter = robust_sigma(flux[oot] - baseline) if np.sum(oot) >= 20 else robust_sigma(flux - baseline)
    scale = scatter if np.isfinite(scatter) and scatter > 0 else max(abs(_safe_float(fit.depth_fraction, candidate.depth_fraction)), 1e-6)
    flux_values = np.clip((flux - baseline) / scale, -20.0, 20.0)

    views: dict[str, np.ndarray] = {}
    views["global_flux"] = _bin_view(phase_centered, flux_values, np.linspace(-0.5, 0.5, global_bins + 1))
    views["local_flux"] = _window_view(time, flux_values, period, epoch, 0.0, local_half_width, local_bins)
    views["secondary_flux"] = _window_view(time, flux_values, period, epoch, 0.5, local_half_width, local_bins)

    if len(time):
        event_ids = transit_event_numbers(time, period, epoch) if np.isfinite(period) and period > 0 else np.zeros(len(time), dtype=int)
        odd = (event_ids % 2) != 0
        even = (event_ids % 2) == 0
    else:
        odd = even = np.zeros(0, dtype=bool)
    views["odd_flux"] = _window_view(time[odd], flux_values[odd], period, epoch, 0.0, local_half_width, local_bins)
    views["even_flux"] = _window_view(time[even], flux_values[even], period, epoch, 0.0, local_half_width, local_bins)

    views["centroid_col_local"] = _centroid_view(clean.centroid_col, clean.time, finite, period, epoch, local_half_width, local_bins)
    views["centroid_row_local"] = _centroid_view(clean.centroid_row, clean.time, finite, period, epoch, local_half_width, local_bins)

    scalars = scalar_features_from_candidate(candidate, fit, vet)
    metadata = {
        "tic_id": candidate.tic_id,
        "sector": candidate.sector,
        "candidate_id": candidate.candidate_id,
        "global_bins": int(global_bins),
        "local_bins": int(local_bins),
        "local_half_width_phase": float(local_half_width),
        "flux_value_mode": "robust_zscore",
    }
    return CNNCandidateViews(views=views, scalar_features=scalars, metadata=metadata)


def scalar_features_from_candidate(
    candidate: CandidateSignal,
    fit: TransitFitResult,
    vet: VettingFeatures,
) -> dict[str, float]:
    """Return catalog-style scalar features for the CNN scalar branch."""
    return {
        "period_days": _safe_float(candidate.period_days),
        "duration_days": _safe_float(candidate.duration_days),
        "depth_fraction": _safe_float(candidate.depth_fraction),
        "depth_ppm": _safe_float(candidate.depth_ppm),
        "snr": _safe_float(candidate.snr),
        "local_snr": _safe_float(candidate.local_snr),
        "sde": _safe_float(candidate.sde),
        "n_transits": _safe_float(candidate.n_transits),
        "n_full_transits": _safe_float(candidate.n_full_transits),
        "n_in_transit_points": _safe_float(candidate.n_in_transit_points),
        "fit_period_days": _safe_float(fit.period_days),
        "fit_duration_days": _safe_float(fit.duration_days),
        "fit_depth_fraction": _safe_float(fit.depth_fraction),
        "fit_depth_ppm": _safe_float(fit.depth_ppm),
        "fit_snr": _safe_float(fit.snr),
        "fit_n_events": _safe_float(fit.n_events),
        "fit_n_good_events": _safe_float(fit.n_good_events),
        "fit_event_depth_scatter_ppm": _safe_float(fit.event_depth_scatter_ppm),
        "vet_crowdsap": _safe_float(vet.crowdsap),
        "vet_crowding_risk": _safe_float(vet.crowding_risk),
        "vet_odd_even_sigma": _safe_float(vet.odd_even_sigma),
        "vet_secondary_sigma": _safe_float(vet.secondary_sigma),
        "vet_centroid_shift_sigma": _safe_float(vet.centroid_shift_sigma),
        "vet_red_noise_proxy": _safe_float(vet.red_noise_proxy),
        "vet_data_quality_score": _safe_float(vet.data_quality_score),
    }


def scalar_features_from_catalog_row(row: Any) -> dict[str, float]:
    """Extract default CNN scalar features from a pandas row or mapping."""
    return {name: _safe_float(_row_get(row, name)) for name in CNN_SCALAR_FEATURE_COLUMNS}


def save_cnn_example_npz(example: CNNCandidateViews, path: str | Path, *, canonical_label: str | None = None, binary_label: str | None = None) -> None:
    """Write one CNN example to an ``.npz`` file used by the training scripts."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "global_flux": example.views["global_flux"].astype(np.float32),
        "local_views": example.local_tensor().astype(np.float32),
        "scalar_features": example.scalar_vector().astype(np.float32),
        "scalar_feature_names": np.asarray(CNN_SCALAR_FEATURE_COLUMNS),
        "metadata": np.asarray([example.metadata], dtype=object),
    }
    if canonical_label is not None:
        payload["canonical_label"] = np.asarray(str(canonical_label))
    if binary_label is not None:
        payload["binary_label"] = np.asarray(str(binary_label))
    np.savez_compressed(path, **payload)


def _window_view(
    time: np.ndarray,
    values: np.ndarray,
    period: float,
    epoch: float,
    center_phase: float,
    half_width_phase: float,
    bins: int,
) -> np.ndarray:
    if len(time) == 0:
        return _empty_view(bins)
    delta = _phase_delta(time, period, epoch, center_phase=center_phase)
    edges = np.linspace(-half_width_phase, half_width_phase, bins + 1)
    return _bin_view(delta, values, edges)


def _centroid_view(
    centroid: np.ndarray | None,
    original_time: np.ndarray,
    finite_flux_mask: np.ndarray,
    period: float,
    epoch: float,
    half_width_phase: float,
    bins: int,
) -> np.ndarray:
    if centroid is None:
        return _empty_view(bins)
    centroid_arr = np.asarray(centroid, dtype=float)
    time_arr = np.asarray(original_time, dtype=float)
    if centroid_arr.shape != time_arr.shape:
        return _empty_view(bins)
    finite = finite_flux_mask & np.isfinite(centroid_arr)
    if finite.sum() < 5:
        return _empty_view(bins)
    values = centroid_arr[finite]
    values = values - _nanmedian_or_default(values, 0.0)
    scale = robust_sigma(values)
    if not np.isfinite(scale) or scale <= 0:
        scale = np.nanstd(values)
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    values = np.clip(values / scale, -20.0, 20.0)
    return _window_view(time_arr[finite], values, period, epoch, 0.0, half_width_phase, bins)


def _bin_view(x: np.ndarray, values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    bins = len(edges) - 1
    out = np.zeros((2, bins), dtype=np.float32)
    x = np.asarray(x, dtype=float)
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(x) & np.isfinite(values)
    if not np.any(finite):
        return out
    idx = np.searchsorted(edges, x[finite], side="right") - 1
    idx = np.clip(idx, 0, bins - 1)
    vals = values[finite]
    for i in np.unique(idx):
        in_bin = idx == i
        if np.any(in_bin):
            out[0, i] = float(np.nanmedian(vals[in_bin]))
            out[1, i] = 1.0
    return out


def _phase_delta(time: np.ndarray, period: float, epoch: float, *, center_phase: float) -> np.ndarray:
    time = np.asarray(time, dtype=float)
    if not np.isfinite(period) or period <= 0 or not np.isfinite(epoch):
        return np.full_like(time, np.nan, dtype=float)
    phase = ((time - epoch) / period) % 1.0
    return ((phase - center_phase + 0.5) % 1.0) - 0.5


def _local_half_width(duration_phase: float) -> float:
    if not np.isfinite(duration_phase) or duration_phase <= 0:
        duration_phase = 0.02
    return float(np.clip(max(4.0 * duration_phase, 0.03), 0.03, 0.20))


def _empty_view(bins: int) -> np.ndarray:
    return np.zeros((2, bins), dtype=np.float32)


def _safe_float(x: Any, default: float = np.nan) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) else float(default)
    except Exception:
        return float(default)


def _finite_or(x: float, default: float) -> float:
    return float(x) if np.isfinite(x) else float(default)


def _nanmedian_or_default(x: np.ndarray, default: float) -> float:
    x = np.asarray(x, dtype=float)
    finite = x[np.isfinite(x)]
    return float(np.nanmedian(finite)) if finite.size else float(default)


def _row_get(row: Any, key: str) -> Any:
    if hasattr(row, "get"):
        return row.get(key, np.nan)
    try:
        return row[key]
    except Exception:
        return np.nan
