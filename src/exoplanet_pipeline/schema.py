from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any
import numpy as np


@dataclass
class RawLightCurve:
    tic_id: int | None
    sector: int | None
    time: np.ndarray
    sap_flux: np.ndarray | None = None
    sap_flux_err: np.ndarray | None = None
    pdcsap_flux: np.ndarray | None = None
    pdcsap_flux_err: np.ndarray | None = None
    quality: np.ndarray | None = None
    centroid_col: np.ndarray | None = None
    centroid_row: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    status: str = "RAW_LOADED"
    error: str | None = None


@dataclass
class CleanLightCurve:
    tic_id: int | None
    sector: int | None
    time: np.ndarray
    flux_raw_selected: np.ndarray
    flux_normalized: np.ndarray
    flux_detrended: np.ndarray
    flux_err: np.ndarray | None
    trend: np.ndarray | None
    selected_flux_source: str
    finite_mask: np.ndarray
    quality_mask: np.ndarray
    outlier_mask: np.ndarray
    final_mask: np.ndarray
    negative_outlier_flag: np.ndarray
    centroid_col: np.ndarray | None
    centroid_row: np.ndarray | None
    metadata: dict[str, Any] = field(default_factory=dict)
    qc: dict[str, Any] = field(default_factory=dict)
    detrended_variants: dict[str, np.ndarray] = field(default_factory=dict)
    status: str = "OK"
    warnings: list[str] = field(default_factory=list)
    flux_detrended_pass1: np.ndarray | None = None


@dataclass
class CandidateSignal:
    tic_id: int | None
    sector: int | None
    candidate_id: int
    period_days: float
    epoch_time: float
    duration_days: float
    depth_fraction: float
    depth_ppm: float
    snr: float
    local_snr: float
    sde: float | None
    fap: float | None
    n_transits: int
    n_full_transits: int
    n_in_transit_points: int
    detection_method: str
    flux_source: str
    detrend_variant: str
    periodogram_peak_power: float | None = None
    period_uncertainty_rough: float | None = None
    status: str = "DETECTED"
    warnings: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["warnings"] = ";".join(self.warnings)
        return d


@dataclass
class DetectionResult:
    tic_id: int | None
    sector: int | None
    status: str
    candidates: list[CandidateSignal] = field(default_factory=list)
    best_candidate: CandidateSignal | None = None
    warnings: list[str] = field(default_factory=list)
    error: str | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def candidate_table_rows(self) -> list[dict[str, Any]]:
        return [c.to_dict() for c in self.candidates]


@dataclass
class TransitFitResult:
    tic_id: int | None
    sector: int | None
    candidate_id: int
    period_days: float
    period_err_days: float
    epoch_time: float
    epoch_err_days: float
    duration_days: float
    duration_err_days: float
    depth_fraction: float
    depth_err_fraction: float
    depth_ppm: float
    depth_err_ppm: float
    rp_over_rstar: float
    rp_earth: float
    stellar_radius_rsun: float
    snr: float
    n_in_transit_points: int
    n_events: int
    n_good_events: int
    event_depth_scatter_ppm: float
    method: str = "box_profile_grid_refinement"
    warnings: list[str] = field(default_factory=list)
    event_depths: list[dict[str, Any]] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["warnings"] = ";".join(self.warnings)
        # Keep event table out of flat catalog rows.
        d.pop("event_depths", None)
        return d


@dataclass
class VettingFeatures:
    tic_id: int | None
    sector: int | None
    candidate_id: int
    odd_depth_ppm: float
    even_depth_ppm: float
    odd_even_sigma: float
    odd_even_depth_diff_ppm: float
    secondary_depth_ppm: float
    secondary_sigma: float
    secondary_phase: float
    secondary_to_primary_ratio: float
    centroid_shift_pix: float
    centroid_shift_sigma: float
    crowdsap: float | None
    flfrcsap: float | None
    crowding_risk: float | None
    corrected_depth_ppm: float
    v_shape_score: float
    transit_asymmetry: float
    out_of_transit_rms_ppm: float
    red_noise_proxy: float
    harmonic_risk: float
    data_quality_score: float
    warnings: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["warnings"] = ";".join(self.warnings)
        return d


@dataclass
class ClassificationResult:
    tic_id: int | None
    sector: int | None
    candidate_id: int
    predicted_class: str
    confidence: float
    planet_score: float
    eb_score: float
    blend_score: float
    stellar_variability_score: float
    systematic_score: float
    evidence: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["evidence"] = ";".join(self.evidence)
        d["warnings"] = ";".join(self.warnings)
        return d
