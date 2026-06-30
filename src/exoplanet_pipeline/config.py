from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


QualityMaskMode = Literal["none", "minimal", "conservative", "strict"]
DetectionMethod = Literal["bls", "tls", "both"]


@dataclass
class PipelineConfig:
    """Central configuration for Parts 1 and 2.

    The defaults are intentionally conservative: we prefer PDCSAP, never silently
    synthesize data when real downloads fail, preserve raw/normalized/detrended
    arrays, and record all quality-control decisions.
    """

    # Paths
    project_root: Path = Path(".")
    raw_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")
    metadata_dir: Path = Path("data/metadata")
    plot_dir: Path = Path("plots")

    # Ingestion / flux choice
    preferred_flux: Literal["PDCSAP", "SAP"] = "PDCSAP"
    allow_sap_fallback: bool = True
    allow_synthetic_fallback: bool = False
    min_valid_flux_fraction: float = 0.50
    min_clean_points: int = 500
    max_removed_fraction: float = 0.60

    # Quality flags
    quality_mask_mode: QualityMaskMode = "conservative"

    # Normalization and outliers
    normalize_method: Literal["median"] = "median"
    positive_clip_sigma: float = 7.0
    negative_clip_sigma: float = 15.0
    remove_extreme_negative_outliers: bool = False

    # Detrending
    detrend_method: Literal["rolling_median", "wotan_biweight", "none"] = "rolling_median"
    detrend_window_days: float = 1.0
    detrend_variants_days: tuple[float, ...] = (0.5, 1.0, 2.0)

    # Detection bounds
    detection_method: DetectionMethod = "bls"
    period_min_days: float = 0.20
    period_max_days: float | None = None  # if None, use min(13.5, baseline/2)
    n_periods: int = 4000
    period_grid_mode: Literal["linear", "frequency"] = "linear"
    min_transits: int = 2
    min_duration_days: float = 0.02      # about 29 minutes
    max_duration_days: float = 0.30      # about 7.2 hours for first-pass search
    n_durations: int = 12

    # Detection thresholds
    strong_snr_threshold: float = 10.0
    weak_snr_threshold: float = 7.0
    strong_sde_threshold: float = 8.0
    weak_sde_threshold: float = 6.0
    max_candidates_per_star: int = 3
    transit_mask_width_factor: float = 1.5
    detection_use_variants: bool = True

    # Diagnostics / persistence
    save_intermediate: bool = True
    make_plots: bool = True
    random_seed: int = 42

    def resolve_paths(self) -> None:
        """Create configured directories if they do not exist."""
        for p in [self.raw_dir, self.processed_dir, self.metadata_dir, self.plot_dir]:
            Path(p).mkdir(parents=True, exist_ok=True)
