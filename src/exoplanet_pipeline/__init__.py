"""Best-grade pipeline for TESS exoplanet light-curve detection and triage.

This package implements Parts 1-10 of the project:
    Part 1: ingestion, quality control, normalization, detrending, QC metrics
    Part 2: BLS/TLS candidate detection, SNR/depth estimation, diagnostics
    Part 3: first-pass transit parameter refinement and uncertainties
    Part 4: scientific vetting feature extraction
    Part 5: transparent rule-based baseline classification
    Part 6: supervised AI classification from curated feature labels
    Part 7: uncertainty estimation and confidence scoring
    Part 8: validation and injection recovery
    Part 9: sector-scale batch execution with caching/resume/failure logs
    Part 10: final candidate catalog, visual summaries, and report assets

Later validation/scaling stages should consume the candidate/fit/vetting/AI catalog rows produced here.
"""

from .config import PipelineConfig
from .schema import RawLightCurve, CleanLightCurve, CandidateSignal, DetectionResult, TransitFitResult, VettingFeatures, ClassificationResult
from .preprocess import preprocess_raw_lightcurve, preprocess_fits_file
from .detect import detect_candidates
from .fit import refine_candidate_parameters
from .vetting import extract_vetting_features
from .classify import classify_candidate_rule_based
from .ml import train_ai_classifier, predict_ai_classifier, save_model_bundle, load_model_bundle
from .cnn_views import build_cnn_candidate_views, CNNCandidateViews
from .cnn import train_cnn_classifier, predict_cnn_candidate_views, save_cnn_bundle, load_cnn_bundle
from .pipeline import run_parts_1_to_5_from_raw, run_parts_1_to_5_from_clean, run_parts_1_to_5_from_fits

__version__ = "1.0.0"

__all__ = [
    "__version__",
    "PipelineConfig",
    "RawLightCurve",
    "CleanLightCurve",
    "CandidateSignal",
    "DetectionResult",
    "TransitFitResult",
    "VettingFeatures",
    "ClassificationResult",
    "preprocess_raw_lightcurve",
    "preprocess_fits_file",
    "detect_candidates",
    "refine_candidate_parameters",
    "extract_vetting_features",
    "classify_candidate_rule_based",
    "train_ai_classifier",
    "predict_ai_classifier",
    "save_model_bundle",
    "load_model_bundle",
    "build_cnn_candidate_views",
    "CNNCandidateViews",
    "train_cnn_classifier",
    "predict_cnn_candidate_views",
    "save_cnn_bundle",
    "load_cnn_bundle",
    "run_parts_1_to_5_from_raw",
    "run_parts_1_to_5_from_clean",
    "run_parts_1_to_5_from_fits",
]

from .uncertainty import UncertaintyResult, estimate_candidate_uncertainty, add_uncertainty_columns
from .validation import ValidationReport, validate_candidate_catalog
from .batch import BatchRunConfig, run_raw_lightcurve_batch, run_fits_file_batch, discover_fits_files
from .final_catalog import harmonize_candidate_catalog, summarize_final_catalog, save_final_catalog, validate_final_catalog_schema
from .final_outputs import generate_submission_package_outputs, make_final_visual_summary, generate_three_page_report_markdown
from .pipeline_parts_1_to_10 import (
    run_single_target_parts_1_to_10_from_raw,
    run_sector_like_batch_parts_1_to_10_from_raw,
    run_sector_like_batch_parts_1_to_10_from_fits_dir,
)

__all__.extend([
    "UncertaintyResult",
    "estimate_candidate_uncertainty",
    "add_uncertainty_columns",
    "ValidationReport",
    "validate_candidate_catalog",
    "BatchRunConfig",
    "run_raw_lightcurve_batch",
    "run_fits_file_batch",
    "discover_fits_files",
    "harmonize_candidate_catalog",
    "summarize_final_catalog",
    "save_final_catalog",
    "validate_final_catalog_schema",
    "generate_submission_package_outputs",
    "make_final_visual_summary",
    "generate_three_page_report_markdown",
    "run_single_target_parts_1_to_10_from_raw",
    "run_sector_like_batch_parts_1_to_10_from_raw",
    "run_sector_like_batch_parts_1_to_10_from_fits_dir",
])
