from __future__ import annotations

from dataclasses import asdict
import pandas as pd

from .config import PipelineConfig
from .schema import RawLightCurve, CleanLightCurve, DetectionResult, TransitFitResult, VettingFeatures, ClassificationResult
from .preprocess import preprocess_raw_lightcurve, preprocess_fits_file
from .detect import detect_candidates
from .fit import refine_candidate_parameters
from .vetting import extract_vetting_features
from .classify import classify_candidate_rule_based


def run_parts_1_to_5_from_clean(clean: CleanLightCurve, config: PipelineConfig | None = None):
    config = config or PipelineConfig()
    # Pass 1 Detection
    detection = detect_candidates(clean, config=config, use_variants=config.detection_use_variants)
    
    # Pass 2: Iterative Transit-Masked Detrending
    if detection.best_candidate is not None and detection.best_candidate.status in ("STRONG_DETECTION", "WEAK_DETECTION"):
        from .preprocess import redetrend_with_mask
        best = detection.best_candidate
        clean = redetrend_with_mask(
            clean,
            period=best.period_days,
            t0=best.epoch_time,
            duration=best.duration_days,
            config=config
        )
        # Re-run detection on the new clean light curve
        detection = detect_candidates(clean, config=config, use_variants=config.detection_use_variants)

    rows = []
    fitted_candidates = []
    fit_results: list[TransitFitResult] = []
    vetting_results: list[VettingFeatures] = []
    class_results: list[ClassificationResult] = []
    for cand in detection.candidates:
        if cand.status not in ("STRONG_DETECTION", "WEAK_DETECTION"):
            continue
        fit = refine_candidate_parameters(clean, cand)
        vet = extract_vetting_features(clean, cand, fit)
        cls = classify_candidate_rule_based(cand, fit, vet)
        fitted_candidates.append(cand)
        fit_results.append(fit)
        vetting_results.append(vet)
        class_results.append(cls)
        row = {}
        row.update(cand.to_dict())
        row.update({f"fit_{k}": v for k, v in fit.to_dict().items() if k not in ("tic_id", "sector", "candidate_id")})
        row.update({f"vet_{k}": v for k, v in vet.to_dict().items() if k not in ("tic_id", "sector", "candidate_id")})
        row.update({f"class_{k}": v for k, v in cls.to_dict().items() if k not in ("tic_id", "sector", "candidate_id")})
        rows.append(row)
    catalog = pd.DataFrame(rows)
    return {
        "clean": clean,
        "detection": detection,
        "fitted_candidates": fitted_candidates,
        "fit_results": fit_results,
        "vetting_results": vetting_results,
        "classification_results": class_results,
        "catalog": catalog,
    }


def run_parts_1_to_5_from_raw(raw: RawLightCurve, config: PipelineConfig | None = None):
    config = config or PipelineConfig()
    clean = preprocess_raw_lightcurve(raw, config=config)
    return run_parts_1_to_5_from_clean(clean, config=config)


def run_parts_1_to_5_from_fits(file_path: str, config: PipelineConfig | None = None):
    config = config or PipelineConfig()
    clean = preprocess_fits_file(file_path, config=config)
    return run_parts_1_to_5_from_clean(clean, config=config)
