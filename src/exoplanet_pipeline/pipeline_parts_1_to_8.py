from __future__ import annotations

import pandas as pd

from .config import PipelineConfig
from .schema import RawLightCurve, CleanLightCurve
from .pipeline import run_parts_1_to_5_from_raw, run_parts_1_to_5_from_clean, run_parts_1_to_5_from_fits
from .ml import predict_ai_classifier
from .cnn import predict_cnn_candidate_views
from .cnn_views import build_cnn_candidate_views
from .uncertainty import estimate_candidate_uncertainty


def attach_ai_and_uncertainty(
    parts_1_to_5_result: dict,
    model_bundle: dict | None = None,
    cnn_bundle: dict | None = None,
    config: PipelineConfig | None = None,
) -> dict:
    """Connect Parts 1–6 with Part 7 uncertainty in one result object."""
    catalog = parts_1_to_5_result["catalog"].copy()
    fitted_candidates = parts_1_to_5_result.get("fitted_candidates")
    if fitted_candidates is None:
        # Backward-compatible fallback for older result dictionaries. New pipeline
        # results include fitted_candidates explicitly because detection.candidates
        # may contain non-fitted rows.
        fitted_candidates = [
            c
            for c in parts_1_to_5_result["detection"].candidates
            if c.status in ("STRONG_DETECTION", "WEAK_DETECTION")
        ]

    if cnn_bundle is not None and not catalog.empty:
        cnn_rows = []
        for cand, fit, vet in zip(
            fitted_candidates,
            parts_1_to_5_result["fit_results"],
            parts_1_to_5_result["vetting_results"],
        ):
            matches = catalog[pd.to_numeric(catalog.get("candidate_id", -1), errors="coerce") == cand.candidate_id]
            catalog_row = matches.iloc[0] if not matches.empty else None
            views = build_cnn_candidate_views(parts_1_to_5_result["clean"], cand, fit, vet)
            cnn_rows.append({
                "candidate_id": cand.candidate_id,
                **predict_cnn_candidate_views(cnn_bundle, views, catalog_row=catalog_row),
            })
        if cnn_rows:
            cnn_df = pd.DataFrame(cnn_rows)
            catalog = catalog.merge(cnn_df, on="candidate_id", how="left", validate="one_to_one")
    elif model_bundle is not None and not catalog.empty:
        catalog = predict_ai_classifier(model_bundle, catalog)
    uncertainty_results = []
    uncertainty_rows = []
    clean = parts_1_to_5_result["clean"]

    for cand, fit, vet, cls in zip(
        fitted_candidates,
        parts_1_to_5_result["fit_results"],
        parts_1_to_5_result["vetting_results"],
        parts_1_to_5_result["classification_results"],
    ):
        ai_row = None
        if not catalog.empty:
            matches = catalog[pd.to_numeric(catalog.get("candidate_id", -1), errors="coerce") == cand.candidate_id]
            if not matches.empty:
                ai_row = matches.iloc[0]
        unc = estimate_candidate_uncertainty(clean, cand, fit, vet, cls, ai_row=ai_row)
        uncertainty_results.append(unc)
        uncertainty_rows.append({
            "candidate_id": cand.candidate_id,
            **{f"unc_{k}": v for k, v in unc.to_dict().items() if k not in ("tic_id", "sector", "candidate_id")},
        })
    if uncertainty_rows and not catalog.empty:
        unc_df = pd.DataFrame(uncertainty_rows)
        catalog = catalog.merge(unc_df, on="candidate_id", how="left", validate="one_to_one")
    parts_1_to_5_result = dict(parts_1_to_5_result)
    parts_1_to_5_result["catalog"] = catalog
    parts_1_to_5_result["uncertainty_results"] = uncertainty_results
    return parts_1_to_5_result


def run_parts_1_to_8_from_raw(
    raw: RawLightCurve,
    model_bundle: dict | None = None,
    cnn_bundle: dict | None = None,
    config: PipelineConfig | None = None,
) -> dict:
    result = run_parts_1_to_5_from_raw(raw, config=config)
    return attach_ai_and_uncertainty(result, model_bundle=model_bundle, cnn_bundle=cnn_bundle, config=config)


def run_parts_1_to_8_from_clean(
    clean: CleanLightCurve,
    model_bundle: dict | None = None,
    cnn_bundle: dict | None = None,
    config: PipelineConfig | None = None,
) -> dict:
    result = run_parts_1_to_5_from_clean(clean, config=config)
    return attach_ai_and_uncertainty(result, model_bundle=model_bundle, cnn_bundle=cnn_bundle, config=config)


def run_parts_1_to_8_from_fits(
    file_path: str,
    model_bundle: dict | None = None,
    cnn_bundle: dict | None = None,
    config: PipelineConfig | None = None,
) -> dict:
    result = run_parts_1_to_5_from_fits(file_path, config=config)
    return attach_ai_and_uncertainty(result, model_bundle=model_bundle, cnn_bundle=cnn_bundle, config=config)
