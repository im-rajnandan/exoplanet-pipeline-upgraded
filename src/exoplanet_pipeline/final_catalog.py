from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import numpy as np
import pandas as pd


PLANET_CLASS = "PLANETARY_TRANSIT_CANDIDATE"
EB_CLASS = "ECLIPSING_BINARY"
BLEND_CLASS = "BLEND_OR_CONTAMINATED_SIGNAL"
UNCERTAIN_CLASS = "UNCERTAIN_TRANSIT_LIKE_SIGNAL"
NO_SIGNAL_CLASS = "NO_SIGNIFICANT_SIGNAL"
SYSTEMATIC_CLASS = "INSTRUMENTAL_OR_LOW_QUALITY_SYSTEMATIC"
KNOWN_FINAL_CLASSES = {
    PLANET_CLASS,
    EB_CLASS,
    BLEND_CLASS,
    UNCERTAIN_CLASS,
    NO_SIGNAL_CLASS,
    SYSTEMATIC_CLASS,
    "STELLAR_VARIABILITY",
}


SCIENCE_COLUMN_ORDER = [
    "tic_id",
    "sector",
    "candidate_id",
    "science_priority_rank",
    "science_priority_score",
    "final_science_class",
    "final_science_confidence",
    "confidence_level",
    "recommended_action",
    "needs_manual_review",
    "period_days",
    "period_err_days",
    "epoch_time",
    "epoch_err_days",
    "duration_hours",
    "duration_err_hours",
    "depth_ppm",
    "depth_err_ppm",
    "snr",
    "effective_snr",
    "n_transits",
    "n_full_transits",
    "detection_method",
    "selected_flux_source",
    "crowdsap",
    "crowding_risk",
    "secondary_sigma",
    "odd_even_sigma",
    "centroid_shift_sigma",
    "data_quality_score",
    "evidence_summary",
    "risk_summary",
]


def _pick_first_existing(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    for c in columns:
        if c in df.columns:
            return df[c]
    return pd.Series(np.nan, index=df.index)


def _as_float(s: Any, default: float = np.nan) -> pd.Series:
    if isinstance(s, pd.Series):
        return pd.to_numeric(s, errors="coerce")
    return pd.Series(default)


def _safe_str_series(s: Any, index: pd.Index) -> pd.Series:
    if isinstance(s, pd.Series):
        return s.fillna("").astype(str)
    return pd.Series("", index=index, dtype=str)


def harmonize_candidate_catalog(catalog: pd.DataFrame) -> pd.DataFrame:
    """Create evaluator-friendly final columns from the internal Parts 1–8 catalog.

    The earlier modules deliberately preserve rich internal prefixes: candidate
    fields, fit_* fields, vet_* fields, class_* fields, ai_* fields, and unc_* fields.
    This function produces a compact, stable science catalog while keeping all
    original columns for debugging.
    """
    if catalog is None or catalog.empty:
        return pd.DataFrame()
    out = catalog.copy().reset_index(drop=True)
    idx = out.index

    out["final_science_class"] = _pick_first_existing(out, [
        "final_predicted_class",
        "class_predicted_class",
        "predicted_class",
    ]).fillna(UNCERTAIN_CLASS).astype(str)
    out["final_science_confidence"] = _as_float(_pick_first_existing(out, [
        "unc_final_confidence",
        "final_confidence",
        "ai_confidence",
        "class_confidence",
        "confidence",
    ])).clip(0, 1)
    out["confidence_level"] = _pick_first_existing(out, [
        "unc_confidence_level",
        "confidence_level",
    ]).fillna("").astype(str)
    # Numeric science fields with fallbacks.
    out["period_days"] = _as_float(_pick_first_existing(out, ["unc_period_days", "fit_period_days", "period_days"]))
    out["period_err_days"] = _as_float(_pick_first_existing(out, ["unc_period_err_days", "fit_period_err_days", "period_uncertainty_rough"]))
    out["epoch_time"] = _as_float(_pick_first_existing(out, ["unc_epoch_time", "fit_epoch_time", "epoch_time"]))
    out["epoch_err_days"] = _as_float(_pick_first_existing(out, ["unc_epoch_err_days", "fit_epoch_err_days"]))
    out["duration_days"] = _as_float(_pick_first_existing(out, ["unc_duration_days", "fit_duration_days", "duration_days"]))
    out["duration_hours"] = out["duration_days"] * 24.0
    out["duration_err_days"] = _as_float(_pick_first_existing(out, ["unc_duration_err_days", "fit_duration_err_days"]))
    out["duration_err_hours"] = out["duration_err_days"] * 24.0
    out["depth_ppm"] = _as_float(_pick_first_existing(out, ["unc_depth_ppm", "fit_depth_ppm", "depth_ppm"]))
    out["depth_err_ppm"] = _as_float(_pick_first_existing(out, ["unc_depth_err_ppm", "fit_depth_err_ppm"]))
    out["snr"] = _as_float(_pick_first_existing(out, ["unc_snr", "fit_snr", "local_snr", "snr"]))
    out["effective_snr"] = _as_float(_pick_first_existing(out, ["unc_effective_snr", "fit_snr", "local_snr", "snr"]))
    out["selected_flux_source"] = _pick_first_existing(out, ["flux_source", "selected_flux_source"]).fillna("").astype(str)

    out["crowdsap"] = _as_float(_pick_first_existing(out, ["vet_crowdsap", "crowdsap"]))
    out["crowding_risk"] = _as_float(_pick_first_existing(out, ["vet_crowding_risk", "crowding_risk"]))
    out["secondary_sigma"] = _as_float(_pick_first_existing(out, ["vet_secondary_sigma", "secondary_sigma"]))
    out["odd_even_sigma"] = _as_float(_pick_first_existing(out, ["vet_odd_even_sigma", "odd_even_sigma"]))
    out["centroid_shift_sigma"] = _as_float(_pick_first_existing(out, ["vet_centroid_shift_sigma", "centroid_shift_sigma"]))
    out["data_quality_score"] = _as_float(_pick_first_existing(out, ["vet_data_quality_score", "data_quality_score"]))

    # Evidence/risk summaries are made readable for reports and quick review.
    ev = _safe_str_series(_pick_first_existing(out, ["class_evidence", "evidence"]), idx)
    warn_cols = [c for c in ["warnings", "fit_warnings", "vet_warnings", "class_warnings", "unc_warnings", "final_classifier_warnings"] if c in out.columns]
    risks = []
    for i, row in out.iterrows():
        parts = []
        for c in warn_cols:
            val = str(row.get(c, ""))
            if val and val.lower() != "nan":
                parts.append(val)
        # Add major physical-risk facts even when warnings are absent.
        if pd.notna(out.loc[i, "secondary_sigma"]) and out.loc[i, "secondary_sigma"] >= 5:
            parts.append("strong_secondary_eclipse")
        if pd.notna(out.loc[i, "odd_even_sigma"]) and out.loc[i, "odd_even_sigma"] >= 3:
            parts.append("odd_even_depth_mismatch")
        if pd.notna(out.loc[i, "centroid_shift_sigma"]) and out.loc[i, "centroid_shift_sigma"] >= 3:
            parts.append("centroid_shift_risk")
        if pd.notna(out.loc[i, "crowding_risk"]) and out.loc[i, "crowding_risk"] >= 0.3:
            parts.append("crowded_aperture")
        risks.append(";".join(dict.fromkeys([p for p in parts if p])))
    out["evidence_summary"] = ev
    out["risk_summary"] = risks

    out["needs_manual_review"] = [needs_manual_review(row) for _, row in out.iterrows()]
    out["recommended_action"] = [recommend_action(row) for _, row in out.iterrows()]
    out["science_priority_score"] = [compute_science_priority(row) for _, row in out.iterrows()]
    out = out.sort_values("science_priority_score", ascending=False).reset_index(drop=True)
    out["science_priority_rank"] = np.arange(1, len(out) + 1)

    # Bring important final columns to the front, preserving all originals.
    front = [c for c in SCIENCE_COLUMN_ORDER if c in out.columns]
    rest = [c for c in out.columns if c not in front]
    return out[front + rest]


def validate_final_catalog_schema(catalog: pd.DataFrame, require_nonempty: bool = False) -> list[str]:
    """Return human-readable schema issues for a harmonized final catalog.

    Empty catalogs are allowed by default because no-detection batch runs should
    still write valid output files. Set require_nonempty=True for submissions
    that must contain at least one candidate.
    """
    issues: list[str] = []
    if catalog is None:
        return ["catalog is None"]
    if catalog.empty:
        return ["catalog is empty"] if require_nonempty else []

    df = harmonize_candidate_catalog(catalog) if "final_science_class" not in catalog.columns else catalog.copy()
    required = [
        "tic_id",
        "sector",
        "candidate_id",
        "science_priority_rank",
        "science_priority_score",
        "final_science_class",
        "final_science_confidence",
        "recommended_action",
        "period_days",
        "duration_hours",
        "depth_ppm",
        "effective_snr",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        issues.append(f"missing required columns: {', '.join(missing)}")
        return issues

    classes = df["final_science_class"].fillna("").astype(str)
    unknown_classes = sorted(set(classes) - KNOWN_FINAL_CLASSES)
    if unknown_classes:
        issues.append(f"unknown final_science_class values: {', '.join(unknown_classes)}")

    numeric_checks = {
        "final_science_confidence": (0.0, 1.0, True),
        "period_days": (0.0, np.inf, False),
        "duration_hours": (0.0, np.inf, False),
        "depth_ppm": (0.0, np.inf, False),
        "effective_snr": (0.0, np.inf, True),
        "science_priority_rank": (0.0, np.inf, False),
    }
    for col, (lo, hi, allow_zero) in numeric_checks.items():
        vals = pd.to_numeric(df[col], errors="coerce")
        bad = vals.notna() & ~np.isfinite(vals)
        if bad.any():
            issues.append(f"{col} contains non-finite values")
        finite = vals[np.isfinite(vals)]
        if finite.empty:
            issues.append(f"{col} has no finite values")
            continue
        lower_bad = finite < lo if allow_zero else finite <= lo
        upper_bad = finite > hi
        if lower_bad.any() or upper_bad.any():
            bound = f"{'[' if allow_zero else '('}{lo}, {hi}]"
            issues.append(f"{col} has values outside {bound}")

    ranks = pd.to_numeric(df["science_priority_rank"], errors="coerce")
    if ranks.notna().all() and sorted(ranks.astype(int).tolist()) != list(range(1, len(df) + 1)):
        issues.append("science_priority_rank is not a contiguous 1-based ranking")

    return issues


def compute_science_priority(row: pd.Series) -> float:
    """Rank candidates for human review/follow-up, not as a proof of planet status."""
    cls = str(row.get("final_science_class", ""))
    conf = _to_float(row.get("final_science_confidence"), 0.0)
    snr = _to_float(row.get("effective_snr", row.get("snr")), 0.0)
    ntr = _to_float(row.get("n_full_transits", row.get("n_transits")), 0.0)
    dq = _to_float(row.get("data_quality_score"), 0.6)
    sec = _to_float(row.get("secondary_sigma"), 0.0)
    oe = _to_float(row.get("odd_even_sigma"), 0.0)
    cen = _to_float(row.get("centroid_shift_sigma"), 0.0)
    crowd = _to_float(row.get("crowding_risk"), 0.0)

    base = 100.0 * conf + 4.0 * min(snr, 30.0) + 5.0 * min(ntr, 6.0) + 20.0 * dq
    if cls == PLANET_CLASS:
        base += 50.0
    elif cls == UNCERTAIN_CLASS:
        base += 10.0
    elif cls == EB_CLASS:
        base -= 20.0
    elif cls == BLEND_CLASS:
        base -= 25.0
    elif cls == NO_SIGNAL_CLASS:
        base -= 80.0
    elif cls == SYSTEMATIC_CLASS:
        base -= 60.0
    # Penalize false-positive evidence for follow-up priority.
    base -= 8.0 * max(sec - 3.0, 0.0)
    base -= 7.0 * max(oe - 2.0, 0.0)
    base -= 7.0 * max(cen - 2.0, 0.0)
    base -= 25.0 * max(crowd - 0.25, 0.0)
    return float(base)


def needs_manual_review(row: pd.Series) -> bool:
    """
    Determine if a candidate needs manual review based on probability margin, max probability, and epistemic uncertainty.
    """
    conf = _to_float(row.get("final_science_confidence"), 0.0)

    # 1. Max probability (confidence) < 0.5
    if conf < 0.5:
        return True

    # 2. Top-2 difference (margin) < 0.15
    prob_cols = [f"final_prob_{cls}" for cls in KNOWN_FINAL_CLASSES if f"final_prob_{cls}" in row.index]
    if len(prob_cols) >= 2:
        probs = sorted([_to_float(row.get(col), 0.0) for col in prob_cols], reverse=True)
        margin = probs[0] - probs[1]
        if margin < 0.15:
            return True

    # 3. Epistemic uncertainty > 0.1
    epistemic = _to_float(row.get("cnn_epistemic_uncertainty"), 0.0)
    if epistemic > 0.10:
        return True

    return False


def recommend_action(row: pd.Series) -> str:
    cls = str(row.get("final_science_class", ""))
    conf = _to_float(row.get("final_science_confidence"), 0.0)
    sec = _to_float(row.get("secondary_sigma"), 0.0)
    oe = _to_float(row.get("odd_even_sigma"), 0.0)
    cen = _to_float(row.get("centroid_shift_sigma"), 0.0)
    crowd = _to_float(row.get("crowding_risk"), 0.0)
    snr = _to_float(row.get("effective_snr", row.get("snr")), 0.0)
    
    # Check manual review flag
    review = bool(row.get("needs_manual_review", False))

    if cls == PLANET_CLASS and conf >= 0.75 and sec < 4 and oe < 3 and cen < 3:
        if crowd >= 0.25:
            action = "HIGH_PRIORITY_PLANET_CANDIDATE__CHECK_GAIA_TPF_CONTAMINATION"
        else:
            action = "HIGH_PRIORITY_PLANET_CANDIDATE__VISUAL_REVIEW_AND_FOLLOWUP"
    elif cls == PLANET_CLASS:
        action = "PLANET_CANDIDATE__NEEDS_ADDITIONAL_VETTING"
    elif cls == EB_CLASS:
        action = "LIKELY_ECLIPSING_BINARY__VERIFY_PERIOD_ALIAS_AND_SECONDARY"
    elif cls == BLEND_CLASS:
        action = "LIKELY_BLEND__INSPECT_CENTROIDS_TPF_AND_NEARBY_SOURCES"
    elif cls == UNCERTAIN_CLASS and snr >= 7:
        action = "UNCERTAIN_TRANSIT_LIKE_SIGNAL__KEEP_FOR_REVIEW"
    elif cls == SYSTEMATIC_CLASS:
        action = "LIKELY_SYSTEMATIC_OR_LOW_QUALITY__LOW_PRIORITY"
    else:
        action = "NO_ACTION_OR_LOW_PRIORITY"

    if review:
        return f"MANUAL_REVIEW_REQUIRED__{action}"
    return action


def _to_float(x: Any, default: float = np.nan) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) else default
    except Exception:
        return default


def summarize_final_catalog(catalog: pd.DataFrame) -> dict[str, Any]:
    if catalog is None or catalog.empty:
        return {"n_candidates": 0, "status": "EMPTY"}
    df = harmonize_candidate_catalog(catalog) if "final_science_class" not in catalog.columns else catalog.copy()
    summary: dict[str, Any] = {
        "n_candidates": int(len(df)),
        "n_unique_targets": int(df["tic_id"].nunique()) if "tic_id" in df else None,
        "class_counts": df["final_science_class"].value_counts(dropna=False).to_dict() if "final_science_class" in df else {},
        "confidence_mean": float(pd.to_numeric(df.get("final_science_confidence"), errors="coerce").mean()) if "final_science_confidence" in df else None,
        "top_candidates": [],
    }
    top_cols = [c for c in ["tic_id", "sector", "candidate_id", "final_science_class", "final_science_confidence", "period_days", "duration_hours", "depth_ppm", "effective_snr", "recommended_action"] if c in df]
    summary["top_candidates"] = df.head(10)[top_cols].to_dict(orient="records") if top_cols else []
    return summary


def save_final_catalog(catalog: pd.DataFrame, output_dir: str | Path, prefix: str = "final") -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    final = harmonize_candidate_catalog(catalog)
    issues = validate_final_catalog_schema(final)
    if issues:
        raise ValueError("Final catalog schema validation failed: " + "; ".join(issues))
    paths = {
        "catalog_csv": output_dir / f"{prefix}_candidate_catalog.csv",
        "summary_json": output_dir / f"{prefix}_catalog_summary.json",
    }
    final.to_csv(paths["catalog_csv"], index=False)
    with open(paths["summary_json"], "w", encoding="utf-8") as f:
        json.dump(summarize_final_catalog(final), f, indent=2)
    return paths
