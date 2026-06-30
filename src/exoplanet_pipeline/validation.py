from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any
import json
from pathlib import Path
import numpy as np
import pandas as pd

from .ml import CANONICAL_CLASSES, normalize_label

try:
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        confusion_matrix,
        classification_report,
    )
except Exception:  # pragma: no cover
    accuracy_score = balanced_accuracy_score = f1_score = precision_score = recall_score = None
    confusion_matrix = classification_report = None


@dataclass
class ValidationReport:
    """Part 8 validation summary for detection, classification, and parameters."""

    n_rows: int
    n_labeled_rows: int
    detection_metrics: dict[str, Any]
    classification_metrics: dict[str, Any]
    parameter_metrics: dict[str, Any]
    reliability_metrics: dict[str, Any]
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save_json(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)


def _safe_float_series(s: Any) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _canonicalize_series(s: pd.Series) -> pd.Series:
    return s.map(normalize_label)


def _pick_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def has_detected_signal(df: pd.DataFrame) -> pd.Series:
    """Infer detection flag from candidate catalog columns."""
    if "status" in df.columns:
        st = df["status"].astype(str).str.upper()
        return st.str.contains("STRONG|WEAK|DETECTED") & ~st.str.contains("NO_DETECTION")
    if "snr" in df.columns or "local_snr" in df.columns or "fit_snr" in df.columns:
        snr = _safe_float_series(df.get("fit_snr", df.get("local_snr", df.get("snr"))))
        return snr >= 7.0
    if "final_predicted_class" in df.columns:
        return df["final_predicted_class"].astype(str).str.upper() != "NO_SIGNIFICANT_SIGNAL"
    return pd.Series(False, index=df.index)


def ground_truth_has_signal(labels: pd.Series) -> pd.Series:
    y = _canonicalize_series(labels)
    return y.notna() & (y != "NO_SIGNIFICANT_SIGNAL")


def compute_detection_metrics(df: pd.DataFrame, label_col: str = "label") -> dict[str, Any]:
    if label_col not in df.columns:
        return {"available": False, "reason": f"label_col {label_col!r} not found"}
    truth = ground_truth_has_signal(df[label_col])
    pred = has_detected_signal(df)
    tp = int((truth & pred).sum())
    fp = int((~truth & pred).sum())
    tn = int((~truth & ~pred).sum())
    fn = int((truth & ~pred).sum())
    precision = tp / (tp + fp) if (tp + fp) else np.nan
    recall = tp / (tp + fn) if (tp + fn) else np.nan
    specificity = tn / (tn + fp) if (tn + fp) else np.nan
    f1 = 2 * precision * recall / (precision + recall) if np.isfinite(precision) and np.isfinite(recall) and (precision + recall) else np.nan
    return {
        "available": True,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": float(precision) if np.isfinite(precision) else None,
        "recall": float(recall) if np.isfinite(recall) else None,
        "specificity": float(specificity) if np.isfinite(specificity) else None,
        "f1": float(f1) if np.isfinite(f1) else None,
    }


def compute_classification_metrics(
    df: pd.DataFrame,
    label_col: str = "label",
    pred_col: str | None = None,
) -> dict[str, Any]:
    if label_col not in df.columns:
        return {"available": False, "reason": f"label_col {label_col!r} not found"}
    pred_col = pred_col or _pick_column(df, ["final_predicted_class", "ai_predicted_class", "class_predicted_class", "predicted_class"])
    if pred_col is None:
        return {"available": False, "reason": "No prediction column found"}
    y_true = _canonicalize_series(df[label_col])
    y_pred = _canonicalize_series(df[pred_col])
    keep = y_true.notna() & y_pred.notna()
    if keep.sum() == 0:
        return {"available": False, "reason": "No rows with both canonical labels and predictions"}
    y_true = y_true[keep].astype(str)
    y_pred = y_pred[keep].astype(str)
    labels = [c for c in CANONICAL_CLASSES if c in set(y_true) or c in set(y_pred)]
    if accuracy_score is None:
        acc = float((y_true == y_pred).mean())
        return {"available": True, "n": int(len(y_true)), "accuracy": acc, "labels": labels}
    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    return {
        "available": True,
        "n": int(len(y_true)),
        "prediction_column": pred_col,
        "labels": labels,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "confusion_matrix": cm,
        "classification_report": classification_report(y_true, y_pred, labels=labels, zero_division=0, output_dict=True),
    }


def _metric_errors(pred: pd.Series, true: pd.Series) -> dict[str, Any]:
    pred = _safe_float_series(pred)
    true = _safe_float_series(true)
    keep = pred.notna() & true.notna() & np.isfinite(pred) & np.isfinite(true)
    if keep.sum() == 0:
        return {"available": False, "n": 0}
    diff = pred[keep] - true[keep]
    rel = diff.abs() / true[keep].abs().replace(0, np.nan)
    return {
        "available": True,
        "n": int(keep.sum()),
        "mae": float(diff.abs().mean()),
        "median_abs_error": float(diff.abs().median()),
        "rmse": float(np.sqrt(np.mean(diff**2))),
        "median_relative_abs_error": float(rel.median()) if rel.notna().any() else None,
        "within_1pct": float((rel <= 0.01).mean()) if rel.notna().any() else None,
        "within_5pct": float((rel <= 0.05).mean()) if rel.notna().any() else None,
        "within_10pct": float((rel <= 0.10).mean()) if rel.notna().any() else None,
    }


def compute_parameter_metrics(df: pd.DataFrame) -> dict[str, Any]:
    """Compare recovered period/depth/duration against true columns if present."""
    mapping = {
        "period_days": (["fit_period_days", "period_days"], ["true_period_days", "period_true_days", "injected_period_days"]),
        "depth_ppm": (["fit_depth_ppm", "depth_ppm"], ["true_depth_ppm", "depth_true_ppm", "injected_depth_ppm", "true_primary_depth_ppm"]),
        "duration_hours": (["fit_duration_days", "duration_days"], ["true_duration_hours", "duration_true_hours", "injected_duration_hours"]),
    }
    out: dict[str, Any] = {}
    for name, (pred_candidates, true_candidates) in mapping.items():
        pc = _pick_column(df, pred_candidates)
        tc = _pick_column(df, true_candidates)
        if pc is None or tc is None:
            out[name] = {"available": False, "reason": "missing prediction or truth column"}
            continue
        pred = df[pc]
        true = df[tc]
        if name == "duration_hours":
            # predictions are in days in the existing pipeline
            pred = _safe_float_series(pred) * 24.0
        out[name] = _metric_errors(pred, true)
        out[name]["prediction_column"] = pc
        out[name]["truth_column"] = tc
    return out


def compute_reliability_metrics(df: pd.DataFrame, label_col: str = "label") -> dict[str, Any]:
    """Evaluate whether reported confidence is meaningful.

    This computes simple calibration by binning confidence values and comparing
    mean confidence to empirical correctness.
    """
    pred_col = _pick_column(df, ["final_predicted_class", "ai_predicted_class", "class_predicted_class", "predicted_class"])
    conf_col = _pick_column(df, ["unc_final_confidence", "final_confidence", "ai_confidence", "class_confidence"])
    if label_col not in df.columns or pred_col is None or conf_col is None:
        return {"available": False, "reason": "Need label, prediction, and confidence columns"}
    y_true = _canonicalize_series(df[label_col])
    y_pred = _canonicalize_series(df[pred_col])
    conf = _safe_float_series(df[conf_col])
    keep = y_true.notna() & y_pred.notna() & conf.notna()
    if keep.sum() == 0:
        return {"available": False, "reason": "No valid reliability rows"}
    correct = (y_true[keep].astype(str) == y_pred[keep].astype(str)).astype(float)
    conf = conf[keep].clip(0, 1)
    bins = np.linspace(0, 1, 6)
    rows = []
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf >= lo) & (conf < hi if hi < 1 else conf <= hi)
        if m.sum() == 0:
            continue
        acc = float(correct[m].mean())
        cbar = float(conf[m].mean())
        frac = float(m.mean())
        ece += frac * abs(acc - cbar)
        rows.append({"bin_low": float(lo), "bin_high": float(hi), "n": int(m.sum()), "mean_confidence": cbar, "accuracy": acc})
    return {
        "available": True,
        "confidence_column": conf_col,
        "prediction_column": pred_col,
        "n": int(keep.sum()),
        "mean_confidence": float(conf.mean()),
        "empirical_accuracy": float(correct.mean()),
        "expected_calibration_error": float(ece),
        "bins": rows,
    }


def validate_candidate_catalog(
    catalog: pd.DataFrame,
    label_col: str = "label",
    pred_col: str | None = None,
) -> ValidationReport:
    warnings: list[str] = []
    if catalog.empty:
        return ValidationReport(0, 0, {}, {}, {}, {}, warnings=["EMPTY_CATALOG"])
    n_labeled = int(catalog[label_col].notna().sum()) if label_col in catalog.columns else 0
    if n_labeled == 0:
        warnings.append("NO_LABELS_AVAILABLE_FOR_FULL_VALIDATION")
    detection = compute_detection_metrics(catalog, label_col=label_col)
    classification = compute_classification_metrics(catalog, label_col=label_col, pred_col=pred_col)
    params = compute_parameter_metrics(catalog)
    reliability = compute_reliability_metrics(catalog, label_col=label_col)
    return ValidationReport(
        n_rows=int(len(catalog)),
        n_labeled_rows=n_labeled,
        detection_metrics=detection,
        classification_metrics=classification,
        parameter_metrics=params,
        reliability_metrics=reliability,
        warnings=warnings,
    )


def validation_report_to_markdown(report: ValidationReport) -> str:
    d = report.to_dict()
    lines = [
        "# Parts 7–8 Validation Report",
        "",
        f"Rows evaluated: **{report.n_rows}**",
        f"Labeled rows: **{report.n_labeled_rows}**",
        "",
        "## Detection",
    ]
    det = report.detection_metrics
    if det.get("available"):
        lines += [
            f"- Precision: {det.get('precision')}",
            f"- Recall: {det.get('recall')}",
            f"- F1: {det.get('f1')}",
            f"- TP/FP/TN/FN: {det.get('tp')}/{det.get('fp')}/{det.get('tn')}/{det.get('fn')}",
        ]
    else:
        lines.append(f"- Not available: {det.get('reason')}")
    cls = report.classification_metrics
    lines += ["", "## Classification"]
    if cls.get("available"):
        lines += [
            f"- Accuracy: {cls.get('accuracy')}",
            f"- Balanced accuracy: {cls.get('balanced_accuracy')}",
            f"- Macro F1: {cls.get('macro_f1')}",
            f"- Prediction column: `{cls.get('prediction_column')}`",
        ]
    else:
        lines.append(f"- Not available: {cls.get('reason')}")
    lines += ["", "## Parameter recovery"]
    for k, v in report.parameter_metrics.items():
        if v.get("available"):
            lines.append(f"- {k}: median relative abs error={v.get('median_relative_abs_error')}, within 10%={v.get('within_10pct')}")
        else:
            lines.append(f"- {k}: not available")
    rel = report.reliability_metrics
    lines += ["", "## Confidence reliability"]
    if rel.get("available"):
        lines += [
            f"- Mean confidence: {rel.get('mean_confidence')}",
            f"- Empirical accuracy: {rel.get('empirical_accuracy')}",
            f"- Expected calibration error: {rel.get('expected_calibration_error')}",
        ]
    else:
        lines.append(f"- Not available: {rel.get('reason')}")
    if report.warnings:
        lines += ["", "## Warnings"] + [f"- {w}" for w in report.warnings]
    return "\n".join(lines) + "\n"
