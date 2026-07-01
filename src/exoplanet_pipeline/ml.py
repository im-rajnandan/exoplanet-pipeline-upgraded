from __future__ import annotations

from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Iterable
import json
import warnings

import numpy as np
import pandas as pd

from .classification_policy import (
    CANONICAL_CLASSES,
    finalize_probabilities,
    renormalize_probs as _renormalize_probs,
)

try:
    import joblib
    from sklearn.base import clone
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
    )
    from sklearn.model_selection import StratifiedKFold, train_test_split
    from sklearn.pipeline import Pipeline
except Exception as exc:  # pragma: no cover - handled at runtime for optional dependency clarity
    joblib = None
    _SKLEARN_IMPORT_ERROR = exc
else:
    _SKLEARN_IMPORT_ERROR = None


# Numeric feature columns that are expected from the Parts 1-5 catalog.
# The model can also ingest additional numeric features from a curated dataset.
PREFERRED_FEATURE_COLUMNS: tuple[str, ...] = (
    # Candidate/detection features
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
    "periodogram_peak_power",
    "period_uncertainty_rough",
    # Refined fit features
    "fit_period_days",
    "fit_period_err_days",
    "fit_epoch_err_days",
    "fit_duration_days",
    "fit_duration_err_days",
    "fit_depth_fraction",
    "fit_depth_err_fraction",
    "fit_depth_ppm",
    "fit_depth_err_ppm",
    "fit_rp_over_rstar",
    "fit_rp_earth",
    "fit_stellar_radius_rsun",
    "fit_snr",
    "fit_n_in_transit_points",
    "fit_n_events",
    "fit_n_good_events",
    "fit_event_depth_scatter_ppm",
    # Vetting features
    "vet_odd_depth_ppm",
    "vet_even_depth_ppm",
    "vet_odd_even_sigma",
    "vet_odd_even_depth_diff_ppm",
    "vet_secondary_depth_ppm",
    "vet_secondary_sigma",
    "vet_secondary_phase",
    "vet_secondary_to_primary_ratio",
    "vet_centroid_shift_pix",
    "vet_centroid_shift_sigma",
    "vet_crowdsap",
    "vet_flfrcsap",
    "vet_crowding_risk",
    "vet_corrected_depth_ppm",
    "vet_v_shape_score",
    "vet_transit_asymmetry",
    "vet_out_of_transit_rms_ppm",
    "vet_red_noise_proxy",
    "vet_harmonic_risk",
    "vet_data_quality_score",
    # Rule-based baseline scores are intentionally included as weak meta-features.
    # They are transparent scientific priors, not labels.
    "class_confidence",
    "class_planet_score",
    "class_eb_score",
    "class_blend_score",
    "class_stellar_variability_score",
    "class_systematic_score",
)

NON_FEATURE_SUBSTRINGS: tuple[str, ...] = (
    "label",
    "class_predicted",
    "predicted_class",
    "status",
    "warning",
    "evidence",
    "method",
    "source",
    "error",
    "sample",
    "name",
    "path",
    "file",
)

ID_COLUMNS: tuple[str, ...] = ("tic_id", "sector", "candidate_id")

LABEL_ALIASES: dict[str, str] = {
    # Planet-like labels
    "planet": "PLANETARY_TRANSIT_CANDIDATE",
    "planet_candidate": "PLANETARY_TRANSIT_CANDIDATE",
    "planetary_transit": "PLANETARY_TRANSIT_CANDIDATE",
    "planetary_transit_candidate": "PLANETARY_TRANSIT_CANDIDATE",
    "confirmed_planet": "PLANETARY_TRANSIT_CANDIDATE",
    "known_planet": "PLANETARY_TRANSIT_CANDIDATE",
    "candidate": "PLANETARY_TRANSIT_CANDIDATE",
    "planet_like": "PLANETARY_TRANSIT_CANDIDATE",
    "pc": "PLANETARY_TRANSIT_CANDIDATE",
    "apc": "PLANETARY_TRANSIT_CANDIDATE",
    "cp": "PLANETARY_TRANSIT_CANDIDATE",
    "kp": "PLANETARY_TRANSIT_CANDIDATE",
    "toi_pc": "PLANETARY_TRANSIT_CANDIDATE",
    "transit": "PLANETARY_TRANSIT_CANDIDATE",
    "exoplanet": "PLANETARY_TRANSIT_CANDIDATE",
    # Eclipsing binaries
    "eb": "ECLIPSING_BINARY",
    "eclipsing_binary": "ECLIPSING_BINARY",
    "binary": "ECLIPSING_BINARY",
    "detached_eb": "ECLIPSING_BINARY",
    "contact_binary": "ECLIPSING_BINARY",
    "eclipse": "ECLIPSING_BINARY",
    # Blends / contamination
    "blend": "BLEND_OR_CONTAMINATED_SIGNAL",
    "blended": "BLEND_OR_CONTAMINATED_SIGNAL",
    "background_eb": "BLEND_OR_CONTAMINATED_SIGNAL",
    "beb": "BLEND_OR_CONTAMINATED_SIGNAL",
    "contaminated": "BLEND_OR_CONTAMINATED_SIGNAL",
    "contamination": "BLEND_OR_CONTAMINATED_SIGNAL",
    "background_eclipsing_binary": "BLEND_OR_CONTAMINATED_SIGNAL",
    # Stellar variability
    "stellar_variability": "STELLAR_VARIABILITY",
    "variable": "STELLAR_VARIABILITY",
    "starspot": "STELLAR_VARIABILITY",
    "rotation": "STELLAR_VARIABILITY",
    "rotational_variability": "STELLAR_VARIABILITY",
    "pulsator": "STELLAR_VARIABILITY",
    "flare": "STELLAR_VARIABILITY",
    # Instrumental/systematics
    "systematic": "INSTRUMENTAL_OR_LOW_QUALITY_SYSTEMATIC",
    "instrumental": "INSTRUMENTAL_OR_LOW_QUALITY_SYSTEMATIC",
    "low_quality": "INSTRUMENTAL_OR_LOW_QUALITY_SYSTEMATIC",
    "artifact": "INSTRUMENTAL_OR_LOW_QUALITY_SYSTEMATIC",
    "false_alarm": "INSTRUMENTAL_OR_LOW_QUALITY_SYSTEMATIC",
    "fa": "INSTRUMENTAL_OR_LOW_QUALITY_SYSTEMATIC",
    "false_positive": "INSTRUMENTAL_OR_LOW_QUALITY_SYSTEMATIC",
    "false_positive_or_other": "INSTRUMENTAL_OR_LOW_QUALITY_SYSTEMATIC",
    "fp": "INSTRUMENTAL_OR_LOW_QUALITY_SYSTEMATIC",
    # No-signal / uncertain
    "no_signal": "NO_SIGNIFICANT_SIGNAL",
    "noise": "NO_SIGNIFICANT_SIGNAL",
    "quiet": "NO_SIGNIFICANT_SIGNAL",
    "none": "NO_SIGNIFICANT_SIGNAL",
    "negative": "NO_SIGNIFICANT_SIGNAL",
    "uncertain": "UNCERTAIN_TRANSIT_LIKE_SIGNAL",
    "unknown": "UNCERTAIN_TRANSIT_LIKE_SIGNAL",
    "ambiguous": "UNCERTAIN_TRANSIT_LIKE_SIGNAL",
    "marginal": "UNCERTAIN_TRANSIT_LIKE_SIGNAL",
}


@dataclass
class MLTrainingResult:
    """Container returned by train_ai_classifier."""

    model_bundle: dict[str, Any]
    feature_columns: list[str]
    class_names: list[str]
    train_metrics: dict[str, Any]
    test_metrics: dict[str, Any]
    confusion_matrix: list[list[int]]
    feature_importance: pd.DataFrame
    warnings: list[str] = field(default_factory=list)

    def to_json_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["feature_importance"] = self.feature_importance.to_dict(orient="records")
        return d


def _require_sklearn() -> None:
    if _SKLEARN_IMPORT_ERROR is not None:
        raise ImportError(
            "Part 6 requires scikit-learn and joblib. Install with: pip install scikit-learn joblib"
        ) from _SKLEARN_IMPORT_ERROR


def normalize_label(label: Any) -> str | None:
    """Map organizer/curated labels to canonical project classes.

    Unknown labels return None rather than guessing. This prevents accidental
    leakage or silent relabeling when the provided curated dataset uses a new
    taxonomy.
    """
    if label is None or (isinstance(label, float) and np.isnan(label)):
        return None
    raw = str(label).strip()
    if not raw:
        return None
    upper = raw.upper().replace(" ", "_").replace("-", "_").replace("/", "_")
    if upper in CANONICAL_CLASSES:
        return upper
    key = raw.lower().strip().replace(" ", "_").replace("-", "_").replace("/", "_")
    return LABEL_ALIASES.get(key)


def infer_feature_columns(
    df: pd.DataFrame,
    label_col: str | None = None,
    include_extra_numeric: bool = True,
    preferred_columns: Iterable[str] = PREFERRED_FEATURE_COLUMNS,
) -> list[str]:
    """Infer safe numeric feature columns from a candidate catalog.

    Preferred scientific columns are kept first. Extra numeric columns may be
    included, but obvious IDs/text/label/leakage columns are excluded.
    """
    cols: list[str] = []
    for c in preferred_columns:
        if c in df.columns:
            cols.append(c)

    if include_extra_numeric:
        for c in df.columns:
            if c in cols:
                continue
            c_lower = c.lower()
            if label_col is not None and c == label_col:
                continue
            if c in ID_COLUMNS:
                continue
            if any(s in c_lower for s in NON_FEATURE_SUBSTRINGS):
                continue
            # Test whether the column can behave numerically.
            numeric = pd.to_numeric(df[c], errors="coerce")
            if numeric.notna().sum() > 0:
                cols.append(c)
    return cols


def prepare_ml_frame(
    df: pd.DataFrame,
    label_col: str | None = None,
    feature_columns: list[str] | None = None,
    include_extra_numeric: bool = True,
    drop_unknown_labels: bool = True,
) -> tuple[pd.DataFrame, pd.Series | None, list[str], pd.DataFrame]:
    """Convert a Parts 1-5 catalog or curated table into model-ready X/y.

    Returns X, y, feature_columns, and the aligned metadata frame. X is numeric
    but still may contain NaNs; the sklearn pipeline imputes them.
    """
    if df.empty:
        raise ValueError("Input dataframe is empty; cannot prepare ML frame.")

    work = df.copy()
    y: pd.Series | None = None
    if label_col is not None:
        if label_col not in work.columns:
            raise ValueError(f"label_col={label_col!r} not found in dataframe columns.")
        y_norm = work[label_col].map(normalize_label)
        if drop_unknown_labels:
            keep = y_norm.notna()
            work = work.loc[keep].copy()
            y_norm = y_norm.loc[keep]
        if y_norm.isna().any():
            bad = sorted(set(work.loc[y_norm.isna(), label_col].astype(str)))
            raise ValueError(f"Unknown labels encountered: {bad}. Add aliases or clean labels first.")
        y = y_norm.astype(str)

    if feature_columns is None:
        feature_columns = infer_feature_columns(work, label_col=label_col, include_extra_numeric=include_extra_numeric)
    if not feature_columns:
        raise ValueError("No usable numeric feature columns found.")

    X = pd.DataFrame(index=work.index)
    for c in feature_columns:
        if c not in work.columns:
            X[c] = np.nan
        else:
            X[c] = pd.to_numeric(work[c], errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    meta = work[[c for c in ID_COLUMNS if c in work.columns]].copy()
    return X, y, feature_columns, meta


def _build_base_estimator(
    model_type: str = "random_forest",
    random_state: int = 42,
) -> Pipeline:
    _require_sklearn()
    if model_type == "random_forest":
        model = RandomForestClassifier(
            n_estimators=220,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=-1,
        )
    elif model_type == "extra_trees":
        model = ExtraTreesClassifier(
            n_estimators=220,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )
    elif model_type == "hist_gbdt":
        # HistGradientBoosting can handle non-linearities but has no class_weight
        # in older sklearn versions, so it is less robust for imbalanced labels.
        model = HistGradientBoostingClassifier(random_state=random_state, max_iter=200)
    else:
        raise ValueError(f"Unknown model_type={model_type!r}")

    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("model", model),
        ]
    )


def _metrics(y_true: Iterable[str], y_pred: Iterable[str], labels: list[str]) -> dict[str, Any]:
    y_true = list(y_true)
    y_pred = list(y_pred)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "classification_report": classification_report(y_true, y_pred, labels=labels, zero_division=0, output_dict=True),
    }


def _class_counts(y: pd.Series) -> dict[str, int]:
    return {str(k): int(v) for k, v in y.value_counts().sort_index().items()}


def _can_stratify(y: pd.Series, test_size: float) -> bool:
    counts = y.value_counts()
    if len(counts) < 2:
        return False
    if counts.min() < 2:
        return False
    n_test = int(np.ceil(len(y) * test_size))
    return n_test >= len(counts)


def train_ai_classifier(
    catalog: pd.DataFrame,
    label_col: str = "label",
    feature_columns: list[str] | None = None,
    include_extra_numeric: bool = True,
    model_type: str = "random_forest",
    calibrate: bool = True,
    test_size: float = 0.25,
    random_state: int = 42,
) -> MLTrainingResult:
    """Train the Part 6 supervised classifier.

    This is designed for the curated labeled dataset. It can also train on the
    synthetic ML feature catalog for development/demo purposes.
    """
    _require_sklearn()
    warnings_list: list[str] = []
    X, y, feature_columns, _ = prepare_ml_frame(
        catalog,
        label_col=label_col,
        feature_columns=feature_columns,
        include_extra_numeric=include_extra_numeric,
        drop_unknown_labels=True,
    )
    assert y is not None
    if len(y.unique()) < 2:
        raise ValueError("Need at least two classes to train a supervised classifier.")

    labels = [c for c in CANONICAL_CLASSES if c in set(y)]
    if not _can_stratify(y, test_size):
        warnings_list.append("Dataset too small or imbalanced for a stratified holdout; evaluation is in-sample only.")
        X_train, X_test, y_train, y_test = X, X, y, y
        holdout_used = False
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            stratify=y,
            random_state=random_state,
        )
        holdout_used = True

    base = _build_base_estimator(model_type=model_type, random_state=random_state)

    # Probability calibration improves confidence estimates when enough labeled
    # examples exist per class. If the curated dataset is small, we avoid fragile
    # calibration and keep the random-forest probabilities.
    train_counts = y_train.value_counts()
    min_train_class = int(train_counts.min())
    if calibrate and min_train_class >= 3 and len(train_counts) >= 2:
        cv_folds = min(5, min_train_class)
        estimator = CalibratedClassifierCV(
            estimator=base,
            cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state),
            method="sigmoid",
        )
        calibrated = True
    else:
        estimator = base
        calibrated = False
        if calibrate:
            warnings_list.append("Skipped probability calibration because at least one class has <3 training examples.")

    estimator.fit(X_train, y_train)
    train_pred = estimator.predict(X_train)
    test_pred = estimator.predict(X_test)

    train_metrics = _metrics(y_train, train_pred, labels=labels)
    test_metrics = _metrics(y_test, test_pred, labels=labels)
    cm = confusion_matrix(y_test, test_pred, labels=labels).tolist()

    # Train a non-calibrated importance pipeline on all data for feature rankings.
    importance_estimator = _build_base_estimator(model_type="random_forest", random_state=random_state)
    importance_estimator.fit(X, y)
    model = importance_estimator.named_steps["model"]
    if hasattr(model, "feature_importances_"):
        # SimpleImputer(add_indicator=True) adds missingness indicators after the
        # original columns. We report importances only for original features by
        # truncating to len(feature_columns); this is conservative and readable.
        imps = np.asarray(model.feature_importances_)[: len(feature_columns)]
    else:
        imps = np.zeros(len(feature_columns), dtype=float)
    feature_importance = pd.DataFrame(
        {"feature": feature_columns, "importance": imps}
    ).sort_values("importance", ascending=False, ignore_index=True)

    # Fit final estimator on all labeled data for deployment.
    final_base = _build_base_estimator(model_type=model_type, random_state=random_state)
    if calibrated:
        final_counts = y.value_counts()
        cv_folds = min(5, int(final_counts.min()))
        final_estimator = CalibratedClassifierCV(
            estimator=final_base,
            cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state),
            method="sigmoid",
        )
    else:
        final_estimator = final_base
    final_estimator.fit(X, y)

    model_bundle = {
        "estimator": final_estimator,
        "feature_columns": feature_columns,
        "class_names": labels,
        "canonical_classes": list(CANONICAL_CLASSES),
        "label_aliases": LABEL_ALIASES,
        "model_type": model_type,
        "calibrated": calibrated,
        "holdout_used": holdout_used,
        "training_metadata": {
            "n_rows": int(len(X)),
            "n_features": int(len(feature_columns)),
            "class_counts": _class_counts(y),
            "test_size": float(test_size),
            "random_state": int(random_state),
        },
        "feature_importance": feature_importance,
    }

    return MLTrainingResult(
        model_bundle=model_bundle,
        feature_columns=feature_columns,
        class_names=labels,
        train_metrics=train_metrics,
        test_metrics=test_metrics,
        confusion_matrix=cm,
        feature_importance=feature_importance,
        warnings=warnings_list,
    )


def compute_counterfactual_importance(
    row: pd.Series | dict[str, Any],
    estimator: Any,
    feature_columns: list[str],
    original_pred: str,
) -> dict[str, Any]:
    """
    Evaluate load-bearing features by ablating feature groups for a single candidate row.
    """
    feature_groups = {
        'centroid': [c for c in feature_columns if 'cent' in c or 'pos' in c or 'shift' in c],
        'vetting': [c for c in feature_columns if 'odd' in c or 'even' in c or 'sec' in c or 'eclipse' in c],
        'stellar': [c for c in feature_columns if 'rad' in c or 'mass' in c or 'teff' in c or 'logg' in c],
        'crowding': [c for c in feature_columns if 'crowd' in c or 'dilution' in c]
    }

    ablated_predictions = {}
    row_dict = dict(row)
    classes = list(estimator.classes_)

    for group_name, cols in feature_groups.items():
        if not cols:
            continue

        counter_row = row_dict.copy()
        for col in cols:
            if col in counter_row:
                if 'cent' in col or 'pos' in col or 'shift' in col:
                    counter_row[col] = 0.0
                elif 'odd' in col or 'even' in col or 'sec' in col:
                    counter_row[col] = 0.0
                elif 'crowd' in col:
                    counter_row[col] = 1.0

        single_df = pd.DataFrame([counter_row])
        single_df = single_df.reindex(columns=feature_columns).fillna(0.0)

        try:
            prob = estimator.predict_proba(single_df)[0]
            new_pred = classes[prob.argmax()]
        except Exception:
            new_pred = original_pred

        is_load_bearing = (new_pred != original_pred)
        ablated_predictions[group_name] = {
            'new_prediction': new_pred,
            'is_load_bearing': int(is_load_bearing)
        }
    return ablated_predictions


def predict_ai_classifier(
    model_bundle: dict[str, Any],
    catalog: pd.DataFrame,
    apply_physical_guardrails: bool = True,
    rule_weight: float = 0.25,
) -> pd.DataFrame:
    """Predict class probabilities for candidate rows.

    The final probabilities blend supervised probabilities with the transparent
    rule-based scores from Part 5 when those columns are available. Physical
    guardrails prevent obviously EB/blend/systematic cases from being called
    high-confidence planets solely because of a learned decision boundary.
    """
    _require_sklearn()
    feature_columns = list(model_bundle["feature_columns"])
    estimator = model_bundle["estimator"]
    X, _, _, meta = prepare_ml_frame(
        catalog,
        label_col=None,
        feature_columns=feature_columns,
        include_extra_numeric=False,
    )
    raw_probs = estimator.predict_proba(X)
    classes = list(estimator.classes_)
    out = catalog.reset_index(drop=True).copy()

    # Raw AI probabilities.
    for cls in CANONICAL_CLASSES:
        if cls in classes:
            out[f"ai_prob_{cls}"] = raw_probs[:, classes.index(cls)]
        else:
            out[f"ai_prob_{cls}"] = 0.0
    ai_idx = raw_probs.argmax(axis=1)
    out["ai_predicted_class"] = [classes[i] for i in ai_idx]
    out["ai_confidence"] = raw_probs.max(axis=1)

    final_probs_rows: list[dict[str, float]] = []
    final_pred: list[str] = []
    final_conf: list[float] = []
    final_warnings: list[str] = []

    for _, row in out.iterrows():
        probs = {cls: float(row.get(f"ai_prob_{cls}", 0.0)) for cls in CANONICAL_CLASSES}
        warnings_here: list[str] = []

        rule_probs = _rule_scores_from_row(row)
        if rule_probs is not None and 0.0 < rule_weight <= 1.0:
            probs = {
                cls: (1.0 - rule_weight) * probs.get(cls, 0.0) + rule_weight * rule_probs.get(cls, 0.0)
                for cls in CANONICAL_CLASSES
            }
            warnings_here.append("blended_ai_with_rule_based_scores")

        probs, pred, conf, policy_warnings = finalize_probabilities(
            probs,
            row=row,
            apply_guardrails=apply_physical_guardrails,
            low_margin_warning="low_ai_margin_downgraded_to_uncertain",
        )
        warnings_here.extend(policy_warnings)

        final_probs_rows.append(probs)
        final_pred.append(pred)
        final_conf.append(float(conf))
        final_warnings.append(";".join(warnings_here))

    centroid_load = []
    vetting_load = []
    stellar_load = []
    crowding_load = []

    for _, row in out.iterrows():
        original_pred = row["ai_predicted_class"]
        cf_imp = compute_counterfactual_importance(row, estimator, feature_columns, original_pred)
        centroid_load.append(cf_imp.get("centroid", {}).get("is_load_bearing", 0))
        vetting_load.append(cf_imp.get("vetting", {}).get("is_load_bearing", 0))
        stellar_load.append(cf_imp.get("stellar", {}).get("is_load_bearing", 0))
        crowding_load.append(cf_imp.get("crowding", {}).get("is_load_bearing", 0))

    out["counterfactual_centroid_load_bearing"] = centroid_load
    out["counterfactual_vetting_load_bearing"] = vetting_load
    out["counterfactual_stellar_load_bearing"] = stellar_load
    out["counterfactual_crowding_load_bearing"] = crowding_load

    for cls in CANONICAL_CLASSES:
        out[f"final_prob_{cls}"] = [p.get(cls, 0.0) for p in final_probs_rows]
    out["final_predicted_class"] = final_pred
    out["final_confidence"] = final_conf
    out["final_classifier_warnings"] = final_warnings
    out["final_classifier_method"] = "supervised_ai_plus_physical_guardrails"
    return out


def _rule_scores_from_row(row: pd.Series) -> dict[str, float] | None:
    mapping = {
        "PLANETARY_TRANSIT_CANDIDATE": "class_planet_score",
        "ECLIPSING_BINARY": "class_eb_score",
        "BLEND_OR_CONTAMINATED_SIGNAL": "class_blend_score",
        "STELLAR_VARIABILITY": "class_stellar_variability_score",
        "INSTRUMENTAL_OR_LOW_QUALITY_SYSTEMATIC": "class_systematic_score",
    }
    scores = {cls: float(pd.to_numeric(row.get(col, np.nan), errors="coerce")) for cls, col in mapping.items()}
    if not any(np.isfinite(v) and v > 0 for v in scores.values()):
        return None
    scores = {k: (v if np.isfinite(v) and v > 0 else 0.0) for k, v in scores.items()}
    scores["NO_SIGNIFICANT_SIGNAL"] = 0.0
    scores["UNCERTAIN_TRANSIT_LIKE_SIGNAL"] = max(0.0, 1.0 - max(scores.values())) * 0.25
    return _renormalize_probs(scores)


def save_model_bundle(model_bundle: dict[str, Any], path: str | Path) -> None:
    _require_sklearn()
    if joblib is None:
        raise ImportError("joblib is required to save the model bundle")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_bundle, path)


def load_model_bundle(path: str | Path) -> dict[str, Any]:
    _require_sklearn()
    if joblib is None:
        raise ImportError("joblib is required to load the model bundle")
    return joblib.load(path)


def write_training_artifacts(
    result: MLTrainingResult,
    output_dir: str | Path,
    prefix: str = "part6_ai_classifier",
) -> dict[str, Path]:
    """Write model, metrics, feature importances, and a short model card."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "model": output_dir / f"{prefix}.joblib",
        "metrics": output_dir / f"{prefix}_metrics.json",
        "feature_importance": output_dir / f"{prefix}_feature_importance.csv",
        "model_card": output_dir / f"{prefix}_MODEL_CARD.md",
    }
    save_model_bundle(result.model_bundle, paths["model"])
    metrics_payload = {
        "class_names": result.class_names,
        "feature_columns": result.feature_columns,
        "train_metrics": result.train_metrics,
        "test_metrics": result.test_metrics,
        "confusion_matrix": result.confusion_matrix,
        "warnings": result.warnings,
        "training_metadata": result.model_bundle.get("training_metadata", {}),
        "calibrated": result.model_bundle.get("calibrated"),
        "holdout_used": result.model_bundle.get("holdout_used"),
    }
    with open(paths["metrics"], "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)
    result.feature_importance.to_csv(paths["feature_importance"], index=False)
    paths["model_card"].write_text(make_model_card(result), encoding="utf-8")
    return paths


def make_model_card(result: MLTrainingResult) -> str:
    meta = result.model_bundle.get("training_metadata", {})
    lines = [
        "# Part 6 AI Classifier Model Card",
        "",
        "## Intended use",
        "This supervised model classifies candidate light-curve dips after Parts 1-5 have already extracted physical and statistical features. It should not be used directly on raw light curves, and it should not be treated as final astronomical validation.",
        "",
        "## Classes",
        *[f"- `{c}`" for c in result.class_names],
        "",
        "## Training data summary",
        f"- Rows: {meta.get('n_rows')}",
        f"- Features: {meta.get('n_features')}",
        f"- Class counts: {meta.get('class_counts')}",
        f"- Calibrated probabilities: {result.model_bundle.get('calibrated')}",
        f"- Holdout evaluation used: {result.model_bundle.get('holdout_used')}",
        "",
        "## Evaluation",
        f"- Holdout / evaluation accuracy: {result.test_metrics.get('accuracy'):.4f}",
        f"- Balanced accuracy: {result.test_metrics.get('balanced_accuracy'):.4f}",
        f"- Macro F1: {result.test_metrics.get('macro_f1'):.4f}",
        "",
        "## Top features",
    ]
    for _, row in result.feature_importance.head(15).iterrows():
        lines.append(f"- `{row['feature']}`: {row['importance']:.5f}")
    lines += [
        "",
        "## Guardrails",
        "At prediction time, the package can blend AI probabilities with the rule-based scientific vetter and apply hard physical guardrails for strong secondary eclipses, odd/even mismatches, centroid shifts, low data quality, and low SNR.",
        "",
        "## Limitations",
        "This model learns the label quality and class definitions of the provided curated dataset. It should be validated on held-out real TESS targets, known planets, known eclipsing binaries, synthetic injections, and negative controls before being used for scientific claims.",
    ]
    if result.warnings:
        lines += ["", "## Warnings", *[f"- {w}" for w in result.warnings]]
    return "\n".join(lines) + "\n"


def cross_validated_oof_predictions(
    catalog: pd.DataFrame,
    label_col: str = "label",
    feature_columns: list[str] | None = None,
    n_splits: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """Out-of-fold predictions for stronger validation on curated labels."""
    _require_sklearn()
    X, y, feature_columns, meta = prepare_ml_frame(catalog, label_col=label_col, feature_columns=feature_columns)
    assert y is not None
    counts = y.value_counts()
    if len(counts) < 2 or counts.min() < 2:
        raise ValueError("Need at least two examples per class for out-of-fold validation.")
    n_splits = min(n_splits, int(counts.min()))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    rows = []
    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):
        est = _build_base_estimator(random_state=random_state + fold)
        est.fit(X.iloc[train_idx], y.iloc[train_idx])
        probs = est.predict_proba(X.iloc[test_idx])
        classes = list(est.classes_)
        pred = est.predict(X.iloc[test_idx])
        fold_df = meta.iloc[test_idx].copy()
        fold_df["true_label"] = y.iloc[test_idx].values
        fold_df["oof_predicted_class"] = pred
        fold_df["oof_confidence"] = probs.max(axis=1)
        fold_df["fold"] = fold
        for cls in CANONICAL_CLASSES:
            fold_df[f"oof_prob_{cls}"] = probs[:, classes.index(cls)] if cls in classes else 0.0
        rows.append(fold_df)
    return pd.concat(rows, ignore_index=True)
