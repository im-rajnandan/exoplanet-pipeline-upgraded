import tempfile
from pathlib import Path

import pandas as pd

from exoplanet_pipeline.ml import (
    normalize_label,
    train_ai_classifier,
    predict_ai_classifier,
    save_model_bundle,
    load_model_bundle,
)
from exoplanet_pipeline.ml_synthetic import make_synthetic_ml_feature_catalog


def test_normalize_label_aliases():
    assert normalize_label("planet") == "PLANETARY_TRANSIT_CANDIDATE"
    assert normalize_label("confirmed_planet") == "PLANETARY_TRANSIT_CANDIDATE"
    assert normalize_label("EB") == "ECLIPSING_BINARY"
    assert normalize_label("background EB") == "BLEND_OR_CONTAMINATED_SIGNAL"
    assert normalize_label("starspot") == "STELLAR_VARIABILITY"
    assert normalize_label("false positive") == "INSTRUMENTAL_OR_LOW_QUALITY_SYSTEMATIC"
    assert normalize_label("not_a_known_label") is None


def test_synthetic_ml_catalog_training_prediction_roundtrip():
    df = make_synthetic_ml_feature_catalog(n_per_class=12, random_seed=99)
    result = train_ai_classifier(df, label_col="label", calibrate=False, test_size=0.3, random_state=99)
    assert len(result.feature_columns) > 10
    assert len(result.class_names) >= 2
    assert "macro_f1" in result.test_metrics
    assert not result.feature_importance.empty

    pred_input = df.drop(columns=["label"]).head(10)
    pred = predict_ai_classifier(result.model_bundle, pred_input)
    assert "final_predicted_class" in pred.columns
    assert "final_confidence" in pred.columns
    assert pred["final_confidence"].between(0, 1.5).all()

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "model.joblib"
        save_model_bundle(result.model_bundle, path)
        loaded = load_model_bundle(path)
        pred2 = predict_ai_classifier(loaded, pred_input)
        assert len(pred2) == len(pred_input)


def test_training_drops_all_nan_feature_columns():
    df = pd.DataFrame(
        {
            "period_days": [1.0, 1.2, 2.0, 2.2, 3.0, 3.2],
            "depth_ppm": [800, 850, 1500, 1450, 4000, 4200],
            "fit_snr": [8, 9, 14, 15, 30, 32],
            "snr": [None, None, None, None, None, None],
            "label": ["planet", "planet", "planet", "false_positive", "false_positive", "false_positive"],
        }
    )

    result = train_ai_classifier(
        df,
        label_col="label",
        feature_columns=["period_days", "depth_ppm", "fit_snr", "snr"],
        calibrate=False,
        test_size=0.34,
        random_state=7,
    )

    assert "snr" not in result.feature_columns
    assert any("Dropped all-NaN feature columns" in warning for warning in result.warnings)
