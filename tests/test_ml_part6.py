import tempfile
from pathlib import Path

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
