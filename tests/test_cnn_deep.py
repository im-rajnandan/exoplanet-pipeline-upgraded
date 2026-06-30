import numpy as np
import pytest


torch = pytest.importorskip("torch")

from exoplanet_pipeline.cnn import (
    CNNModelConfig,
    CNNTrainingExample,
    create_cnn_model,
    predict_cnn_candidate_views,
    train_cnn_classifier,
)


def _toy_example(label: str, tic_id: int = 1) -> CNNTrainingExample:
    rng = np.random.default_rng(tic_id)
    global_flux = rng.normal(0, 0.1, size=(2, 1001)).astype(np.float32)
    global_flux[1] = 1.0
    local_views = rng.normal(0, 0.1, size=(6, 2, 401)).astype(np.float32)
    local_views[:, 1, :] = 1.0
    scalars = rng.normal(0, 1, size=25).astype(np.float32)
    return CNNTrainingExample(
        global_flux=global_flux,
        local_views=local_views,
        scalar_features=scalars,
        scalar_feature_names=list(CNNModelConfig().scalar_feature_names),
        canonical_label=label,
        metadata={"tic_id": tic_id, "candidate_id": 1},
    )


def test_tiny_cnn_forward_pass_cpu():
    config = CNNModelConfig(conv_channels=4, view_embedding_dim=8, scalar_embedding_dim=4, fusion_hidden_dim=16)
    model = create_cnn_model(config)
    out = model(
        torch.zeros(2, 2, 1001),
        torch.zeros(2, 6, 2, 401),
        torch.zeros(2, len(config.scalar_feature_names)),
    )
    assert out["class_logits"].shape == (2, len(config.canonical_classes))
    assert out["binary_logit"].shape == (2,)


def test_tiny_cnn_training_smoke_cpu():
    config = CNNModelConfig(conv_channels=4, view_embedding_dim=8, scalar_embedding_dim=4, fusion_hidden_dim=16)
    examples = [
        _toy_example("PLANETARY_TRANSIT_CANDIDATE", 1),
        _toy_example("ECLIPSING_BINARY", 2),
        _toy_example("PLANETARY_TRANSIT_CANDIDATE", 3),
        _toy_example("ECLIPSING_BINARY", 4),
    ]
    result = train_cnn_classifier(examples, config=config, epochs=1, batch_size=2, seed=7)
    assert result.metrics["epochs"] == 1
    assert result.metrics["n_examples"] == 4


def test_cnn_guardrails_can_downgrade_high_planet_output():
    config = CNNModelConfig(conv_channels=4, view_embedding_dim=8, scalar_embedding_dim=4, fusion_hidden_dim=16)

    class DummyModel:
        def __call__(self, global_flux, local_views, scalars):
            logits = torch.full((1, len(config.canonical_classes)), -5.0)
            logits[0, config.canonical_classes.index("PLANETARY_TRANSIT_CANDIDATE")] = 5.0
            return {"class_logits": logits, "binary_logit": torch.tensor([5.0])}

    ex = _toy_example("PLANETARY_TRANSIT_CANDIDATE", 10)
    bundle = {
        "model": DummyModel(),
        "config": config,
        "scalar_scaler": {
            "feature_names": config.scalar_feature_names,
            "impute": [0.0] * len(config.scalar_feature_names),
            "mean": [0.0] * len(config.scalar_feature_names),
            "scale": [1.0] * len(config.scalar_feature_names),
        },
    }
    row = {
        "vet_secondary_sigma": 10.0,
        "vet_secondary_to_primary_ratio": 0.2,
        "vet_odd_even_sigma": 0.0,
        "vet_centroid_shift_sigma": 0.0,
        "vet_crowding_risk": 0.0,
        "vet_data_quality_score": 0.9,
        "fit_snr": 20.0,
    }
    pred = predict_cnn_candidate_views(bundle, ex, catalog_row=row)
    assert pred["cnn_predicted_class"] == "PLANETARY_TRANSIT_CANDIDATE"
    assert pred["final_predicted_class"] == "ECLIPSING_BINARY"
    assert "guardrail_strong_secondary_eclipse" in pred["final_classifier_warnings"]
