import numpy as np
import pandas as pd

from exoplanet_pipeline.cnn_views import CNN_VIEW_NAMES, build_cnn_candidate_views
from exoplanet_pipeline.config import PipelineConfig
from exoplanet_pipeline.fit import refine_candidate_parameters
from exoplanet_pipeline.pipeline_parts_1_to_8 import attach_ai_and_uncertainty
from exoplanet_pipeline.preprocess import preprocess_raw_lightcurve
from exoplanet_pipeline.schema import CandidateSignal
from exoplanet_pipeline.synthetic import make_synthetic_transit_lc
from exoplanet_pipeline.vetting import extract_vetting_features


def _candidate_from_truth(raw):
    return CandidateSignal(
        tic_id=raw.tic_id,
        sector=raw.sector,
        candidate_id=1,
        period_days=3.0,
        epoch_time=1.0,
        duration_days=2 / 24,
        depth_fraction=1000e-6,
        depth_ppm=1000,
        snr=20,
        local_snr=20,
        sde=12,
        fap=None,
        n_transits=8,
        n_full_transits=8,
        n_in_transit_points=100,
        detection_method="truth_seeded",
        flux_source="PDCSAP",
        detrend_variant="default",
        status="STRONG_DETECTION",
    )


def _view_fixture(with_centroids=True):
    raw = make_synthetic_transit_lc(period_days=3.0, depth_ppm=1000, duration_hours=2, noise_ppm=50, random_seed=5)
    if not with_centroids:
        raw.centroid_col = None
        raw.centroid_row = None
    clean = preprocess_raw_lightcurve(raw, PipelineConfig(detrend_method="none"))
    cand = _candidate_from_truth(raw)
    fit = refine_candidate_parameters(clean, cand, n_bootstrap=20)
    vet = extract_vetting_features(clean, cand, fit)
    return clean, cand, fit, vet


def test_cnn_view_builder_shapes_and_masks():
    clean, cand, fit, vet = _view_fixture()
    ex = build_cnn_candidate_views(clean, cand, fit, vet)
    assert set(CNN_VIEW_NAMES) == set(ex.views)
    assert ex.views["global_flux"].shape == (2, 1001)
    for name in CNN_VIEW_NAMES:
        if name != "global_flux":
            assert ex.views[name].shape == (2, 401)
        assert np.isfinite(ex.views[name]).all()
        assert set(np.unique(ex.views[name][1])).issubset({0.0, 1.0})
    assert ex.scalar_vector().shape[0] >= 20


def test_cnn_view_builder_missing_centroids_are_explicit():
    clean, cand, fit, vet = _view_fixture(with_centroids=False)
    ex = build_cnn_candidate_views(clean, cand, fit, vet)
    assert np.all(ex.views["centroid_col_local"][1] == 0)
    assert np.all(ex.views["centroid_row_local"][1] == 0)


def test_cnn_view_builder_is_deterministic():
    clean, cand, fit, vet = _view_fixture()
    a = build_cnn_candidate_views(clean, cand, fit, vet)
    b = build_cnn_candidate_views(clean, cand, fit, vet)
    for name in CNN_VIEW_NAMES:
        np.testing.assert_allclose(a.views[name], b.views[name])
    np.testing.assert_allclose(a.scalar_vector(), b.scalar_vector(), equal_nan=True)


def test_attach_ai_and_uncertainty_adds_cnn_columns_with_precedence(monkeypatch):
    clean, cand, fit, vet = _view_fixture()
    catalog = pd.DataFrame([
        {
            "tic_id": cand.tic_id,
            "sector": cand.sector,
            "candidate_id": cand.candidate_id,
            "fit_snr": fit.snr,
            "vet_secondary_sigma": vet.secondary_sigma,
            "vet_secondary_to_primary_ratio": vet.secondary_to_primary_ratio,
        }
    ])
    result = {
        "clean": clean,
        "detection": type("_D", (), {"candidates": [cand]})(),
        "fitted_candidates": [cand],
        "fit_results": [fit],
        "vetting_results": [vet],
        "classification_results": [],
        "catalog": catalog,
    }
    from exoplanet_pipeline import pipeline_parts_1_to_8 as p18

    def fake_predict(*args, **kwargs):
        return {
            "cnn_predicted_class": "PLANETARY_TRANSIT_CANDIDATE",
            "cnn_confidence": 0.9,
            "cnn_binary_planet_probability": 0.95,
            "cnn_model_version": "test",
            "final_predicted_class": "PLANETARY_TRANSIT_CANDIDATE",
            "final_confidence": 0.88,
            "final_classifier_method": "cnn_plus_physical_guardrails",
            "final_classifier_warnings": "",
        }

    monkeypatch.setattr(p18, "predict_cnn_candidate_views", fake_predict)
    out = attach_ai_and_uncertainty(result, cnn_bundle={"fake": True})
    assert "cnn_predicted_class" in out["catalog"].columns
    assert out["catalog"].loc[0, "final_classifier_method"] == "cnn_plus_physical_guardrails"
