from types import SimpleNamespace
import numpy as np
import pandas as pd

from exoplanet_pipeline.config import PipelineConfig
from exoplanet_pipeline.schema import CandidateSignal
from exoplanet_pipeline.synthetic import make_synthetic_transit_lc
from exoplanet_pipeline.pipeline import run_parts_1_to_5_from_raw
from exoplanet_pipeline.pipeline_parts_1_to_8 import attach_ai_and_uncertainty
from exoplanet_pipeline.uncertainty import estimate_candidate_uncertainty, add_uncertainty_columns
from exoplanet_pipeline.validation import validate_candidate_catalog
import exoplanet_pipeline.injection_recovery as injection_recovery
from exoplanet_pipeline.injection_recovery import (
    InjectionSpec,
    compact_injection_demo_grid,
    run_injection_recovery_grid,
    run_single_injection,
    summarize_injection_recovery,
)


def test_candidate_uncertainty_on_synthetic_planet():
    raw = make_synthetic_transit_lc(period_days=3.0, depth_ppm=1500, noise_ppm=250, random_seed=123)
    cfg = PipelineConfig(n_periods=800, n_durations=6, min_clean_points=300, detection_use_variants=False)
    res = run_parts_1_to_5_from_raw(raw, config=cfg)
    assert not res["catalog"].empty
    unc = estimate_candidate_uncertainty(
        res["clean"],
        res["detection"].candidates[0],
        res["fit_results"][0],
        res["vetting_results"][0],
        res["classification_results"][0],
        n_bootstrap=20,
        random_seed=123,
    )
    assert np.isfinite(unc.final_confidence)
    assert 0.0 <= unc.final_confidence <= 1.0
    assert unc.confidence_level in {"HIGH", "MEDIUM", "LOW", "VERY_LOW"}


def test_uncertainty_columns_join_by_candidate_id_when_catalog_is_reordered():
    raw = make_synthetic_transit_lc(period_days=3.0, depth_ppm=1500, noise_ppm=250, random_seed=321)
    cfg = PipelineConfig(n_periods=500, n_durations=5, min_clean_points=300, detection_use_variants=True)
    res = run_parts_1_to_5_from_raw(raw, config=cfg)
    assert len(res["fitted_candidates"]) >= 2

    res["catalog"] = res["catalog"].sort_values("candidate_id", ascending=False).reset_index(drop=True)
    out = attach_ai_and_uncertainty(res)
    fit_depth_by_candidate = {
        fit.candidate_id: fit.depth_ppm
        for fit in res["fit_results"]
    }
    for _, row in out["catalog"].iterrows():
        assert np.isclose(row["unc_depth_ppm"], fit_depth_by_candidate[row["candidate_id"]])


def test_catalog_uncertainty_columns():
    df = pd.DataFrame({
        "fit_depth_ppm": [1000.0, 2000.0],
        "fit_depth_err_ppm": [50.0, np.nan],
        "fit_snr": [12.0, 8.0],
        "vet_red_noise_proxy": [0.1, 0.5],
    })
    out = add_uncertainty_columns(df)
    assert "unc_effective_snr" in out.columns
    assert out["unc_depth_err_ppm"].notna().all()


def test_validation_report_with_labels():
    df = pd.DataFrame({
        "label": ["planet", "eb", "blend", "quiet"],
        "final_predicted_class": [
            "PLANETARY_TRANSIT_CANDIDATE",
            "ECLIPSING_BINARY",
            "BLEND_OR_CONTAMINATED_SIGNAL",
            "NO_SIGNIFICANT_SIGNAL",
        ],
        "final_confidence": [0.9, 0.8, 0.7, 0.6],
        "fit_period_days": [3.0, 2.0, 4.0, np.nan],
        "true_period_days": [3.02, 2.1, 4.1, np.nan],
    })
    report = validate_candidate_catalog(df, label_col="label")
    assert report.classification_metrics["available"] is True
    assert report.classification_metrics["accuracy"] >= 0.75
    assert report.detection_metrics["available"] is True


def test_small_injection_recovery_grid_runs():
    specs = [
        InjectionSpec("p", "PLANETARY_TRANSIT_CANDIDATE", 3.0, 1800.0, 2.0, 250.0, 0.95, random_seed=50),
        InjectionSpec("b", "BLEND_OR_CONTAMINATED_SIGNAL", 3.5, 2000.0, 2.0, 300.0, 0.55, centroid_shift_pix=0.03, random_seed=51),
    ]
    cfg = PipelineConfig(n_periods=500, n_durations=5, min_clean_points=300, detection_use_variants=False)
    df = run_injection_recovery_grid(specs=specs, config=cfg, n_bootstrap=10)
    assert len(df) == 2
    assert "detected" in df.columns
    summary = summarize_injection_recovery(df)
    assert summary["n"] == 2


def test_compact_injection_demo_grid_is_class_balanced():
    specs = compact_injection_demo_grid(random_seed=42, n_per_class=2)
    counts = pd.Series([spec.label for spec in specs]).value_counts().to_dict()
    assert counts == {
        "PLANETARY_TRANSIT_CANDIDATE": 2,
        "ECLIPSING_BINARY": 2,
        "BLEND_OR_CONTAMINATED_SIGNAL": 2,
    }


def test_injection_uncertainty_uses_selected_candidate_id(monkeypatch):
    cand1 = CandidateSignal(
        tic_id=1,
        sector=1,
        candidate_id=1,
        period_days=3.0,
        epoch_time=1.0,
        duration_days=2 / 24,
        depth_fraction=1000e-6,
        depth_ppm=1000,
        snr=8,
        local_snr=8,
        sde=6,
        fap=None,
        n_transits=3,
        n_full_transits=3,
        n_in_transit_points=20,
        detection_method="test",
        flux_source="PDCSAP",
        detrend_variant="default",
        status="WEAK_DETECTION",
    )
    cand2 = CandidateSignal(
        tic_id=1,
        sector=1,
        candidate_id=2,
        period_days=3.1,
        epoch_time=1.0,
        duration_days=2 / 24,
        depth_fraction=2000e-6,
        depth_ppm=2000,
        snr=18,
        local_snr=18,
        sde=10,
        fap=None,
        n_transits=3,
        n_full_transits=3,
        n_in_transit_points=20,
        detection_method="test",
        flux_source="PDCSAP",
        detrend_variant="0.50d",
        status="STRONG_DETECTION",
    )
    fit1 = SimpleNamespace(candidate_id=1, depth_ppm=111.0)
    fit2 = SimpleNamespace(candidate_id=2, depth_ppm=222.0)
    result = {
        "clean": SimpleNamespace(),
        "catalog": pd.DataFrame([
            {
                "candidate_id": 1,
                "fit_snr": 8.0,
                "fit_period_days": 3.0,
                "fit_depth_ppm": 111.0,
                "fit_duration_days": 2 / 24,
                "status": "WEAK_DETECTION",
            },
            {
                "candidate_id": 2,
                "fit_snr": 18.0,
                "fit_period_days": 3.1,
                "fit_depth_ppm": 222.0,
                "fit_duration_days": 2 / 24,
                "status": "STRONG_DETECTION",
            },
        ]),
        "fitted_candidates": [cand1, cand2],
        "fit_results": [fit1, fit2],
        "vetting_results": [SimpleNamespace(candidate_id=1), SimpleNamespace(candidate_id=2)],
        "classification_results": [SimpleNamespace(candidate_id=1), SimpleNamespace(candidate_id=2)],
        "detection": SimpleNamespace(candidates=[cand1, cand2], status="STRONG_DETECTION"),
    }

    class FakeUncertainty:
        def __init__(self, candidate_id, depth_ppm):
            self.candidate_id = candidate_id
            self.depth_ppm = depth_ppm

        def to_dict(self):
            return {"tic_id": 1, "sector": 1, "candidate_id": self.candidate_id, "depth_ppm": self.depth_ppm}

    def fake_run_parts_1_to_5_from_raw(raw, config=None):
        return result

    def fake_estimate_candidate_uncertainty(clean, cand, fit, vet, cls, **kwargs):
        return FakeUncertainty(cand.candidate_id, fit.depth_ppm)

    monkeypatch.setattr(injection_recovery, "run_parts_1_to_5_from_raw", fake_run_parts_1_to_5_from_raw)
    monkeypatch.setattr(injection_recovery, "estimate_candidate_uncertainty", fake_estimate_candidate_uncertainty)

    row = run_single_injection(
        InjectionSpec("p", "PLANETARY_TRANSIT_CANDIDATE", 3.0, 1000.0, 2.0, 300.0, 0.95),
        config=PipelineConfig(),
        n_bootstrap=1,
    )
    assert row["candidate_id"] == 2
    assert row["unc_depth_ppm"] == 222.0
