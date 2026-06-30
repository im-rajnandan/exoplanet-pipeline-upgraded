import numpy as np

from exoplanet_pipeline.config import PipelineConfig
from exoplanet_pipeline.synthetic import make_synthetic_transit_lc, make_synthetic_eb_lc, make_synthetic_blend_lc
from exoplanet_pipeline.preprocess import preprocess_raw_lightcurve
from exoplanet_pipeline.schema import CandidateSignal
from exoplanet_pipeline.fit import refine_candidate_parameters
from exoplanet_pipeline.vetting import extract_vetting_features, secondary_eclipse_test
from exoplanet_pipeline.classify import classify_candidate_rule_based


def _candidate_from_truth(raw, period, t0, duration_hours, depth_ppm):
    return CandidateSignal(
        tic_id=raw.tic_id,
        sector=raw.sector,
        candidate_id=1,
        period_days=period,
        epoch_time=t0,
        duration_days=duration_hours / 24,
        depth_fraction=depth_ppm * 1e-6,
        depth_ppm=depth_ppm,
        snr=20,
        local_snr=20,
        sde=12,
        fap=None,
        n_transits=5,
        n_full_transits=5,
        n_in_transit_points=100,
        detection_method="truth_seeded",
        flux_source="PDCSAP",
        detrend_variant="default",
        status="STRONG_DETECTION",
    )


def test_secondary_eclipse_does_not_confuse_primary_with_secondary():
    raw = make_synthetic_transit_lc(period_days=3.0, depth_ppm=1000, duration_hours=2, noise_ppm=50, random_seed=1)
    clean = preprocess_raw_lightcurve(raw, PipelineConfig(detrend_method="none"))
    sec = secondary_eclipse_test(clean.time, clean.flux_detrended, 3.0, 1.0, 2/24, 1000e-6)
    assert sec["secondary_sigma"] < 5


def test_eb_secondary_is_detected():
    raw = make_synthetic_eb_lc(period_days=4.0, primary_depth_ppm=20000, secondary_depth_ppm=8000, duration_hours=3, noise_ppm=100, random_seed=2)
    clean = preprocess_raw_lightcurve(raw, PipelineConfig(detrend_method="none"))
    sec = secondary_eclipse_test(clean.time, clean.flux_detrended, 4.0, 1.0, 3/24, 20000e-6)
    assert sec["secondary_sigma"] > 5
    assert abs(sec["secondary_phase"] - 0.5) < 0.05


def test_blend_centroid_increases_blend_score():
    raw = make_synthetic_blend_lc(centroid_shift_pix=0.05, noise_ppm=100, random_seed=3)
    clean = preprocess_raw_lightcurve(raw, PipelineConfig(detrend_method="none"))
    cand = _candidate_from_truth(raw, 3.5, 1.0, 2.0, 1200)
    fit = refine_candidate_parameters(clean, cand, n_bootstrap=50)
    vet = extract_vetting_features(clean, cand, fit)
    cls = classify_candidate_rule_based(cand, fit, vet)
    assert vet.centroid_shift_sigma > 3
    assert cls.blend_score > 0.3
