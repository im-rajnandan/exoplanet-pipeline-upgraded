from exoplanet_pipeline.config import PipelineConfig
from exoplanet_pipeline.synthetic import make_synthetic_transit_lc
from exoplanet_pipeline.preprocess import preprocess_raw_lightcurve
from exoplanet_pipeline.detect import detect_candidates, make_transit_mask


def test_transit_mask_centers_periodic_events():
    time = __import__("numpy").array([0.0, 1.0, 2.0, 3.0, 3.05, 4.5])
    mask = make_transit_mask(time, period=3.0, t0=0.0, duration=0.2)
    assert mask[0]
    assert mask[3]
    assert not mask[-1]


def test_bls_recovers_simple_injected_period():
    raw = make_synthetic_transit_lc(period_days=3.0, depth_ppm=1500, duration_hours=2.5, noise_ppm=250)
    cfg = PipelineConfig(detection_method="bls", n_periods=700, period_max_days=5.0, strong_snr_threshold=8.0)
    clean = preprocess_raw_lightcurve(raw, cfg)
    result = detect_candidates(clean, cfg, use_variants=False)
    assert result.best_candidate is not None
    recovered = result.best_candidate.period_days
    assert abs(recovered - 3.0) / 3.0 < 0.03
    assert result.best_candidate.local_snr > 7
