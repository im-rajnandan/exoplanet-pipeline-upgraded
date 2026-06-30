import builtins
import numpy as np

from exoplanet_pipeline.config import PipelineConfig
from exoplanet_pipeline.synthetic import make_synthetic_transit_lc
from exoplanet_pipeline.preprocess import preprocess_raw_lightcurve


def test_preprocess_normalizes_flux():
    raw = make_synthetic_transit_lc()
    clean = preprocess_raw_lightcurve(raw, PipelineConfig())
    assert clean.status == "OK"
    assert abs(np.nanmedian(clean.flux_normalized) - 1.0) < 1e-3
    assert abs(np.nanmedian(clean.flux_detrended) - 1.0) < 1e-3
    assert clean.selected_flux_source == "PDCSAP"
    assert clean.qc["n_final"] > 1000


def test_crowdsap_is_recorded_not_rejected():
    raw = make_synthetic_transit_lc(crowdsap=0.45)
    clean = preprocess_raw_lightcurve(raw, PipelineConfig())
    assert clean.status == "OK"
    assert clean.qc["crowdsap"] == 0.45
    assert "crowding_risk" in clean.qc


def test_wotan_fallback_is_reported(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "wotan":
            raise ImportError("forced missing wotan")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    raw = make_synthetic_transit_lc()
    clean = preprocess_raw_lightcurve(raw, PipelineConfig(detrend_method="wotan_biweight", detection_use_variants=False))
    assert clean.status == "OK"
    assert any(w.startswith("WOTAN_BIWEIGHT_FAILED_USED_ROLLING_MEDIAN") for w in clean.warnings)
