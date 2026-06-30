from pathlib import Path
import pandas as pd

from exoplanet_pipeline.final_catalog import harmonize_candidate_catalog, summarize_final_catalog, validate_final_catalog_schema
from exoplanet_pipeline.batch import BatchRunConfig, run_raw_lightcurve_batch, run_fits_file_batch
from exoplanet_pipeline.config import PipelineConfig
from exoplanet_pipeline.synthetic import make_synthetic_transit_lc
from exoplanet_pipeline.final_outputs import generate_submission_package_outputs


def test_harmonize_candidate_catalog_basic():
    df = pd.DataFrame([
        {
            "tic_id": 1,
            "sector": 2,
            "candidate_id": 0,
            "final_predicted_class": "PLANETARY_TRANSIT_CANDIDATE",
            "final_confidence": 0.82,
            "fit_period_days": 3.1,
            "fit_duration_days": 0.1,
            "fit_depth_ppm": 900,
            "fit_snr": 12,
            "vet_secondary_sigma": 0.2,
            "vet_odd_even_sigma": 0.1,
            "vet_centroid_shift_sigma": 0.5,
            "vet_crowding_risk": 0.05,
            "vet_data_quality_score": 0.9,
        }
    ])
    out = harmonize_candidate_catalog(df)
    assert "final_science_class" in out.columns
    assert abs(out.loc[0, "duration_hours"] - 2.4) < 1e-9
    assert out.loc[0, "science_priority_rank"] == 1
    assert "HIGH_PRIORITY" in out.loc[0, "recommended_action"]
    assert validate_final_catalog_schema(out) == []


def test_final_catalog_schema_validation_flags_bad_values():
    df = pd.DataFrame([
        {
            "tic_id": 1,
            "sector": 2,
            "candidate_id": 0,
            "final_predicted_class": "PLANETARY_TRANSIT_CANDIDATE",
            "final_confidence": 0.82,
            "fit_period_days": 3.1,
            "fit_duration_days": 0.1,
            "fit_depth_ppm": 900,
            "fit_snr": 12,
            "vet_data_quality_score": 0.9,
        }
    ])
    final = harmonize_candidate_catalog(df)
    bad = final.copy()
    bad.loc[0, "final_science_class"] = "NOT_A_REAL_CLASS"
    bad.loc[0, "period_days"] = -3.1
    issues = validate_final_catalog_schema(bad)
    assert any("unknown final_science_class" in issue for issue in issues)
    assert any("period_days" in issue for issue in issues)


def test_submission_assets_generation(tmp_path: Path):
    df = pd.DataFrame([
        {
            "tic_id": 1,
            "sector": 2,
            "candidate_id": 0,
            "final_predicted_class": "PLANETARY_TRANSIT_CANDIDATE",
            "final_confidence": 0.82,
            "fit_period_days": 3.1,
            "fit_duration_days": 0.1,
            "fit_depth_ppm": 900,
            "fit_snr": 12,
            "vet_data_quality_score": 0.9,
        }
    ])
    paths = generate_submission_package_outputs(df, tmp_path)
    assert paths["final_catalog"].exists()
    assert paths["three_page_report_draft"].exists()
    assert paths["candidate_review"].exists()


def test_small_raw_batch_runs(tmp_path: Path):
    raw = make_synthetic_transit_lc(tic_id=12345, period_days=3.0, depth_ppm=1500, noise_ppm=250, random_seed=11)
    cfg = PipelineConfig(n_periods=400, detection_method="bls", make_plots=False, detection_use_variants=False)
    bcfg = BatchRunConfig(output_dir=tmp_path, cache_dir=tmp_path / "cache", resume=False, write_heartbeat_every=0)
    result = run_raw_lightcurve_batch([raw], pipeline_config=cfg, batch_config=bcfg)
    assert result["target_summary"].shape[0] == 1
    assert (tmp_path / "batch_final_candidate_catalog.csv").exists()
    assert (tmp_path / "batch_target_summary.csv").exists()
    assert (tmp_path / "batch_failure_log.csv").exists()


def test_empty_fits_batch_writes_empty_outputs(tmp_path: Path):
    cfg = PipelineConfig(n_periods=100, make_plots=False)
    bcfg = BatchRunConfig(output_dir=tmp_path, cache_dir=tmp_path / "cache", resume=False, write_heartbeat_every=0)
    result = run_fits_file_batch([], pipeline_config=cfg, batch_config=bcfg)
    assert result["raw_candidate_catalog"].empty
    assert result["final_candidate_catalog"].empty
    assert result["target_summary"].empty
    assert (tmp_path / "batch_raw_candidate_catalog.csv").exists()
    assert (tmp_path / "batch_final_candidate_catalog.csv").exists()
    assert (tmp_path / "batch_final_summary.json").exists()


def test_batch_resume_skips_zero_candidate_targets(tmp_path: Path):
    import numpy as np
    from exoplanet_pipeline.schema import RawLightCurve

    rng = np.random.default_rng(123)
    time = np.arange(0, 27, 2 / (24 * 60))
    flux = (1 + rng.normal(0, 300e-6, len(time))) * 1e5
    raw = RawLightCurve(
        tic_id=424242,
        sector=1,
        time=time,
        sap_flux=flux,
        sap_flux_err=np.ones_like(flux) * 30,
        pdcsap_flux=flux,
        pdcsap_flux_err=np.ones_like(flux) * 30,
        quality=np.zeros_like(time, dtype=int),
        metadata={"crowdsap": 0.95, "flfrcsap": 0.95},
        status="RAW_LOADED",
    )
    cfg = PipelineConfig(n_periods=300, n_durations=4, min_clean_points=300, detection_use_variants=False, make_plots=False)
    first_cfg = BatchRunConfig(output_dir=tmp_path, cache_dir=tmp_path / "cache", resume=False, write_heartbeat_every=0)
    first = run_raw_lightcurve_batch([raw], pipeline_config=cfg, batch_config=first_cfg)
    assert first["target_summary"].shape[0] == 1
    assert first["final_candidate_catalog"].empty
    assert (tmp_path / "cache" / "tic_424242_s1_00000_summary.json").exists()

    resume_cfg = BatchRunConfig(output_dir=tmp_path, cache_dir=tmp_path / "cache", resume=True, write_heartbeat_every=0)
    second = run_raw_lightcurve_batch([raw], pipeline_config=cfg, batch_config=resume_cfg)
    assert second["target_summary"].shape[0] == 1
    assert second["final_candidate_catalog"].empty
