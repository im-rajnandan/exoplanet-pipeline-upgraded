from pathlib import Path

import numpy as np

from exoplanet_pipeline.config import PipelineConfig
from exoplanet_pipeline.ingest import load_tess_fits
from exoplanet_pipeline.pipeline import run_parts_1_to_5_from_fits
from exoplanet_pipeline.preprocess import preprocess_raw_lightcurve


def _write_tess_like_fits(
    path: Path,
    include_time: bool = True,
    include_pdcsap: bool = True,
    include_quality: bool = True,
) -> None:
    from astropy.io import fits

    rng = np.random.default_rng(123)
    cadence_days = 30 / (24 * 60)
    time = np.arange(0, 27, cadence_days)
    flux = np.ones_like(time)
    period = 3.0
    t0 = 1.0
    duration_days = 3 / 24
    phase_time = ((time - t0 + 0.5 * period) % period) - 0.5 * period
    flux[np.abs(phase_time) < 0.5 * duration_days] -= 5000e-6
    flux += rng.normal(0, 100e-6, size=len(time))
    counts = flux * 1e5

    cols = []
    if include_time:
        cols.append(fits.Column(name="TIME", format="D", array=time))
    cols.append(fits.Column(name="SAP_FLUX", format="D", array=counts))
    cols.append(fits.Column(name="SAP_FLUX_ERR", format="D", array=np.ones_like(counts) * 10))
    if include_pdcsap:
        cols.append(fits.Column(name="PDCSAP_FLUX", format="D", array=counts))
        cols.append(fits.Column(name="PDCSAP_FLUX_ERR", format="D", array=np.ones_like(counts) * 10))
    if include_quality:
        cols.append(fits.Column(name="QUALITY", format="J", array=np.zeros_like(time, dtype=np.int32)))
    cols.append(fits.Column(name="MOM_CENTR1", format="D", array=np.ones_like(time) * 100))
    cols.append(fits.Column(name="MOM_CENTR2", format="D", array=np.ones_like(time) * 200))

    primary = fits.PrimaryHDU()
    primary.header["TICID"] = 123456789
    primary.header["SECTOR"] = 42
    primary.header["RADIUS"] = 1.0
    table = fits.BinTableHDU.from_columns(cols)
    table.header["CROWDSAP"] = 0.91
    table.header["FLFRCSAP"] = 0.95
    fits.HDUList([primary, table]).writeto(path)


def test_load_tess_fits_reads_standard_columns(tmp_path: Path):
    path = tmp_path / "standard_lc.fits"
    _write_tess_like_fits(path)
    raw = load_tess_fits(path)
    assert raw.status == "RAW_LOADED"
    assert raw.tic_id == 123456789
    assert raw.sector == 42
    assert raw.pdcsap_flux is not None
    assert raw.quality is not None
    assert raw.metadata["crowdsap"] == 0.91


def test_preprocess_fits_uses_sap_fallback_when_pdcsap_missing(tmp_path: Path):
    path = tmp_path / "sap_only_lc.fits"
    _write_tess_like_fits(path, include_pdcsap=False)
    raw = load_tess_fits(path)
    clean = preprocess_raw_lightcurve(raw, PipelineConfig(min_clean_points=300))
    assert clean.status == "OK"
    assert clean.selected_flux_source == "SAP"
    assert "PDCSAP_UNAVAILABLE_OR_INVALID_USED_SAP" in clean.warnings


def test_load_tess_fits_reports_missing_time_column(tmp_path: Path):
    path = tmp_path / "no_time_lc.fits"
    _write_tess_like_fits(path, include_time=False)
    raw = load_tess_fits(path)
    assert raw.status == "NO_TIME_COLUMN"


def test_parts_1_to_5_runs_from_generated_fits(tmp_path: Path):
    path = tmp_path / "standard_lc.fits"
    _write_tess_like_fits(path)
    cfg = PipelineConfig(
        n_periods=500,
        n_durations=6,
        min_clean_points=300,
        period_max_days=5.0,
        detection_use_variants=False,
        strong_snr_threshold=8.0,
    )
    result = run_parts_1_to_5_from_fits(str(path), config=cfg)
    assert result["clean"].status == "OK"
    assert not result["catalog"].empty
    recovered = result["detection"].best_candidate.period_days
    assert abs(recovered - 3.0) / 3.0 < 0.05
