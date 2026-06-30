from __future__ import annotations

import numpy as np
from .schema import RawLightCurve


def make_synthetic_transit_lc(
    tic_id: int = 999001,
    sector: int = 1,
    period_days: float = 3.0,
    depth_ppm: float = 1000.0,
    duration_hours: float = 2.0,
    baseline_days: float = 27.0,
    cadence_minutes: float = 2.0,
    noise_ppm: float = 300.0,
    t0: float = 1.0,
    crowdsap: float = 0.95,
    random_seed: int = 42,
) -> RawLightCurve:
    rng = np.random.default_rng(random_seed)
    cadence_days = cadence_minutes / (24 * 60)
    time = np.arange(0, baseline_days, cadence_days)
    flux = np.ones_like(time)
    duration_days = duration_hours / 24
    phase_time = ((time - t0 + 0.5 * period_days) % period_days) - 0.5 * period_days
    in_tr = np.abs(phase_time) < 0.5 * duration_days
    flux[in_tr] -= depth_ppm * 1e-6
    # Add mild slow variability to make preprocessing realistic.
    flux += 250e-6 * np.sin(2 * np.pi * time / 9.0)
    flux += rng.normal(0, noise_ppm * 1e-6, size=len(time))
    flux_counts = flux * 1e5
    flux_err = np.ones_like(flux_counts) * noise_ppm * 1e-6 * 1e5
    quality = np.zeros_like(time, dtype=int)
    metadata = {
        "tic_id": tic_id,
        "sector": sector,
        "crowdsap": crowdsap,
        "flfrcsap": 0.95,
        "synthetic": True,
        "true_period_days": period_days,
        "true_depth_ppm": depth_ppm,
        "true_duration_hours": duration_hours,
        "true_t0": t0,
    }
    return RawLightCurve(
        tic_id=tic_id,
        sector=sector,
        time=time,
        sap_flux=flux_counts.copy(),
        sap_flux_err=flux_err.copy(),
        pdcsap_flux=flux_counts.copy(),
        pdcsap_flux_err=flux_err.copy(),
        quality=quality,
        centroid_col=np.ones_like(time) * 100.0 + rng.normal(0, 0.001, size=len(time)),
        centroid_row=np.ones_like(time) * 100.0 + rng.normal(0, 0.001, size=len(time)),
        metadata=metadata,
        status="RAW_LOADED",
    )


def make_synthetic_eb_lc(
    tic_id: int = 999002,
    sector: int = 1,
    period_days: float = 4.0,
    primary_depth_ppm: float = 20000.0,
    secondary_depth_ppm: float = 6000.0,
    duration_hours: float = 3.0,
    baseline_days: float = 27.0,
    cadence_minutes: float = 2.0,
    noise_ppm: float = 500.0,
    t0: float = 1.0,
    crowdsap: float = 0.95,
    random_seed: int = 43,
) -> RawLightCurve:
    """Synthetic eclipsing-binary-like light curve with primary and secondary eclipses."""
    rng = np.random.default_rng(random_seed)
    cadence_days = cadence_minutes / (24 * 60)
    time = np.arange(0, baseline_days, cadence_days)
    flux = np.ones_like(time)
    duration_days = duration_hours / 24
    primary_phase_time = ((time - t0 + 0.5 * period_days) % period_days) - 0.5 * period_days
    secondary_t0 = t0 + 0.5 * period_days
    secondary_phase_time = ((time - secondary_t0 + 0.5 * period_days) % period_days) - 0.5 * period_days
    flux[np.abs(primary_phase_time) < 0.5 * duration_days] -= primary_depth_ppm * 1e-6
    flux[np.abs(secondary_phase_time) < 0.5 * duration_days] -= secondary_depth_ppm * 1e-6
    flux += 400e-6 * np.sin(2 * np.pi * time / 11.0)
    flux += rng.normal(0, noise_ppm * 1e-6, size=len(time))
    flux_counts = flux * 1e5
    flux_err = np.ones_like(flux_counts) * noise_ppm * 1e-6 * 1e5
    metadata = {
        "tic_id": tic_id,
        "sector": sector,
        "crowdsap": crowdsap,
        "flfrcsap": 0.95,
        "synthetic": True,
        "synthetic_type": "eclipsing_binary",
        "true_period_days": period_days,
        "true_primary_depth_ppm": primary_depth_ppm,
        "true_secondary_depth_ppm": secondary_depth_ppm,
        "true_duration_hours": duration_hours,
        "true_t0": t0,
    }
    return RawLightCurve(
        tic_id=tic_id,
        sector=sector,
        time=time,
        sap_flux=flux_counts.copy(),
        sap_flux_err=flux_err.copy(),
        pdcsap_flux=flux_counts.copy(),
        pdcsap_flux_err=flux_err.copy(),
        quality=np.zeros_like(time, dtype=int),
        centroid_col=np.ones_like(time) * 100.0 + rng.normal(0, 0.001, size=len(time)),
        centroid_row=np.ones_like(time) * 100.0 + rng.normal(0, 0.001, size=len(time)),
        metadata=metadata,
        status="RAW_LOADED",
    )


def make_synthetic_blend_lc(
    tic_id: int = 999003,
    sector: int = 1,
    period_days: float = 3.5,
    observed_depth_ppm: float = 1200.0,
    duration_hours: float = 2.0,
    centroid_shift_pix: float = 0.03,
    baseline_days: float = 27.0,
    cadence_minutes: float = 2.0,
    noise_ppm: float = 350.0,
    t0: float = 1.0,
    crowdsap: float = 0.55,
    random_seed: int = 44,
) -> RawLightCurve:
    """Synthetic blended/off-target transit-like event with centroid motion."""
    raw = make_synthetic_transit_lc(
        tic_id=tic_id,
        sector=sector,
        period_days=period_days,
        depth_ppm=observed_depth_ppm,
        duration_hours=duration_hours,
        baseline_days=baseline_days,
        cadence_minutes=cadence_minutes,
        noise_ppm=noise_ppm,
        t0=t0,
        crowdsap=crowdsap,
        random_seed=random_seed,
    )
    duration_days = duration_hours / 24
    phase_time = ((raw.time - t0 + 0.5 * period_days) % period_days) - 0.5 * period_days
    in_tr = np.abs(phase_time) < 0.5 * duration_days
    raw.centroid_col = raw.centroid_col.copy()
    raw.centroid_row = raw.centroid_row.copy()
    raw.centroid_col[in_tr] += centroid_shift_pix
    raw.centroid_row[in_tr] += 0.5 * centroid_shift_pix
    raw.metadata.update({
        "synthetic_type": "blend",
        "true_centroid_shift_pix": centroid_shift_pix,
        "true_observed_depth_ppm": observed_depth_ppm,
    })
    return raw
