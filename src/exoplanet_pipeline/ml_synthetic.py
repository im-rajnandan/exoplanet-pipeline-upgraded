from __future__ import annotations

import numpy as np
import pandas as pd

from .ml import CANONICAL_CLASSES


def make_synthetic_ml_feature_catalog(
    n_per_class: int = 120,
    random_seed: int = 123,
    include_uncertain: bool = True,
) -> pd.DataFrame:
    """Create a labeled feature catalog for Part 6 development.

    This does not replace the organizer's curated dataset. It exists so the ML
    training/evaluation/prediction code can be tested before real labels arrive.
    The generated features mimic the Parts 1-5 catalog columns.
    """
    rng = np.random.default_rng(random_seed)
    rows: list[dict] = []
    classes = list(CANONICAL_CLASSES)
    if not include_uncertain:
        classes.remove("UNCERTAIN_TRANSIT_LIKE_SIGNAL")

    tic_base = 880000000
    for class_idx, label in enumerate(classes):
        for i in range(n_per_class):
            tic_id = tic_base + class_idx * 100000 + i
            row = _sample_row_for_class(rng, label, tic_id=tic_id, sector=int(rng.integers(1, 70)), candidate_id=1)
            rows.append(row)
    df = pd.DataFrame(rows)
    # Shuffle so scripts exercise robust splitting rather than ordered classes.
    return df.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)


def _sample_row_for_class(rng: np.random.Generator, label: str, tic_id: int, sector: int, candidate_id: int) -> dict:
    # Shared random physical scale features.
    period = float(rng.uniform(0.4, 14.0))
    duration = float(rng.uniform(0.03, 0.22))
    n_transits = max(1, int(np.floor(27.0 / period)) + int(rng.integers(-1, 2)))
    n_transits = int(np.clip(n_transits, 1, 25))
    n_in = int(np.clip(n_transits * rng.integers(12, 70), 3, 2000))

    # Defaults represent an uncertain weak transit-like event. Classes modify them.
    snr = rng.normal(6.5, 1.5)
    sde = rng.normal(5.5, 1.4)
    depth_ppm = rng.lognormal(mean=np.log(1200), sigma=0.7)
    depth_ppm = float(np.clip(depth_ppm, 80, 50000))
    rp_rs = float(np.sqrt(depth_ppm * 1e-6))
    secondary_sigma = abs(rng.normal(0.8, 0.6))
    secondary_ratio = abs(rng.normal(0.01, 0.015))
    odd_even_sigma = abs(rng.normal(0.8, 0.6))
    centroid_sigma = abs(rng.normal(0.8, 0.7))
    centroid_pix = centroid_sigma * rng.uniform(0.0005, 0.003)
    crowdsap = float(np.clip(rng.normal(0.88, 0.08), 0.35, 1.0))
    crowding_risk = 1.0 - crowdsap
    v_shape = float(np.clip(rng.normal(0.72, 0.15), 0.05, 1.0))
    red_noise = float(np.clip(abs(rng.normal(0.12, 0.08)), 0.0, 0.8))
    dq = float(np.clip(rng.normal(0.82, 0.12), 0.05, 1.0))

    if label == "PLANETARY_TRANSIT_CANDIDATE":
        snr = rng.normal(15.0, 4.0)
        sde = rng.normal(11.5, 2.3)
        depth_ppm = float(np.clip(rng.lognormal(np.log(850), 0.65), 120, 6000))
        rp_rs = float(np.sqrt(depth_ppm * 1e-6))
        secondary_sigma = abs(rng.normal(0.6, 0.4))
        secondary_ratio = abs(rng.normal(0.005, 0.008))
        odd_even_sigma = abs(rng.normal(0.7, 0.5))
        centroid_sigma = abs(rng.normal(0.7, 0.45))
        centroid_pix = centroid_sigma * rng.uniform(0.0002, 0.0015)
        crowdsap = float(np.clip(rng.normal(0.91, 0.06), 0.65, 1.0))
        crowding_risk = 1.0 - crowdsap
        v_shape = float(np.clip(rng.normal(0.80, 0.10), 0.45, 1.0))
        red_noise = float(np.clip(abs(rng.normal(0.08, 0.05)), 0, 0.35))
        dq = float(np.clip(rng.normal(0.88, 0.08), 0.55, 1.0))

    elif label == "ECLIPSING_BINARY":
        snr = rng.normal(30.0, 9.0)
        sde = rng.normal(13.0, 3.0)
        depth_ppm = float(np.clip(rng.lognormal(np.log(25000), 0.55), 7000, 160000))
        rp_rs = float(np.sqrt(depth_ppm * 1e-6))
        secondary_sigma = float(np.clip(rng.normal(7.5, 3.0), 2.5, 25))
        secondary_ratio = float(np.clip(rng.normal(0.28, 0.14), 0.03, 0.85))
        odd_even_sigma = float(np.clip(rng.normal(3.8, 2.1), 0.2, 13.0))
        centroid_sigma = abs(rng.normal(1.2, 0.9))
        v_shape = float(np.clip(rng.normal(0.36, 0.16), 0.05, 0.75))
        dq = float(np.clip(rng.normal(0.82, 0.12), 0.40, 1.0))

    elif label == "BLEND_OR_CONTAMINATED_SIGNAL":
        snr = rng.normal(14.0, 4.5)
        sde = rng.normal(9.5, 2.2)
        depth_ppm = float(np.clip(rng.lognormal(np.log(1800), 0.75), 250, 18000))
        rp_rs = float(np.sqrt(depth_ppm * 1e-6))
        secondary_sigma = abs(rng.normal(1.4, 1.0))
        secondary_ratio = abs(rng.normal(0.03, 0.04))
        odd_even_sigma = abs(rng.normal(1.2, 1.0))
        centroid_sigma = float(np.clip(rng.normal(6.0, 2.2), 2.2, 18.0))
        centroid_pix = centroid_sigma * rng.uniform(0.003, 0.015)
        crowdsap = float(np.clip(rng.normal(0.58, 0.16), 0.15, 0.88))
        crowding_risk = 1.0 - crowdsap
        v_shape = float(np.clip(rng.normal(0.60, 0.20), 0.10, 0.95))
        dq = float(np.clip(rng.normal(0.75, 0.15), 0.30, 1.0))

    elif label == "STELLAR_VARIABILITY":
        snr = rng.normal(9.0, 3.0)
        sde = rng.normal(6.5, 2.0)
        depth_ppm = float(np.clip(rng.lognormal(np.log(2500), 0.9), 300, 40000))
        rp_rs = float(np.sqrt(depth_ppm * 1e-6))
        duration = float(np.clip(period * rng.uniform(0.09, 0.22), 0.15, 1.5))
        secondary_sigma = abs(rng.normal(1.0, 0.8))
        secondary_ratio = abs(rng.normal(0.02, 0.03))
        odd_even_sigma = abs(rng.normal(1.2, 0.9))
        centroid_sigma = abs(rng.normal(1.0, 0.8))
        v_shape = float(np.clip(rng.normal(0.20, 0.12), 0.01, 0.55))
        red_noise = float(np.clip(rng.normal(0.45, 0.18), 0.18, 0.95))
        dq = float(np.clip(rng.normal(0.72, 0.16), 0.25, 1.0))

    elif label == "INSTRUMENTAL_OR_LOW_QUALITY_SYSTEMATIC":
        snr = rng.normal(6.0, 2.5)
        sde = rng.normal(5.5, 2.4)
        depth_ppm = float(np.clip(rng.lognormal(np.log(1300), 1.0), 80, 60000))
        rp_rs = float(np.sqrt(depth_ppm * 1e-6))
        n_transits = int(rng.integers(1, 4))
        n_in = int(rng.integers(3, 90))
        secondary_sigma = abs(rng.normal(1.5, 1.3))
        odd_even_sigma = abs(rng.normal(1.5, 1.2))
        centroid_sigma = abs(rng.normal(2.0, 1.8))
        crowdsap = float(np.clip(rng.normal(0.74, 0.20), 0.15, 1.0))
        crowding_risk = 1.0 - crowdsap
        v_shape = float(np.clip(rng.normal(0.35, 0.25), 0.0, 1.0))
        red_noise = float(np.clip(rng.normal(0.35, 0.22), 0.0, 1.0))
        dq = float(np.clip(rng.normal(0.28, 0.14), 0.02, 0.55))

    elif label == "NO_SIGNIFICANT_SIGNAL":
        snr = float(np.clip(rng.normal(3.0, 1.2), 0.1, 6.2))
        sde = float(np.clip(rng.normal(3.2, 1.1), 0.2, 6.0))
        depth_ppm = float(np.clip(rng.lognormal(np.log(300), 0.8), 20, 2500))
        rp_rs = float(np.sqrt(depth_ppm * 1e-6))
        n_transits = int(rng.integers(0, 4))
        n_in = int(rng.integers(0, 60))
        secondary_sigma = abs(rng.normal(0.6, 0.5))
        odd_even_sigma = abs(rng.normal(0.6, 0.5))
        centroid_sigma = abs(rng.normal(0.6, 0.6))
        v_shape = float(np.clip(rng.normal(0.50, 0.30), 0, 1))
        red_noise = float(np.clip(abs(rng.normal(0.15, 0.12)), 0, 0.6))
        dq = float(np.clip(rng.normal(0.70, 0.20), 0.10, 1.0))

    elif label == "UNCERTAIN_TRANSIT_LIKE_SIGNAL":
        snr = float(np.clip(rng.normal(7.2, 1.8), 4.5, 11.0))
        sde = float(np.clip(rng.normal(6.6, 1.5), 3.5, 10.0))
        depth_ppm = float(np.clip(rng.lognormal(np.log(1200), 0.9), 100, 20000))
        rp_rs = float(np.sqrt(depth_ppm * 1e-6))
        secondary_sigma = float(np.clip(rng.normal(2.4, 1.6), 0.0, 6.5))
        secondary_ratio = float(np.clip(rng.normal(0.04, 0.05), 0.0, 0.25))
        odd_even_sigma = float(np.clip(rng.normal(2.0, 1.3), 0.0, 6.0))
        centroid_sigma = float(np.clip(rng.normal(2.2, 1.5), 0.0, 7.0))
        crowdsap = float(np.clip(rng.normal(0.75, 0.18), 0.25, 1.0))
        crowding_risk = 1.0 - crowdsap
        v_shape = float(np.clip(rng.normal(0.55, 0.22), 0.05, 0.95))
        red_noise = float(np.clip(rng.normal(0.22, 0.14), 0.0, 0.75))
        dq = float(np.clip(rng.normal(0.65, 0.18), 0.20, 1.0))

    snr = float(max(0.0, snr))
    sde = float(max(0.0, sde))
    local_snr = float(max(0.0, snr + rng.normal(0, 0.8)))
    corrected_depth = float(depth_ppm / max(crowdsap, 0.05))
    harmonic_risk = float(np.clip(0.08 + 0.12 * (odd_even_sigma > 2.5) + rng.normal(0, 0.07), 0, 1))

    # Rule-based scores as meta-features. Noisy but correlated.
    planet_score = float(np.clip((snr - 5) / 15 - 0.35 * (secondary_sigma > 4) - 0.35 * (centroid_sigma > 4) + rng.normal(0, 0.06), 0, 1))
    eb_score = float(np.clip(0.12 * (depth_ppm / 10000) + 0.08 * secondary_sigma + 0.07 * odd_even_sigma + rng.normal(0, 0.06), 0, 1))
    blend_score = float(np.clip(0.10 * centroid_sigma + 0.55 * max(0, crowding_risk - 0.2) + rng.normal(0, 0.06), 0, 1))
    stellar_score = float(np.clip(0.75 * red_noise + 0.4 * max(0, 0.4 - v_shape) + rng.normal(0, 0.06), 0, 1))
    systematic_score = float(np.clip(0.8 * max(0, 0.65 - dq) + 0.08 * max(0, 7 - snr) + rng.normal(0, 0.06), 0, 1))

    return {
        "tic_id": tic_id,
        "sector": sector,
        "candidate_id": candidate_id,
        "period_days": period,
        "duration_days": duration,
        "depth_fraction": depth_ppm * 1e-6,
        "depth_ppm": depth_ppm,
        "snr": snr,
        "local_snr": local_snr,
        "sde": sde,
        "n_transits": n_transits,
        "n_full_transits": max(0, n_transits - int(rng.integers(0, 2))),
        "n_in_transit_points": n_in,
        "fit_period_days": period * float(rng.normal(1.0, 0.001)),
        "fit_period_err_days": period * rng.uniform(1e-5, 5e-4),
        "fit_epoch_err_days": rng.uniform(0.0005, 0.015),
        "fit_duration_days": duration * float(rng.normal(1.0, 0.06)),
        "fit_duration_err_days": duration * rng.uniform(0.02, 0.18),
        "fit_depth_fraction": depth_ppm * 1e-6,
        "fit_depth_err_fraction": depth_ppm * 1e-6 / max(snr, 1.0),
        "fit_depth_ppm": depth_ppm * float(rng.normal(1.0, 0.06)),
        "fit_depth_err_ppm": max(20.0, depth_ppm / max(snr, 1.0)),
        "fit_rp_over_rstar": rp_rs,
        "fit_rp_earth": rp_rs * 109.2 if rng.random() > 0.25 else np.nan,
        "fit_stellar_radius_rsun": 1.0 if rng.random() > 0.25 else np.nan,
        "fit_snr": local_snr,
        "fit_n_in_transit_points": n_in,
        "fit_n_events": n_transits,
        "fit_n_good_events": max(0, n_transits - int(rng.integers(0, 2))),
        "fit_event_depth_scatter_ppm": abs(rng.normal(depth_ppm * 0.12, depth_ppm * 0.05)),
        "vet_odd_depth_ppm": depth_ppm + rng.normal(0, max(20, depth_ppm * 0.08)),
        "vet_even_depth_ppm": depth_ppm + rng.normal(0, max(20, depth_ppm * 0.08)) + odd_even_sigma * max(20, depth_ppm * 0.04),
        "vet_odd_even_sigma": odd_even_sigma,
        "vet_odd_even_depth_diff_ppm": odd_even_sigma * max(20, depth_ppm * 0.06),
        "vet_secondary_depth_ppm": secondary_ratio * depth_ppm,
        "vet_secondary_sigma": secondary_sigma,
        "vet_secondary_phase": float(np.clip(rng.normal(0.5, 0.05), 0.1, 0.9)),
        "vet_secondary_to_primary_ratio": secondary_ratio,
        "vet_centroid_shift_pix": centroid_pix,
        "vet_centroid_shift_sigma": centroid_sigma,
        "vet_crowdsap": crowdsap,
        "vet_flfrcsap": float(np.clip(rng.normal(0.92, 0.08), 0.45, 1.0)),
        "vet_crowding_risk": crowding_risk,
        "vet_corrected_depth_ppm": corrected_depth,
        "vet_v_shape_score": v_shape,
        "vet_transit_asymmetry": float(np.clip(abs(rng.normal(0.08, 0.08)), 0, 0.8)),
        "vet_out_of_transit_rms_ppm": float(np.clip(rng.normal(350 + depth_ppm * 0.02, 150), 50, 5000)),
        "vet_red_noise_proxy": red_noise,
        "vet_harmonic_risk": harmonic_risk,
        "vet_data_quality_score": dq,
        "class_confidence": max(planet_score, eb_score, blend_score, stellar_score, systematic_score),
        "class_planet_score": planet_score,
        "class_eb_score": eb_score,
        "class_blend_score": blend_score,
        "class_stellar_variability_score": stellar_score,
        "class_systematic_score": systematic_score,
        "label": label,
    }


def make_synthetic_tess_lc(
    tic_id: int = 261136679,
    sector: int = 1,
    period: float = 3.4,
    depth: float = 1800e-6,
    duration_hrs: float = 2.1,
    noise_ppm: float = 600,
    crowdsap: float = 0.92,
    flfrcsap: float = 0.88,
    inject_blend: bool = False,
    random_state: int = 42,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Realistic synthetic TESS light curve."""
    np.random.seed((random_state + hash(tic_id)) % (2**32))
    cad_days = 2.0 / 1440  # 2-min cadence
    n = int(27.0 / cad_days)  # ~19,440 points per sector
    t = np.arange(n) * cad_days + 1325.0  # BTJD sector start

    # Baseline + stellar variability
    flux = np.ones(n)
    rot = np.random.uniform(8, 30)
    flux += np.random.uniform(0.0005, 0.003) * np.sin(2 * np.pi * t / rot + np.random.uniform(0, 6))
    flux += np.random.uniform(0.0001, 0.001) * np.sin(2 * np.pi * t / (rot / 2))

    # Spacecraft systematics (TESS has stronger momentum-dump artefacts)
    flux += 0.001 * np.sin(2 * np.pi * t / 13.7)  # ~half-sector thermal cycle

    # Transit injection (trapezoidal)
    t0_first = t[0] + period * 1.5
    dur_d = duration_hrs / 24.0
    ing = 0.15 * dur_d
    n_inj = 0
    for i in range(int((t[-1] - t0_first) / period) + 1):
        tc = t0_first + i * period
        dt = t - tc
        h = dur_d / 2
        pr = np.zeros(n)
        pr[np.abs(dt) <= h - ing] = 1.0
        mi = (~(np.abs(dt) <= h - ing)) & (np.abs(dt) <= h)
        pr[mi] = (h - np.abs(dt[mi])) / ing
        pr = np.clip(pr, 0, 1)
        if pr.max() > 0.05:
            flux -= depth * pr
            n_inj += 1

    # Blend injection (if requested)
    if inject_blend:
        blend_period = period * 2.1
        blend_depth = 0.05
        t0_b = t[0] + blend_period * 1.2
        for i in range(int((t[-1] - t0_b) / blend_period) + 1):
            tc = t0_b + i * blend_period
            dt = t - tc
            h = dur_d * 1.4 / 2
            pr = np.zeros(n)
            pr[np.abs(dt) <= h] = 1.0
            flux -= blend_depth * (1.0 - crowdsap) * pr

    # White noise
    nf = noise_ppm * 1e-6
    flux += np.random.normal(0, nf, n)
    flux_err = np.full(n, nf)

    # PDCSAP = SAP - spacecraft systematics
    pdcsap = flux - 0.001 * np.sin(2 * np.pi * t / 13.7)
    pdcsap_err = flux_err * 1.05

    # Quality flags
    quality = np.zeros(n, dtype=int)
    for md_t in np.arange(t[0] + 3.125, t[-1], 3.125):
        idx = np.argmin(np.abs(t - md_t))
        quality[max(0, idx - 2) : idx + 3] |= 32
        flux[max(0, idx - 2) : idx + 3] += 0.003
        pdcsap[max(0, idx - 2) : idx + 3] += 0.001

    for cr_i in np.random.choice(n, 12, replace=False):
        flux[cr_i] += np.random.uniform(0.004, 0.012)
        quality[cr_i] |= 128

    # Centroid
    cent_col = 512 + 0.2 * np.sin(2 * np.pi * t / 3) + np.random.normal(0, 0.05, n)
    cent_row = 489 + 0.1 * np.cos(2 * np.pi * t / 3) + np.random.normal(0, 0.05, n)

    if inject_blend:
        t0_b = t[0] + blend_period * 1.2
        for i in range(int((t[-1] - t0_b) / blend_period) + 1):
            tc = t0_b + i * blend_period
            in_bl = np.abs(t - tc) < dur_d
            cent_col[in_bl] += 0.12 * (1.0 - crowdsap)

    # Scale to physical flux units
    baseline = 185000.0
    sap_flux = flux * baseline
    sap_err = flux_err * baseline
    pdcsap_flux = pdcsap * baseline
    pdcsap_err = pdcsap_err * baseline

    data = {
        "time": t,
        "sap_flux": sap_flux,
        "sap_flux_err": sap_err,
        "pdcsap_flux": pdcsap_flux,
        "pdcsap_flux_err": pdcsap_err,
        "centroid_col": cent_col,
        "centroid_row": cent_row,
        "quality": quality,
    }
    meta = {
        "CROWDSAP": crowdsap,
        "FLFRCSAP": flfrcsap,
        "SECTOR": sector,
        "TICID": tic_id,
        "n_injected": n_inj,
        "true_period": period,
        "true_depth": depth,
        "true_t0": t0_first,
    }
    return data, meta
