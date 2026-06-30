from __future__ import annotations

import numpy as np
import pandas as pd

from .schema import CandidateSignal, TransitFitResult, VettingFeatures, ClassificationResult


def _finite(x, default=np.nan) -> float:
    try:
        x = float(x)
        return x if np.isfinite(x) else default
    except Exception:
        return default


def _sigmoid_score(x: float, center: float, scale: float) -> float:
    if not np.isfinite(x):
        return 0.0
    z = (x - center) / max(scale, 1e-9)
    return float(1.0 / (1.0 + np.exp(-z)))


def classify_candidate_rule_based(
    candidate: CandidateSignal,
    fit: TransitFitResult,
    vet: VettingFeatures,
) -> ClassificationResult:
    """Transparent baseline classifier for Parts 5/early Part 6.

    This is intentionally interpretable. It should later be replaced or blended
    with a supervised model trained on the curated dataset, but it gives a strong
    scientific baseline and provides evidence strings for reports.
    """
    evidence: list[str] = []
    warnings: list[str] = []

    snr = _finite(fit.snr, _finite(candidate.local_snr, np.nan))
    sde = _finite(candidate.sde, np.nan)
    depth_ppm = _finite(fit.depth_ppm, np.nan)
    rp_rs = _finite(fit.rp_over_rstar, np.nan)
    n_events = fit.n_good_events
    secondary_sigma = _finite(vet.secondary_sigma, 0.0)
    secondary_ratio = _finite(vet.secondary_to_primary_ratio, 0.0)
    odd_even_sigma = _finite(vet.odd_even_sigma, 0.0)
    centroid_sigma = _finite(vet.centroid_shift_sigma, 0.0)
    crowding_risk = _finite(vet.crowding_risk if vet.crowding_risk is not None else np.nan, np.nan)
    v_shape = _finite(vet.v_shape_score, np.nan)
    red_noise = _finite(vet.red_noise_proxy, np.nan)
    dq = _finite(vet.data_quality_score, 0.5)

    # Detection confidence backbone.
    detection_strength = 0.0
    detection_strength += 0.45 * _sigmoid_score(snr, center=8.0, scale=2.0)
    if np.isfinite(sde):
        detection_strength += 0.25 * _sigmoid_score(sde, center=7.0, scale=1.5)
    else:
        detection_strength += 0.12
    detection_strength += 0.20 * min(max(n_events / 3.0, 0.0), 1.0)
    detection_strength += 0.10 * dq

    if snr >= 10:
        evidence.append("strong_transit_snr")
    elif snr >= 7:
        evidence.append("moderate_transit_snr")
    else:
        warnings.append("low_snr")

    # EB evidence.
    eb_score = 0.0
    if secondary_sigma >= 5 and secondary_ratio >= 0.05:
        eb_score += 0.45
        evidence.append("significant_secondary_eclipse")
    elif secondary_sigma >= 5:
        eb_score += 0.25
        evidence.append("possible_secondary_eclipse")
    if odd_even_sigma >= 3:
        eb_score += 0.35
        evidence.append("odd_even_depth_mismatch")
    if np.isfinite(depth_ppm) and depth_ppm > 15000:
        eb_score += 0.20
        evidence.append("very_deep_event")
    if np.isfinite(rp_rs) and rp_rs > 0.18:
        eb_score += 0.20
        evidence.append("large_radius_ratio")
    if np.isfinite(v_shape) and v_shape < 0.45:
        eb_score += 0.12
        evidence.append("triangular_or_poorly_flat_bottom_event")
    eb_score = float(np.clip(eb_score, 0.0, 1.0))

    # Blend evidence.
    blend_score = 0.0
    if centroid_sigma >= 5:
        blend_score += 0.55
        evidence.append("significant_centroid_shift")
    elif centroid_sigma >= 3:
        blend_score += 0.25
        evidence.append("marginal_centroid_shift")
    if np.isfinite(crowding_risk) and crowding_risk > 0.30:
        blend_score += 0.30
        evidence.append("high_crowding_risk")
    elif np.isfinite(crowding_risk) and crowding_risk > 0.15:
        blend_score += 0.15
        evidence.append("moderate_crowding_risk")
    if np.isfinite(vet.corrected_depth_ppm) and np.isfinite(depth_ppm) and vet.corrected_depth_ppm > 2.0 * depth_ppm and vet.corrected_depth_ppm > 5000:
        blend_score += 0.15
        evidence.append("large_dilution_correction")
    blend_score = float(np.clip(blend_score, 0.0, 1.0))

    # Stellar variability/systematic evidence.
    stellar_var_score = 0.0
    if np.isfinite(red_noise) and red_noise > 0.25:
        stellar_var_score += 0.25
        evidence.append("red_noise_or_quasiperiodic_variability")
    if np.isfinite(v_shape) and v_shape < 0.20 and np.isfinite(fit.duration_days) and fit.period_days > 0 and fit.duration_days / fit.period_days > 0.12:
        stellar_var_score += 0.25
        evidence.append("broad_dip_relative_to_period")
    if candidate.status == "NO_DETECTION":
        stellar_var_score += 0.10
    stellar_var_score = float(np.clip(stellar_var_score, 0.0, 1.0))

    systematic_score = 0.0
    if dq < 0.45:
        systematic_score += 0.35
        evidence.append("low_data_quality")
    if candidate.n_full_transits < 2:
        systematic_score += 0.25
        evidence.append("few_full_transits")
    if candidate.depth_fraction <= 0:
        systematic_score += 0.35
        evidence.append("non_positive_depth")
    systematic_score = float(np.clip(systematic_score, 0.0, 1.0))

    # Planet score is high only when detection is strong and veto evidence is low.
    veto = max(eb_score, blend_score, stellar_var_score, systematic_score)
    planet_score = float(np.clip(detection_strength * (1.0 - 0.75 * veto), 0.0, 1.0))
    if planet_score > 0.65:
        evidence.append("clean_transit_like_signal")

    scores = {
        "PLANETARY_TRANSIT_CANDIDATE": planet_score,
        "ECLIPSING_BINARY": eb_score,
        "BLEND_OR_CONTAMINATED_SIGNAL": blend_score,
        "STELLAR_VARIABILITY": stellar_var_score,
        "INSTRUMENTAL_OR_LOW_QUALITY_SYSTEMATIC": systematic_score,
    }

    predicted = max(scores, key=scores.get)
    confidence = float(scores[predicted])

    # If everything is low, choose uncertain rather than overclaiming.
    if confidence < 0.45:
        predicted = "UNCERTAIN_TRANSIT_LIKE_SIGNAL" if detection_strength >= 0.35 else "NO_SIGNIFICANT_SIGNAL"
        confidence = float(max(detection_strength, 1.0 - detection_strength) if predicted == "NO_SIGNIFICANT_SIGNAL" else 0.45)
        warnings.append("low_classification_margin")

    # If planet and false-positive scores are close, downgrade to uncertain.
    sorted_scores = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    if sorted_scores[0][0] == "PLANETARY_TRANSIT_CANDIDATE" and len(sorted_scores) > 1:
        if sorted_scores[0][1] - sorted_scores[1][1] < 0.12:
            predicted = "UNCERTAIN_TRANSIT_LIKE_SIGNAL"
            confidence = float(sorted_scores[0][1])
            warnings.append("planet_score_close_to_false_positive_score")

    return ClassificationResult(
        tic_id=candidate.tic_id,
        sector=candidate.sector,
        candidate_id=candidate.candidate_id,
        predicted_class=predicted,
        confidence=confidence,
        planet_score=planet_score,
        eb_score=eb_score,
        blend_score=blend_score,
        stellar_variability_score=stellar_var_score,
        systematic_score=systematic_score,
        evidence=evidence,
        warnings=warnings + vet.warnings + fit.warnings,
        extra={
            "detection_strength": detection_strength,
            "snr": snr,
            "sde": sde,
            "depth_ppm": depth_ppm,
            "n_good_events": n_events,
        },
    )


def classification_to_dataframe(result: ClassificationResult) -> pd.DataFrame:
    return pd.DataFrame([result.to_dict()])
