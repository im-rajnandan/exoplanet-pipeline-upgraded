from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


CANONICAL_CLASSES: tuple[str, ...] = (
    "PLANETARY_TRANSIT_CANDIDATE",
    "ECLIPSING_BINARY",
    "BLEND_OR_CONTAMINATED_SIGNAL",
    "STELLAR_VARIABILITY",
    "INSTRUMENTAL_OR_LOW_QUALITY_SYSTEMATIC",
    "NO_SIGNIFICANT_SIGNAL",
    "UNCERTAIN_TRANSIT_LIKE_SIGNAL",
)

NO_SIGNAL_CLASS = "NO_SIGNIFICANT_SIGNAL"
UNCERTAIN_CLASS = "UNCERTAIN_TRANSIT_LIKE_SIGNAL"


def renormalize_probs(probs: dict[str, float]) -> dict[str, float]:
    clean = {cls: max(0.0, float(probs.get(cls, 0.0))) for cls in CANONICAL_CLASSES}
    total = sum(clean.values())
    if total <= 0:
        return {cls: 1.0 / len(CANONICAL_CLASSES) for cls in CANONICAL_CLASSES}
    return {cls: value / total for cls, value in clean.items()}


def apply_physical_guardrails(probs: dict[str, float], row: pd.Series | dict[str, Any]) -> tuple[dict[str, float], list[str]]:
    warnings_here: list[str] = []
    p = dict(probs)
    row = pd.Series(row)

    secondary_sigma = _safe_float(row.get("vet_secondary_sigma"))
    secondary_ratio = _safe_float(row.get("vet_secondary_to_primary_ratio"))
    odd_even_sigma = _safe_float(row.get("vet_odd_even_sigma"))
    centroid_sigma = _safe_float(row.get("vet_centroid_shift_sigma"))
    crowdsap = _safe_float(row.get("vet_crowdsap"))
    crowding_risk = _safe_float(row.get("vet_crowding_risk"))
    data_quality = _safe_float(row.get("vet_data_quality_score"))
    snr = _safe_float(row.get("fit_snr", row.get("snr")))

    if secondary_sigma >= 5.0 and secondary_ratio >= 0.05:
        p["ECLIPSING_BINARY"] = max(p.get("ECLIPSING_BINARY", 0.0), 0.82)
        p["PLANETARY_TRANSIT_CANDIDATE"] = min(p.get("PLANETARY_TRANSIT_CANDIDATE", 0.0), 0.18)
        warnings_here.append("guardrail_strong_secondary_eclipse")
    elif secondary_sigma >= 5.0:
        p["ECLIPSING_BINARY"] = max(p.get("ECLIPSING_BINARY", 0.0), 0.60)
        p["PLANETARY_TRANSIT_CANDIDATE"] = min(p.get("PLANETARY_TRANSIT_CANDIDATE", 0.0), 0.35)
        warnings_here.append("guardrail_possible_secondary_eclipse")
    if odd_even_sigma >= 3.0:
        p["ECLIPSING_BINARY"] = max(p.get("ECLIPSING_BINARY", 0.0), 0.72)
        p["PLANETARY_TRANSIT_CANDIDATE"] = min(p.get("PLANETARY_TRANSIT_CANDIDATE", 0.0), 0.25)
        warnings_here.append("guardrail_odd_even_depth_mismatch")
    if centroid_sigma >= 5.0:
        p["BLEND_OR_CONTAMINATED_SIGNAL"] = max(p.get("BLEND_OR_CONTAMINATED_SIGNAL", 0.0), 0.82)
        p["PLANETARY_TRANSIT_CANDIDATE"] = min(p.get("PLANETARY_TRANSIT_CANDIDATE", 0.0), 0.25)
        warnings_here.append("guardrail_significant_centroid_shift")
    elif centroid_sigma >= 3.0 and crowding_risk >= 0.25:
        p["BLEND_OR_CONTAMINATED_SIGNAL"] = max(p.get("BLEND_OR_CONTAMINATED_SIGNAL", 0.0), 0.65)
        warnings_here.append("guardrail_marginal_centroid_plus_crowding")
    if (np.isfinite(crowdsap) and crowdsap <= 0.60) or crowding_risk >= 0.40:
        p["BLEND_OR_CONTAMINATED_SIGNAL"] = max(p.get("BLEND_OR_CONTAMINATED_SIGNAL", 0.0), 0.62)
        p["PLANETARY_TRANSIT_CANDIDATE"] = min(p.get("PLANETARY_TRANSIT_CANDIDATE", 0.0), 0.38)
        warnings_here.append("guardrail_low_crowdsap_or_high_crowding")
    if data_quality <= 0.35:
        p["INSTRUMENTAL_OR_LOW_QUALITY_SYSTEMATIC"] = max(p.get("INSTRUMENTAL_OR_LOW_QUALITY_SYSTEMATIC", 0.0), 0.70)
        p["PLANETARY_TRANSIT_CANDIDATE"] = min(p.get("PLANETARY_TRANSIT_CANDIDATE", 0.0), 0.22)
        warnings_here.append("guardrail_low_data_quality")
    if snr < 6.0:
        p["NO_SIGNIFICANT_SIGNAL"] = max(p.get("NO_SIGNIFICANT_SIGNAL", 0.0), 0.55)
        p["PLANETARY_TRANSIT_CANDIDATE"] = min(p.get("PLANETARY_TRANSIT_CANDIDATE", 0.0), 0.18)
        warnings_here.append("guardrail_low_snr")
    return p, warnings_here


def finalize_probabilities(
    probs: dict[str, float],
    *,
    row: pd.Series | dict[str, Any] | None = None,
    apply_guardrails: bool = True,
    low_margin_warning: str = "low_classifier_margin_downgraded_to_uncertain",
    min_confidence: float = 0.45,
) -> tuple[dict[str, float], str, float, list[str]]:
    warnings_here: list[str] = []
    final_probs = dict(probs)
    if apply_guardrails and row is not None:
        final_probs, guard_warnings = apply_physical_guardrails(final_probs, row)
        warnings_here.extend(guard_warnings)

    final_probs = renormalize_probs(final_probs)
    predicted_class = max(final_probs, key=final_probs.get)
    confidence = float(final_probs[predicted_class])
    if confidence < min_confidence and predicted_class != NO_SIGNAL_CLASS:
        predicted_class = UNCERTAIN_CLASS
        confidence = max(confidence, final_probs.get(predicted_class, 0.0), min_confidence)
        warnings_here.append(low_margin_warning)
    return final_probs, predicted_class, float(confidence), warnings_here


def _safe_float(x: Any, default: float = np.nan) -> float:
    try:
        value = float(x)
        return value if np.isfinite(value) else default
    except Exception:
        return default
