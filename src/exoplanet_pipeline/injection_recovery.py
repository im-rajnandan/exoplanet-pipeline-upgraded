from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable
import numpy as np
import pandas as pd

from .config import PipelineConfig
from .synthetic import make_synthetic_transit_lc, make_synthetic_eb_lc, make_synthetic_blend_lc
from .pipeline import run_parts_1_to_5_from_raw
from .uncertainty import estimate_candidate_uncertainty


@dataclass
class InjectionSpec:
    sample_id: str
    label: str
    period_days: float
    depth_ppm: float
    duration_hours: float
    noise_ppm: float
    crowdsap: float
    centroid_shift_pix: float = 0.0
    secondary_depth_ppm: float = 0.0
    random_seed: int = 42

    def to_dict(self) -> dict:
        return asdict(self)


def default_injection_grid(random_seed: int = 42) -> list[InjectionSpec]:
    """Small but diverse validation grid for Parts 7–8 demos.

    The organizer's real curated dataset should replace/extend this, but this
    grid proves the entire pipeline and validation machinery before labels arrive.
    """
    rng = np.random.default_rng(random_seed)
    specs: list[InjectionSpec] = []
    idx = 0
    # Planet-like transits over detectability regimes.
    for period in [1.2, 2.7, 5.5, 9.0]:
        for depth in [400.0, 900.0, 1800.0, 3500.0]:
            noise = float(rng.choice([250.0, 450.0, 800.0]))
            specs.append(InjectionSpec(
                sample_id=f"planet_{idx:03d}", label="PLANETARY_TRANSIT_CANDIDATE",
                period_days=float(period), depth_ppm=float(depth), duration_hours=float(rng.choice([1.2, 2.0, 3.0])),
                noise_ppm=noise, crowdsap=float(rng.choice([0.85, 0.92, 0.98])), random_seed=random_seed + idx,
            ))
            idx += 1
    # EBs with secondaries.
    for period in [1.8, 3.5, 6.0, 8.5]:
        for depth in [12000.0, 25000.0, 50000.0]:
            specs.append(InjectionSpec(
                sample_id=f"eb_{idx:03d}", label="ECLIPSING_BINARY",
                period_days=float(period), depth_ppm=float(depth), duration_hours=float(rng.choice([2.5, 3.5, 5.0])),
                noise_ppm=float(rng.choice([300.0, 600.0, 1000.0])), crowdsap=float(rng.choice([0.80, 0.90, 0.97])),
                secondary_depth_ppm=float(depth * rng.choice([0.2, 0.35, 0.6])), random_seed=random_seed + idx,
            ))
            idx += 1
    # Blends/crowded contaminated signals.
    for period in [2.2, 4.0, 7.2]:
        for depth in [800.0, 1600.0, 3000.0]:
            specs.append(InjectionSpec(
                sample_id=f"blend_{idx:03d}", label="BLEND_OR_CONTAMINATED_SIGNAL",
                period_days=float(period), depth_ppm=float(depth), duration_hours=float(rng.choice([1.5, 2.5, 4.0])),
                noise_ppm=float(rng.choice([300.0, 600.0, 900.0])), crowdsap=float(rng.choice([0.45, 0.55, 0.65])),
                centroid_shift_pix=float(rng.choice([0.015, 0.03, 0.06])), random_seed=random_seed + idx,
            ))
            idx += 1
    return specs


def compact_injection_demo_grid(random_seed: int = 42, n_per_class: int = 2) -> list[InjectionSpec]:
    """Return a fast, class-balanced subset of the default injection grid."""
    specs = default_injection_grid(random_seed=random_seed)
    labels = [
        "PLANETARY_TRANSIT_CANDIDATE",
        "ECLIPSING_BINARY",
        "BLEND_OR_CONTAMINATED_SIGNAL",
    ]
    selected: list[InjectionSpec] = []
    for label in labels:
        selected.extend([spec for spec in specs if spec.label == label][:n_per_class])
    return selected


def raw_from_injection(spec: InjectionSpec):
    if spec.label == "ECLIPSING_BINARY":
        return make_synthetic_eb_lc(
            tic_id=800000 + spec.random_seed,
            period_days=spec.period_days,
            primary_depth_ppm=spec.depth_ppm,
            secondary_depth_ppm=spec.secondary_depth_ppm or 0.3 * spec.depth_ppm,
            duration_hours=spec.duration_hours,
            noise_ppm=spec.noise_ppm,
            crowdsap=spec.crowdsap,
            random_seed=spec.random_seed,
        )
    if spec.label == "BLEND_OR_CONTAMINATED_SIGNAL":
        return make_synthetic_blend_lc(
            tic_id=800000 + spec.random_seed,
            period_days=spec.period_days,
            observed_depth_ppm=spec.depth_ppm,
            duration_hours=spec.duration_hours,
            noise_ppm=spec.noise_ppm,
            crowdsap=spec.crowdsap,
            centroid_shift_pix=spec.centroid_shift_pix,
            random_seed=spec.random_seed,
        )
    return make_synthetic_transit_lc(
        tic_id=800000 + spec.random_seed,
        period_days=spec.period_days,
        depth_ppm=spec.depth_ppm,
        duration_hours=spec.duration_hours,
        noise_ppm=spec.noise_ppm,
        crowdsap=spec.crowdsap,
        random_seed=spec.random_seed,
    )


def run_single_injection(spec: InjectionSpec, config: PipelineConfig | None = None, n_bootstrap: int = 120) -> dict:
    config = config or PipelineConfig()
    raw = raw_from_injection(spec)
    result = run_parts_1_to_5_from_raw(raw, config=config)
    catalog = result["catalog"].copy()
    base = spec.to_dict()
    base.update({
        "true_period_days": spec.period_days,
        "true_depth_ppm": spec.depth_ppm,
        "true_duration_hours": spec.duration_hours,
        "injected_label": spec.label,
    })
    if catalog.empty:
        base.update({
            "detected": False,
            "pipeline_status": result["detection"].status,
            "final_predicted_class": "NO_SIGNIFICANT_SIGNAL",
            "final_confidence": 0.0,
        })
        return base
    # choose best row by fit_snr/local_snr
    score_col = "fit_snr" if "fit_snr" in catalog.columns else "local_snr"
    idx = pd.to_numeric(catalog[score_col], errors="coerce").fillna(-np.inf).idxmax()
    row = catalog.loc[idx].to_dict()
    selected_candidate_id = int(row.get("candidate_id", 1))
    candidates = result.get("fitted_candidates") or result["detection"].candidates
    candidate_by_id = {int(c.candidate_id): c for c in candidates}
    fit_by_id = {int(f.candidate_id): f for f in result["fit_results"]}
    vet_by_id = {int(v.candidate_id): v for v in result["vetting_results"]}
    cls_by_id = {int(c.candidate_id): c for c in result["classification_results"]}
    # Estimate uncertainty for the same candidate row selected for validation.
    cand = candidate_by_id.get(selected_candidate_id, candidates[0])
    fit = fit_by_id.get(selected_candidate_id, result["fit_results"][0])
    vet = vet_by_id.get(selected_candidate_id, result["vetting_results"][0])
    cls = cls_by_id.get(selected_candidate_id, result["classification_results"][0])
    unc = estimate_candidate_uncertainty(result["clean"], cand, fit, vet, cls, n_bootstrap=n_bootstrap, random_seed=spec.random_seed)
    # Determine recovery tolerance. Use period alias tolerance too because EBs often appear at P/2.
    p_pred = float(row.get("fit_period_days", row.get("period_days", np.nan)))
    period_rel_err = abs(p_pred - spec.period_days) / spec.period_days if np.isfinite(p_pred) else np.nan
    alias_rel_err = abs(2 * p_pred - spec.period_days) / spec.period_days if np.isfinite(p_pred) else np.nan
    recovered_period = bool((np.isfinite(period_rel_err) and period_rel_err <= 0.05) or (np.isfinite(alias_rel_err) and alias_rel_err <= 0.05))
    base.update(row)
    base.update({f"unc_{k}": v for k, v in unc.to_dict().items() if k not in ("tic_id", "sector", "candidate_id")})
    base.update({
        "detected": str(row.get("status", "")).upper() in ("STRONG_DETECTION", "WEAK_DETECTION") or float(row.get("fit_snr", row.get("snr", 0)) or 0) >= 7,
        "recovered_period": recovered_period,
        "period_rel_error": period_rel_err,
        "period_alias_rel_error": alias_rel_err,
        "depth_rel_error": abs(float(row.get("fit_depth_ppm", np.nan)) - spec.depth_ppm) / spec.depth_ppm if np.isfinite(float(row.get("fit_depth_ppm", np.nan))) and spec.depth_ppm else np.nan,
        "duration_rel_error": abs(float(row.get("fit_duration_days", np.nan)) * 24.0 - spec.duration_hours) / spec.duration_hours if np.isfinite(float(row.get("fit_duration_days", np.nan))) and spec.duration_hours else np.nan,
        "pipeline_status": result["detection"].status,
        "final_predicted_class": row.get("class_predicted_class", row.get("final_predicted_class", "")),
        "final_confidence": row.get("class_confidence", row.get("final_confidence", np.nan)),
    })
    return base


def run_injection_recovery_grid(
    specs: Iterable[InjectionSpec] | None = None,
    config: PipelineConfig | None = None,
    n_bootstrap: int = 80,
) -> pd.DataFrame:
    specs = list(specs) if specs is not None else default_injection_grid()
    config = config or PipelineConfig()
    rows = []
    for spec in specs:
        rows.append(run_single_injection(spec, config=config, n_bootstrap=n_bootstrap))
    return pd.DataFrame(rows)


def summarize_injection_recovery(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"n": 0}
    out = {"n": int(len(df))}
    if "detected" in df:
        out["detection_rate"] = float(pd.Series(df["detected"]).astype(bool).mean())
    if "recovered_period" in df:
        out["period_recovery_rate"] = float(pd.Series(df["recovered_period"]).astype(bool).mean())
    if "label" in df and "final_predicted_class" in df:
        out["class_accuracy"] = float((df["label"].astype(str) == df["final_predicted_class"].astype(str)).mean())
        out["per_class_detection_rate"] = df.groupby("label")["detected"].mean().to_dict() if "detected" in df else {}
    if "period_rel_error" in df or "period_alias_rel_error" in df:
        direct = pd.to_numeric(df.get("period_rel_error", np.nan), errors="coerce")
        alias = pd.to_numeric(df.get("period_alias_rel_error", np.nan), errors="coerce")
        best = pd.concat([direct, alias], axis=1).min(axis=1, skipna=True)
        out["median_period_or_alias_rel_error"] = float(best.median()) if best.notna().any() else None
        out["median_period_rel_error_direct"] = float(direct.median()) if direct.notna().any() else None
    for col in ["depth_rel_error", "duration_rel_error"]:
        if col in df:
            vals = pd.to_numeric(df[col], errors="coerce")
            out[f"median_{col}"] = float(vals.median()) if vals.notna().any() else None
    return out
