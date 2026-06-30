#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from exoplanet_pipeline.classify import classify_candidate_rule_based
from exoplanet_pipeline.cnn_views import build_cnn_candidate_views, save_cnn_example_npz
from exoplanet_pipeline.config import PipelineConfig
from exoplanet_pipeline.fit import refine_candidate_parameters
from exoplanet_pipeline.ingest import load_tess_fits, search_and_download_tess_lc
from exoplanet_pipeline.preprocess import preprocess_raw_lightcurve
from exoplanet_pipeline.public_data import read_public_metadata
from exoplanet_pipeline.schema import CandidateSignal
from exoplanet_pipeline.vetting import extract_vetting_features


def _finite_float(value: Any, default: float = np.nan) -> float:
    try:
        value = float(value)
        return value if np.isfinite(value) else default
    except Exception:
        return default


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and not np.isfinite(value):
        return None
    text = str(value).strip()
    return text if text and text.lower() != "nan" else None


def _candidate_from_public_row(row: pd.Series, raw, candidate_id: int) -> CandidateSignal | None:
    period = _finite_float(row.get("period_days"))
    epoch = _finite_float(row.get("epoch_time"))
    duration = _finite_float(row.get("duration_days"))
    depth_ppm = _finite_float(row.get("depth_ppm"))
    if not (np.isfinite(period) and period > 0 and np.isfinite(epoch) and np.isfinite(duration) and duration > 0):
        return None
    if not np.isfinite(depth_ppm) or depth_ppm <= 0:
        depth_ppm = 1000.0
    return CandidateSignal(
        tic_id=raw.tic_id,
        sector=raw.sector,
        candidate_id=candidate_id,
        period_days=period,
        epoch_time=epoch,
        duration_days=duration,
        depth_fraction=depth_ppm * 1e-6,
        depth_ppm=depth_ppm,
        snr=np.nan,
        local_snr=np.nan,
        sde=np.nan,
        fap=None,
        n_transits=0,
        n_full_transits=0,
        n_in_transit_points=0,
        detection_method="public_ephemeris_seed",
        flux_source="PDCSAP",
        detrend_variant="default",
        status="STRONG_DETECTION",
        extra={"public_source_row": int(row.name) if row.name is not None else None},
    )


def _find_cached_lightcurves(tic_id: int, lightcurve_dir: Path) -> list[Path]:
    patterns = [
        f"*{tic_id}*lc.fits*",
        f"*{tic_id:016d}*lc.fits*",
        f"*tic{tic_id}*lc.fits*",
        f"*TIC{tic_id}*lc.fits*",
    ]
    paths: list[Path] = []
    for pat in patterns:
        paths.extend(lightcurve_dir.rglob(pat))
    return sorted(set(paths))


def _sector_from_row(row: pd.Series) -> int | None:
    raw = row.get("sector", row.get("sectors", None))
    if raw is None or (isinstance(raw, float) and not np.isfinite(raw)):
        return None
    text = str(raw)
    digits = ""
    for ch in text:
        if ch.isdigit():
            digits += ch
        elif digits:
            break
    return int(digits) if digits else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Build CNN examples from official public TOI metadata and MAST TESS light curves.")
    parser.add_argument("metadata_csv", help="Normalized metadata CSV, or raw CSV with --source")
    parser.add_argument("--source", default=None, choices=["tess-toi"], help="Normalize raw TOI metadata before building examples")
    parser.add_argument("--output-dir", default="data/public/cnn_examples")
    parser.add_argument("--lightcurve-dir", default="data/public/lightcurves")
    parser.add_argument("--download-missing", action="store_true", help="Download missing TESS light curves from MAST with astroquery")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--max-lightcurves-per-target", type=int, default=1)
    parser.add_argument("--n-bootstrap", type=int, default=50)
    parser.add_argument("--detrend-method", default="rolling_median", choices=["rolling_median", "wotan_biweight", "none"])
    args = parser.parse_args()

    metadata = read_public_metadata(args.metadata_csv, args.source) if args.source else pd.read_csv(args.metadata_csv)
    if args.max_rows is not None:
        metadata = metadata.head(args.max_rows).copy()

    output_dir = Path(args.output_dir)
    lightcurve_dir = Path(args.lightcurve_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    lightcurve_dir.mkdir(parents=True, exist_ok=True)

    config = PipelineConfig(detrend_method=args.detrend_method, make_plots=False)
    manifest_rows: list[dict[str, Any]] = []
    n_written = 0

    for _, row in metadata.iterrows():
        tic = _finite_float(row.get("tic_id"))
        canonical_label = _optional_text(row.get("canonical_label"))
        binary_label = _optional_text(row.get("binary_label"))
        if not np.isfinite(tic) or (canonical_label is None and binary_label is None):
            continue
        tic_id = int(tic)
        paths = _find_cached_lightcurves(tic_id, lightcurve_dir)
        if not paths and args.download_missing:
            try:
                paths = search_and_download_tess_lc(tic_id, sector=_sector_from_row(row), download_dir=lightcurve_dir)
            except Exception as exc:
                manifest_rows.append({"tic_id": tic_id, "status": "download_failed", "error": repr(exc)})
                continue
        if not paths:
            manifest_rows.append({"tic_id": tic_id, "status": "missing_lightcurve"})
            continue

        for path in paths[: args.max_lightcurves_per_target]:
            raw = load_tess_fits(path)
            if raw.status != "RAW_LOADED":
                manifest_rows.append({"tic_id": tic_id, "fits_path": str(path), "status": raw.status, "error": raw.error})
                continue
            cand = _candidate_from_public_row(row, raw, candidate_id=n_written + 1)
            if cand is None:
                manifest_rows.append({"tic_id": tic_id, "fits_path": str(path), "status": "missing_ephemeris"})
                continue
            try:
                clean = preprocess_raw_lightcurve(raw, config)
                fit = refine_candidate_parameters(clean, cand, n_bootstrap=args.n_bootstrap)
                vet = extract_vetting_features(clean, cand, fit)
                cls = classify_candidate_rule_based(cand, fit, vet)
                views = build_cnn_candidate_views(clean, cand, fit, vet)
                views.metadata.update({
                    "tic_id": tic_id,
                    "sector": raw.sector,
                    "fits_path": str(path),
                    "source": row.get("source", ""),
                    "canonical_label": canonical_label,
                    "binary_label": binary_label,
                    "class_predicted_class": cls.predicted_class,
                    "fit_snr": fit.snr,
                    "vet_secondary_sigma": vet.secondary_sigma,
                    "vet_odd_even_sigma": vet.odd_even_sigma,
                    "vet_centroid_shift_sigma": vet.centroid_shift_sigma,
                    "vet_crowding_risk": vet.crowding_risk,
                    "vet_data_quality_score": vet.data_quality_score,
                })
                example_path = output_dir / f"tic_{tic_id}_cand_{n_written + 1:06d}.npz"
                save_cnn_example_npz(views, example_path, canonical_label=canonical_label, binary_label=binary_label)
                n_written += 1
                manifest_rows.append({"tic_id": tic_id, "fits_path": str(path), "example_path": str(example_path), "status": "ok"})
            except Exception as exc:
                manifest_rows.append({"tic_id": tic_id, "fits_path": str(path), "status": "example_failed", "error": repr(exc)})

    manifest = pd.DataFrame(manifest_rows)
    manifest_path = output_dir / "cnn_example_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    summary_path = output_dir / "cnn_example_summary.json"
    summary = {
        "n_metadata_rows": int(len(metadata)),
        "n_examples_written": int(n_written),
        "status_counts": manifest["status"].value_counts(dropna=False).to_dict() if "status" in manifest else {},
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"manifest: {manifest_path}")
    print(f"examples_dir: {output_dir}")


if __name__ == "__main__":
    main()
