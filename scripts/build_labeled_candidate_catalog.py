#!/usr/bin/env python
from __future__ import annotations

import argparse
import concurrent.futures
from collections import Counter
import json
from pathlib import Path
import re
import sys
import threading
import time
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from exoplanet_pipeline.classify import classify_candidate_rule_based
from exoplanet_pipeline.config import PipelineConfig
from exoplanet_pipeline.fit import refine_candidate_parameters
from exoplanet_pipeline.ingest import load_tess_fits, search_and_download_tess_lc
from exoplanet_pipeline.pipeline import run_parts_1_to_5_from_raw
from exoplanet_pipeline.preprocess import preprocess_raw_lightcurve
from exoplanet_pipeline.public_data import read_public_metadata
from exoplanet_pipeline.schema import CandidateSignal
from exoplanet_pipeline.vetting import extract_vetting_features


TIC_COLUMNS = ("tic_id", "tid", "toi_tic", "tic", "TICID", "ID")
SECTOR_COLUMNS = ("sector", "sectors", "Sector")
PERIOD_COLUMNS = ("period_days", "period", "pl_orbper", "koi_period", "tce_period", "toi_period")
EPOCH_COLUMNS = ("epoch_time", "epoch", "t0", "pl_tranmid", "koi_time0bk", "tce_time0bk", "toi_epoch")
DURATION_DAYS_COLUMNS = ("duration_days", "koi_duration_days", "tce_duration_days")
DURATION_HOURS_COLUMNS = ("duration_hours", "duration_hrs", "pl_trandurh", "koi_duration", "tce_duration", "toi_duration")
DEPTH_PPM_COLUMNS = ("depth_ppm", "depth", "pl_trandep", "koi_depth", "tce_depth", "toi_depth")
_DOWNLOAD_LOCKS: dict[int, threading.Lock] = {}
_DOWNLOAD_LOCKS_GUARD = threading.Lock()


def _download_lock_for_tic(tic_id: int) -> threading.Lock:
    with _DOWNLOAD_LOCKS_GUARD:
        lock = _DOWNLOAD_LOCKS.get(tic_id)
        if lock is None:
            lock = threading.Lock()
            _DOWNLOAD_LOCKS[tic_id] = lock
        return lock


def _row_get(row: pd.Series, columns: tuple[str, ...], default: Any = np.nan) -> Any:
    lower = {str(c).lower(): c for c in row.index}
    for col in columns:
        actual = lower.get(col.lower())
        if actual is not None:
            value = row.get(actual)
            if value is not None and not (isinstance(value, float) and np.isnan(value)):
                return value
    return default


def _finite_float(value: Any, default: float = np.nan) -> float:
    try:
        value = float(value)
        return value if np.isfinite(value) else default
    except Exception:
        return default


def _optional_text(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return None
    text = str(value).strip()
    return text if text and text.lower() != "nan" else None


def _sector_from_row(row: pd.Series) -> int | None:
    raw = _row_get(row, SECTOR_COLUMNS, None)
    text = _optional_text(raw)
    if text is None:
        return None
    digits = ""
    for ch in text:
        if ch.isdigit():
            digits += ch
        elif digits:
            break
    return int(digits) if digits else None


def _find_cached_lightcurves(tic_id: int, lightcurve_dir: Path) -> list[Path]:
    patterns = [
        f"*{tic_id:016d}*lc.fits*",
        f"*{tic_id}*lc.fits*",
        f"*tic{tic_id}*lc.fits*",
        f"*TIC{tic_id}*lc.fits*",
    ]
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(lightcurve_dir.rglob(pattern))
    return sorted({path for path in paths if _path_matches_tic_id(path, tic_id)})


def _path_matches_tic_id(path: Path, tic_id: int) -> bool:
    name = path.name.lower()
    if f"{tic_id:016d}" in name:
        return True
    return re.search(rf"(?<!\d)(?:tic[_-]?)?{int(tic_id)}(?!\d)", name) is not None


def _candidate_from_ephemeris(row: pd.Series, raw, candidate_id: int, epoch_offset: float | None) -> CandidateSignal | None:
    period = _finite_float(_row_get(row, PERIOD_COLUMNS))
    epoch = _finite_float(_row_get(row, EPOCH_COLUMNS))
    if epoch_offset is not None and np.isfinite(epoch) and epoch > 2_400_000.0:
        epoch -= float(epoch_offset)
    duration = _finite_float(_row_get(row, DURATION_DAYS_COLUMNS))
    if not np.isfinite(duration):
        duration_hours = _finite_float(_row_get(row, DURATION_HOURS_COLUMNS))
        duration = duration_hours / 24.0 if np.isfinite(duration_hours) else np.nan
    depth_ppm = _finite_float(_row_get(row, DEPTH_PPM_COLUMNS))

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
        detection_method="curated_ephemeris_seed",
        flux_source="PDCSAP",
        detrend_variant="default",
        status="STRONG_DETECTION",
    )


def _row_from_seeded_candidate(raw, candidate: CandidateSignal, label: str, config: PipelineConfig, n_bootstrap: int) -> dict[str, Any]:
    clean = preprocess_raw_lightcurve(raw, config)
    if clean.status != "OK":
        raise RuntimeError(f"preprocess_status={clean.status}")
    fit = refine_candidate_parameters(clean, candidate, n_bootstrap=n_bootstrap)
    vet = extract_vetting_features(clean, candidate, fit)
    cls = classify_candidate_rule_based(candidate, fit, vet)

    out: dict[str, Any] = {}
    out.update(candidate.to_dict())
    out.update({f"fit_{k}": v for k, v in fit.to_dict().items() if k not in ("tic_id", "sector", "candidate_id")})
    out.update({f"vet_{k}": v for k, v in vet.to_dict().items() if k not in ("tic_id", "sector", "candidate_id")})
    out.update({f"class_{k}": v for k, v in cls.to_dict().items() if k not in ("tic_id", "sector", "candidate_id")})
    out["label"] = label
    return out


def _select_catalog_rows(catalog: pd.DataFrame, label: str, all_candidates: bool) -> pd.DataFrame:
    if catalog.empty:
        return catalog
    rows = catalog.copy()
    if not all_candidates:
        score = pd.Series(-np.inf, index=rows.index)
        for col in ("fit_snr", "local_snr", "snr"):
            if col in rows:
                score = score.combine(pd.to_numeric(rows[col], errors="coerce").fillna(-np.inf), max)
        rows = rows.loc[[score.idxmax()]]
    rows["label"] = label
    return rows


def _process_row(row_idx: int, row: pd.Series, args, config: PipelineConfig, lightcurve_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    label = _optional_text(row.get(args.label_col))
    tic = _finite_float(_row_get(row, TIC_COLUMNS))
    if label is None or not np.isfinite(tic):
        return [], [{"source_row_index": row_idx, "status": "missing_tic_or_label"}]

    tic_id = int(tic)
    paths = _find_cached_lightcurves(tic_id, lightcurve_dir)
    if not paths and args.download_missing:
        with _download_lock_for_tic(tic_id):
            paths = _find_cached_lightcurves(tic_id, lightcurve_dir)
            if not paths:
                try:
                    paths = search_and_download_tess_lc(tic_id, sector=_sector_from_row(row), download_dir=lightcurve_dir)
                except Exception as exc:
                    return [], [{"source_row_index": row_idx, "tic_id": tic_id, "status": "download_failed", "error": repr(exc)}]
    if not paths:
        return [], [{"source_row_index": row_idx, "tic_id": tic_id, "status": "missing_lightcurve"}]

    output_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    for path in paths[: args.max_lightcurves_per_target]:
        raw = load_tess_fits(path)
        if raw.status != "RAW_LOADED":
            manifest_rows.append({"source_row_index": row_idx, "tic_id": tic_id, "fits_path": str(path), "status": raw.status, "error": raw.error})
            continue

        candidate = _candidate_from_ephemeris(row, raw, candidate_id=row_idx + 1, epoch_offset=args.epoch_offset)
        try:
            if candidate is not None and not args.ignore_ephemeris:
                feature_row = _row_from_seeded_candidate(raw, candidate, label, config, n_bootstrap=args.n_bootstrap)
                feature_row["source_row_index"] = row_idx
                feature_row["fits_path"] = str(path)
                feature_row["feature_source"] = "curated_ephemeris"
                output_rows.append(feature_row)
                manifest_rows.append({"source_row_index": row_idx, "tic_id": tic_id, "fits_path": str(path), "status": "ok_ephemeris"})
            else:
                result = run_parts_1_to_5_from_raw(raw, config=config)
                selected = _select_catalog_rows(result["catalog"], label, all_candidates=args.all_detected_candidates)
                if selected.empty:
                    manifest_rows.append({"source_row_index": row_idx, "tic_id": tic_id, "fits_path": str(path), "status": "no_detected_candidate", "pipeline_status": result["detection"].status})
                    continue
                selected = selected.copy()
                selected["source_row_index"] = row_idx
                selected["fits_path"] = str(path)
                selected["feature_source"] = "detected_candidate"
                output_rows.extend(selected.to_dict(orient="records"))
                manifest_rows.append({"source_row_index": row_idx, "tic_id": tic_id, "fits_path": str(path), "status": "ok_detected", "n_rows": int(len(selected))})
        except Exception as exc:
            manifest_rows.append({"source_row_index": row_idx, "tic_id": tic_id, "fits_path": str(path), "status": "feature_failed", "error": repr(exc)})
    return output_rows, manifest_rows


def parse_args():
    parser = argparse.ArgumentParser(description="Build a labeled Parts 1-5 candidate-feature catalog from curated TIC labels and TESS FITS light curves.")
    parser.add_argument("curated_csv", help="Curated CSV with TIC IDs, labels, and optionally ephemerides")
    parser.add_argument("--source", default=None, choices=["tess-toi", "kepler-dr25", "tess-dv"], help="Normalize a public metadata CSV before feature building")
    parser.add_argument("--label-col", default="label", help="Column containing labels for supervised training")
    parser.add_argument("--lightcurve-dir", default="data/public/lightcurves")
    parser.add_argument("--output-dir", default="outputs_labeled_candidates")
    parser.add_argument("--output-csv", default="labeled_candidate_features.csv")
    parser.add_argument("--download-missing", action="store_true", help="Download missing light curves from MAST")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--max-lightcurves-per-target", type=int, default=1)
    parser.add_argument("--all-detected-candidates", action="store_true", help="Keep every detected candidate when no ephemeris is available")
    parser.add_argument("--ignore-ephemeris", action="store_true", help="Always run blind detection instead of seeding from provided ephemerides")
    parser.add_argument("--epoch-offset", type=float, default=None, help="Subtract this from large BJD epochs, e.g. 2457000 for TESS BTJD")
    parser.add_argument("--n-periods", type=int, default=2000)
    parser.add_argument("--method", choices=["bls", "tls", "both"], default="bls")
    parser.add_argument("--no-variants", action="store_true")
    parser.add_argument("--detrend-method", default="rolling_median", choices=["rolling_median", "wotan_biweight", "none"])
    parser.add_argument("--quality-mask-mode", default="conservative", choices=["none", "minimal", "conservative", "strict"])
    parser.add_argument("--min-clean-points", type=int, default=500)
    parser.add_argument("--n-bootstrap", type=int, default=80)
    parser.add_argument("--n-workers", type=int, default=4, help="Concurrent row workers for MAST download/FITS feature building")
    parser.add_argument("--progress-every", type=int, default=1, help="Print progress every N completed rows")
    parser.add_argument("--checkpoint-every", type=int, default=10, help="Rewrite partial output CSVs every N completed rows; use 0 to disable")
    return parser.parse_args()


def _sort_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def key(row: dict[str, Any]) -> tuple[Any, ...]:
        return (
            row.get("source_row_index", 10**12),
            row.get("tic_id", 10**12),
            str(row.get("fits_path", "")),
            row.get("candidate_id", 10**12),
        )

    return sorted(rows, key=key)


def _write_outputs(
    output_dir: Path,
    output_csv: str,
    feature_rows: list[dict[str, Any]],
    manifest_rows: list[dict[str, Any]],
    n_input_rows: int,
    *,
    complete: bool,
) -> tuple[Path, Path, Path]:
    features = pd.DataFrame(_sort_rows(feature_rows))
    manifest = pd.DataFrame(_sort_rows(manifest_rows))
    features_path = output_dir / output_csv
    manifest_path = output_dir / "labeled_candidate_manifest.csv"
    summary_path = output_dir / "labeled_candidate_summary.json"
    features.to_csv(features_path, index=False)
    manifest.to_csv(manifest_path, index=False)
    summary = {
        "complete": bool(complete),
        "n_input_rows": int(n_input_rows),
        "n_completed_source_rows": int(manifest["source_row_index"].nunique()) if "source_row_index" in manifest else 0,
        "n_feature_rows": int(len(features)),
        "status_counts": manifest["status"].value_counts(dropna=False).to_dict() if "status" in manifest else {},
        "label_counts": features["label"].value_counts(dropna=False).to_dict() if "label" in features else {},
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return features_path, manifest_path, summary_path


def _format_eta(seconds: float | None) -> str:
    if seconds is None or not np.isfinite(seconds) or seconds < 0:
        return "unknown"
    seconds = int(seconds)
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def _progress_line(
    completed: int,
    total: int,
    feature_rows: list[dict[str, Any]],
    manifest_rows: list[dict[str, Any]],
    started: float,
) -> str:
    elapsed = max(time.monotonic() - started, 1e-6)
    rate = completed / elapsed
    eta = (total - completed) / rate if rate > 0 else None
    counts = Counter(str(row.get("status", "unknown")) for row in manifest_rows)
    status_text = ", ".join(f"{k}={v}" for k, v in counts.most_common(5)) or "none"
    pct = 100.0 * completed / max(total, 1)
    return (
        f"[{completed}/{total} {pct:5.1f}%] "
        f"features={len(feature_rows)} "
        f"rate={rate * 60.0:.2f} rows/min "
        f"eta={_format_eta(eta)} "
        f"statuses: {status_text}"
    )


def _run_rows(df: pd.DataFrame, args, config: PipelineConfig, lightcurve_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    row_items = list(df.iterrows())
    total = len(row_items)
    feature_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    started = time.monotonic()
    progress_every = max(1, int(args.progress_every))
    checkpoint_every = max(0, int(args.checkpoint_every))
    output_dir = Path(args.output_dir)

    print(
        f"Building labeled features for {total} rows with n_workers={args.n_workers}, "
        f"download_missing={args.download_missing}, n_periods={args.n_periods}, n_bootstrap={args.n_bootstrap}",
        flush=True,
    )

    def handle_result(completed: int, rows: list[dict[str, Any]], manifest: list[dict[str, Any]]) -> None:
        feature_rows.extend(rows)
        manifest_rows.extend(manifest)
        should_print = completed == 1 or completed == total or completed % progress_every == 0
        if should_print:
            print(_progress_line(completed, total, feature_rows, manifest_rows, started), flush=True)
        if checkpoint_every and (completed % checkpoint_every == 0 or completed == total):
            _write_outputs(output_dir, args.output_csv, feature_rows, manifest_rows, total, complete=False)

    if args.n_workers <= 1:
        for completed, (row_idx, row) in enumerate(row_items, 1):
            rows, manifest = _process_row(row_idx, row, args, config, lightcurve_dir)
            handle_result(completed, rows, manifest)
        return feature_rows, manifest_rows

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, int(args.n_workers))) as executor:
        future_to_row = {
            executor.submit(_process_row, row_idx, row, args, config, lightcurve_dir): int(row_idx)
            for row_idx, row in row_items
        }
        for completed, fut in enumerate(concurrent.futures.as_completed(future_to_row), 1):
            row_idx = future_to_row[fut]
            try:
                rows, manifest = fut.result()
            except Exception as exc:
                rows = []
                manifest = [{"source_row_index": row_idx, "status": "worker_failed", "error": repr(exc)}]
            handle_result(completed, rows, manifest)
    return feature_rows, manifest_rows


def main():
    args = parse_args()
    df = read_public_metadata(args.curated_csv, args.source) if args.source else pd.read_csv(args.curated_csv)
    if args.max_rows is not None:
        df = df.head(args.max_rows).copy()

    output_dir = Path(args.output_dir)
    lightcurve_dir = Path(args.lightcurve_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    lightcurve_dir.mkdir(parents=True, exist_ok=True)

    config = PipelineConfig(
        n_periods=args.n_periods,
        detection_method=args.method,
        detection_use_variants=not args.no_variants,
        detrend_method=args.detrend_method,
        quality_mask_mode=args.quality_mask_mode,
        min_clean_points=args.min_clean_points,
        make_plots=False,
    )

    feature_rows, manifest_rows = _run_rows(df, args, config, lightcurve_dir)
    features_path, manifest_path, summary_path = _write_outputs(
        output_dir,
        args.output_csv,
        feature_rows,
        manifest_rows,
        len(df),
        complete=True,
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    print(json.dumps(summary, indent=2), flush=True)
    print(f"features: {features_path}")
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
