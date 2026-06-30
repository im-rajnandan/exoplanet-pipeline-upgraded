from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Any
import hashlib
import json
import time
import traceback

import numpy as np
import pandas as pd

from .config import PipelineConfig
from .schema import RawLightCurve
from .pipeline_parts_1_to_8 import run_parts_1_to_8_from_raw, run_parts_1_to_8_from_fits
from .final_catalog import harmonize_candidate_catalog, summarize_final_catalog


@dataclass
class BatchRunConfig:
    """Part 9 configuration for sector-scale processing.

    This manager is intentionally conservative and file-based. It writes partial
    target outputs as soon as each light curve finishes, so a long sector run can
    resume after interruption without losing completed work.
    """

    output_dir: str | Path = "outputs_parts_9_10_batch"
    cache_dir: str | Path = "outputs_parts_9_10_batch/cache"
    resume: bool = True
    save_per_target_catalogs: bool = True
    make_final_catalog: bool = True
    continue_on_error: bool = True
    max_targets: int | None = None
    run_id: str | None = None
    write_heartbeat_every: int = 10
    n_workers: int = 1
    timeout_seconds: float | None = 60.0

    def resolved_run_id(self) -> str:
        if self.run_id:
            return self.run_id
        return time.strftime("run_%Y%m%d_%H%M%S")


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (Path,)):
        return str(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    return str(obj)


def _target_key_from_path(path: str | Path) -> str:
    p = Path(path)
    h = hashlib.sha1(str(p.resolve()).encode("utf-8")).hexdigest()[:10]
    return f"fits_{p.stem}_{h}".replace(" ", "_")


def _target_key_from_raw(raw: RawLightCurve, index: int) -> str:
    tic = raw.tic_id if raw.tic_id is not None else f"idx{index}"
    sec = raw.sector if raw.sector is not None else "unknown_sector"
    return f"tic_{tic}_s{sec}_{index:05d}"


def _write_manifest(output_dir: Path, batch_config: BatchRunConfig, pipeline_config: PipelineConfig, items: list[Any]) -> Path:
    manifest = {
        "run_id": batch_config.resolved_run_id(),
        "created_unix_time": time.time(),
        "n_input_items": len(items),
        "batch_config": asdict(batch_config),
        "pipeline_config": {k: _json_default(v) for k, v in asdict(pipeline_config).items()},
    }
    path = output_dir / "batch_run_manifest.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=_json_default)
    return path


def _result_to_rows(result: dict) -> pd.DataFrame:
    catalog = result.get("catalog")
    if isinstance(catalog, pd.DataFrame) and not catalog.empty:
        return catalog.copy()
    return pd.DataFrame()


def _target_summary_row(key: str, result: dict | None = None, error: str | None = None, source: str | None = None) -> dict[str, Any]:
    if error is not None or result is None:
        return {
            "target_key": key,
            "source": source,
            "status": "FAILED",
            "n_candidates": 0,
            "error": error,
        }
    clean = result.get("clean")
    detection = result.get("detection")
    catalog = result.get("catalog")
    n_candidates = len(catalog) if isinstance(catalog, pd.DataFrame) else 0
    return {
        "target_key": key,
        "source": source,
        "tic_id": getattr(clean, "tic_id", None),
        "sector": getattr(clean, "sector", None),
        "status": getattr(detection, "status", "OK"),
        "clean_status": getattr(clean, "status", None),
        "n_candidates": int(n_candidates),
        "best_snr": _safe_max(catalog, ["unc_effective_snr", "fit_snr", "local_snr", "snr"]),
        "best_confidence": _safe_max(catalog, ["unc_final_confidence", "final_confidence", "class_confidence"]),
        "n_clean_points": getattr(clean, "qc", {}).get("n_final", None) if clean is not None else None,
        "noise_ppm": getattr(clean, "qc", {}).get("robust_noise_ppm", None) if clean is not None else None,
        "error": None,
    }


def _safe_max(df: Any, cols: list[str]) -> float | None:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return None
    for c in cols:
        if c in df.columns:
            vals = pd.to_numeric(df[c], errors="coerce")
            if vals.notna().any():
                return float(vals.max())
    return None


def _process_single_target_raw(
    raw: RawLightCurve,
    key: str,
    model_bundle: dict | None,
    cnn_bundle: dict | None,
    pipeline_config: PipelineConfig,
    batch_config: BatchRunConfig,
    per_target_path: Path,
    summary_path: Path,
) -> tuple[str, str, pd.DataFrame | None, dict, str | None]:
    try:
        result = run_parts_1_to_8_from_raw(raw, model_bundle=model_bundle, cnn_bundle=cnn_bundle, config=pipeline_config)
        df = _result_to_rows(result)
        if not df.empty:
            df.insert(0, "target_key", key)
            if batch_config.save_per_target_catalogs:
                df.to_csv(per_target_path, index=False)
        row = _target_summary_row(key, result=result, source="raw_object")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(row, f, indent=2, default=_json_default)
        return key, "OK", df, row, None
    except Exception as exc:
        tb = traceback.format_exc()
        row = _target_summary_row(key, error=str(exc), source="raw_object")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(row, f, indent=2, default=_json_default)
        return key, "FAILED", None, row, f"{exc}\n{tb}"


def run_raw_lightcurve_batch(
    raw_lightcurves: Iterable[RawLightCurve],
    model_bundle: dict | None = None,
    cnn_bundle: dict | None = None,
    pipeline_config: PipelineConfig | None = None,
    batch_config: BatchRunConfig | None = None,
) -> dict[str, Any]:
    """Run Parts 1–8 on many already-loaded RawLightCurve objects."""
    pipeline_config = pipeline_config or PipelineConfig()
    batch_config = batch_config or BatchRunConfig()
    output_dir = Path(batch_config.output_dir)
    cache_dir = Path(batch_config.cache_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    items = list(raw_lightcurves)
    if batch_config.max_targets is not None:
        items = items[: batch_config.max_targets]
    _write_manifest(output_dir, batch_config, pipeline_config, items)

    all_rows: list[pd.DataFrame] = []
    target_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []

    # 1. Check for skipped/resume targets
    active_items = []
    for i, raw in enumerate(items):
        key = _target_key_from_raw(raw, i)
        per_target_path = cache_dir / f"{key}_catalog.csv"
        summary_path = cache_dir / f"{key}_summary.json"
        if batch_config.resume and summary_path.exists():
            if per_target_path.exists():
                try:
                    df = pd.read_csv(per_target_path)
                    if not df.empty:
                        all_rows.append(df)
                except pd.errors.EmptyDataError:
                    pass
            with open(summary_path, "r", encoding="utf-8") as f:
                target_rows.append(json.load(f))
            continue
        active_items.append((raw, key, per_target_path, summary_path))

    if not active_items:
        return _write_running_outputs(output_dir, all_rows, target_rows, failure_rows, final=batch_config.make_final_catalog)

    # 2. Sequential execution
    if batch_config.n_workers <= 1:
        for i, (raw, key, per_target_path, summary_path) in enumerate(active_items):
            print(f"[{i+1}/{len(active_items)}] Processing target {key}...", flush=True)
            key, status, df, row, err_str = _process_single_target_raw(
                raw, key, model_bundle, cnn_bundle, pipeline_config, batch_config, per_target_path, summary_path
            )
            print(f"  Finished target {key} with status: {status}", flush=True)
            if df is not None and not df.empty:
                all_rows.append(df)
            target_rows.append(row)
            if status == "FAILED":
                failure_rows.append({"target_key": key, "error": err_str})
                if not batch_config.continue_on_error:
                    raise ValueError(f"Task failed: {err_str}")
            if batch_config.write_heartbeat_every and (i + 1) % batch_config.write_heartbeat_every == 0:
                _write_running_outputs(output_dir, all_rows, target_rows, failure_rows)
    # 3. Parallel execution with timeouts
    else:
        import concurrent.futures
        with concurrent.futures.ProcessPoolExecutor(max_workers=batch_config.n_workers) as executor:
            futures = {}
            for raw, key, per_target_path, summary_path in active_items:
                fut = executor.submit(
                    _process_single_target_raw,
                    raw, key, model_bundle, cnn_bundle, pipeline_config, batch_config, per_target_path, summary_path
                )
                futures[fut] = (key, raw, summary_path)

            for fut in concurrent.futures.as_completed(futures):
                key, raw, summary_path = futures[fut]
                try:
                    _, status, df, row, err_str = fut.result(timeout=batch_config.timeout_seconds)
                except concurrent.futures.TimeoutError:
                    status = "FAILED"
                    df = None
                    err_str = f"Target {key} exceeded timeout of {batch_config.timeout_seconds} seconds."
                    row = _target_summary_row(key, error="TIMEOUT", source="raw_object")
                    with open(summary_path, "w", encoding="utf-8") as f:
                        json.dump(row, f, indent=2, default=_json_default)
                except Exception as exc:
                    status = "FAILED"
                    df = None
                    err_str = str(exc)
                    row = _target_summary_row(key, error=str(exc), source="raw_object")
                    with open(summary_path, "w", encoding="utf-8") as f:
                        json.dump(row, f, indent=2, default=_json_default)

                if df is not None and not df.empty:
                    all_rows.append(df)
                target_rows.append(row)
                if status == "FAILED":
                    failure_rows.append({"target_key": key, "error": err_str})
                    if not batch_config.continue_on_error:
                        executor.shutdown(wait=False, cancel_futures=True)
                        raise ValueError(f"Task failed: {err_str}")
                _write_running_outputs(output_dir, all_rows, target_rows, failure_rows)

    return _write_running_outputs(output_dir, all_rows, target_rows, failure_rows, final=batch_config.make_final_catalog)


def _process_single_target_fits(
    path: Path,
    key: str,
    model_bundle: dict | None,
    cnn_bundle: dict | None,
    pipeline_config: PipelineConfig,
    batch_config: BatchRunConfig,
    per_target_path: Path,
    summary_path: Path,
) -> tuple[str, str, pd.DataFrame | None, dict, str | None]:
    try:
        result = run_parts_1_to_8_from_fits(str(path), model_bundle=model_bundle, cnn_bundle=cnn_bundle, config=pipeline_config)
        df = _result_to_rows(result)
        if not df.empty:
            df.insert(0, "target_key", key)
            df.insert(1, "source_file", str(path))
            if batch_config.save_per_target_catalogs:
                df.to_csv(per_target_path, index=False)
        row = _target_summary_row(key, result=result, source=str(path))
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(row, f, indent=2, default=_json_default)
        return key, "OK", df, row, None
    except Exception as exc:
        tb = traceback.format_exc()
        row = _target_summary_row(key, error=str(exc), source=str(path))
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(row, f, indent=2, default=_json_default)
        return key, "FAILED", None, row, f"{exc}\n{tb}"


def run_fits_file_batch(
    fits_files: Iterable[str | Path],
    model_bundle: dict | None = None,
    cnn_bundle: dict | None = None,
    pipeline_config: PipelineConfig | None = None,
    batch_config: BatchRunConfig | None = None,
) -> dict[str, Any]:
    """Run Parts 1–8 on many local TESS FITS light-curve files."""
    pipeline_config = pipeline_config or PipelineConfig()
    batch_config = batch_config or BatchRunConfig()
    output_dir = Path(batch_config.output_dir)
    cache_dir = Path(batch_config.cache_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    files = [Path(p) for p in fits_files]
    if batch_config.max_targets is not None:
        files = files[: batch_config.max_targets]
    _write_manifest(output_dir, batch_config, pipeline_config, files)

    all_rows: list[pd.DataFrame] = []
    target_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []

    # 1. Check for skipped/resume targets
    active_items = []
    for i, path in enumerate(files):
        key = _target_key_from_path(path)
        per_target_path = cache_dir / f"{key}_catalog.csv"
        summary_path = cache_dir / f"{key}_summary.json"
        if batch_config.resume and summary_path.exists():
            if per_target_path.exists():
                try:
                    df = pd.read_csv(per_target_path)
                    if not df.empty:
                        all_rows.append(df)
                except pd.errors.EmptyDataError:
                    pass
            with open(summary_path, "r", encoding="utf-8") as f:
                target_rows.append(json.load(f))
            continue
        active_items.append((path, key, per_target_path, summary_path))

    if not active_items:
        return _write_running_outputs(output_dir, all_rows, target_rows, failure_rows, final=batch_config.make_final_catalog)

    # 2. Sequential execution
    if batch_config.n_workers <= 1:
        for i, (path, key, per_target_path, summary_path) in enumerate(active_items):
            print(f"[{i+1}/{len(active_items)}] Processing target {key} from {path.name}...", flush=True)
            key, status, df, row, err_str = _process_single_target_fits(
                path, key, model_bundle, cnn_bundle, pipeline_config, batch_config, per_target_path, summary_path
            )
            print(f"  Finished target {key} with status: {status}", flush=True)
            if df is not None and not df.empty:
                all_rows.append(df)
            target_rows.append(row)
            if status == "FAILED":
                failure_rows.append({"target_key": key, "source_file": str(path), "error": err_str})
                if not batch_config.continue_on_error:
                    raise ValueError(f"Task failed: {err_str}")
            if batch_config.write_heartbeat_every and (i + 1) % batch_config.write_heartbeat_every == 0:
                _write_running_outputs(output_dir, all_rows, target_rows, failure_rows)
    # 3. Parallel execution with timeouts
    else:
        import concurrent.futures
        with concurrent.futures.ProcessPoolExecutor(max_workers=batch_config.n_workers) as executor:
            futures = {}
            for path, key, per_target_path, summary_path in active_items:
                fut = executor.submit(
                    _process_single_target_fits,
                    path, key, model_bundle, cnn_bundle, pipeline_config, batch_config, per_target_path, summary_path
                )
                futures[fut] = (key, path, summary_path)

            n_completed = 0
            for fut in concurrent.futures.as_completed(futures):
                key, path, summary_path = futures[fut]
                n_completed += 1
                try:
                    _, status, df, row, err_str = fut.result(timeout=batch_config.timeout_seconds)
                except concurrent.futures.TimeoutError:
                    status = "FAILED"
                    df = None
                    err_str = f"Target {key} exceeded timeout of {batch_config.timeout_seconds} seconds."
                    row = _target_summary_row(key, error="TIMEOUT", source=str(path))
                    with open(summary_path, "w", encoding="utf-8") as f:
                        json.dump(row, f, indent=2, default=_json_default)
                except Exception as exc:
                    status = "FAILED"
                    df = None
                    err_str = str(exc)
                    row = _target_summary_row(key, error=str(exc), source=str(path))
                    with open(summary_path, "w", encoding="utf-8") as f:
                        json.dump(row, f, indent=2, default=_json_default)

                print(f"[{n_completed}/{len(active_items)}] Finished target {key} from {path.name} with status: {status}", flush=True)
                if df is not None and not df.empty:
                    all_rows.append(df)
                target_rows.append(row)
                if status == "FAILED":
                    failure_rows.append({"target_key": key, "source_file": str(path), "error": err_str})
                    if not batch_config.continue_on_error:
                        executor.shutdown(wait=False, cancel_futures=True)
                        raise ValueError(f"Task failed: {err_str}")
                _write_running_outputs(output_dir, all_rows, target_rows, failure_rows)

    return _write_running_outputs(output_dir, all_rows, target_rows, failure_rows, final=batch_config.make_final_catalog)


def discover_fits_files(directory: str | Path, recursive: bool = True) -> list[Path]:
    directory = Path(directory)
    patterns = ["*.fits", "*.fits.gz", "*.lc.fits", "*.fits.fz"]
    files: list[Path] = []
    for pat in patterns:
        files.extend(directory.rglob(pat) if recursive else directory.glob(pat))
    return sorted(set(files))


def _write_running_outputs(
    output_dir: Path,
    all_rows: list[pd.DataFrame],
    target_rows: list[dict[str, Any]],
    failure_rows: list[dict[str, Any]],
    final: bool = False,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_catalog = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    target_summary = pd.DataFrame(target_rows)
    failures = pd.DataFrame(failure_rows)

    paths = {
        "raw_candidate_catalog": output_dir / "batch_raw_candidate_catalog.csv",
        "target_summary": output_dir / "batch_target_summary.csv",
        "failure_log": output_dir / "batch_failure_log.csv",
        "final_candidate_catalog": output_dir / "batch_final_candidate_catalog.csv",
        "summary_json": output_dir / "batch_final_summary.json",
    }
    raw_catalog.to_csv(paths["raw_candidate_catalog"], index=False)
    target_summary.to_csv(paths["target_summary"], index=False)
    failures.to_csv(paths["failure_log"], index=False)

    final_catalog = harmonize_candidate_catalog(raw_catalog) if final and not raw_catalog.empty else pd.DataFrame()
    if final:
        final_catalog.to_csv(paths["final_candidate_catalog"], index=False)
        with open(paths["summary_json"], "w", encoding="utf-8") as f:
            json.dump({
                "n_targets_processed": int(len(target_summary)),
                "n_failed_targets": int(len(failures)),
                "candidate_summary": summarize_final_catalog(final_catalog),
            }, f, indent=2, default=_json_default)
    return {
        "paths": paths,
        "raw_candidate_catalog": raw_catalog,
        "final_candidate_catalog": final_catalog,
        "target_summary": target_summary,
        "failures": failures,
    }
