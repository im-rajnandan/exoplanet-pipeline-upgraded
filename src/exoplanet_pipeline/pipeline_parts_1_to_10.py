from __future__ import annotations

from pathlib import Path
from typing import Iterable, Any
import pandas as pd

from .config import PipelineConfig
from .schema import RawLightCurve
from .pipeline_parts_1_to_8 import run_parts_1_to_8_from_raw
from .batch import BatchRunConfig, run_raw_lightcurve_batch, run_fits_file_batch, discover_fits_files
from .final_catalog import harmonize_candidate_catalog, summarize_final_catalog
from .final_outputs import generate_submission_package_outputs


def run_single_target_parts_1_to_10_from_raw(
    raw: RawLightCurve,
    model_bundle: dict | None = None,
    cnn_bundle: dict | None = None,
    config: PipelineConfig | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    result = run_parts_1_to_8_from_raw(raw, model_bundle=model_bundle, cnn_bundle=cnn_bundle, config=config)
    catalog = harmonize_candidate_catalog(result.get("catalog", pd.DataFrame()))
    result["final_catalog"] = catalog
    result["final_summary"] = summarize_final_catalog(catalog)
    if output_dir is not None:
        paths = generate_submission_package_outputs(catalog, output_dir)
        result["submission_paths"] = paths
    return result


def run_sector_like_batch_parts_1_to_10_from_raw(
    raws: Iterable[RawLightCurve],
    model_bundle: dict | None = None,
    cnn_bundle: dict | None = None,
    pipeline_config: PipelineConfig | None = None,
    batch_config: BatchRunConfig | None = None,
    make_submission_outputs: bool = True,
) -> dict[str, Any]:
    batch_result = run_raw_lightcurve_batch(raws, model_bundle=model_bundle, cnn_bundle=cnn_bundle, pipeline_config=pipeline_config, batch_config=batch_config)
    if make_submission_outputs:
        output_dir = Path((batch_config or BatchRunConfig()).output_dir)
        batch_result["submission_paths"] = generate_submission_package_outputs(batch_result["final_candidate_catalog"], output_dir / "submission_assets")
    return batch_result


def run_sector_like_batch_parts_1_to_10_from_fits_dir(
    fits_dir: str | Path,
    model_bundle: dict | None = None,
    cnn_bundle: dict | None = None,
    pipeline_config: PipelineConfig | None = None,
    batch_config: BatchRunConfig | None = None,
    recursive: bool = True,
    make_submission_outputs: bool = True,
) -> dict[str, Any]:
    files = discover_fits_files(fits_dir, recursive=recursive)
    batch_result = run_fits_file_batch(files, model_bundle=model_bundle, cnn_bundle=cnn_bundle, pipeline_config=pipeline_config, batch_config=batch_config)
    if make_submission_outputs:
        output_dir = Path((batch_config or BatchRunConfig()).output_dir)
        batch_result["submission_paths"] = generate_submission_package_outputs(batch_result["final_candidate_catalog"], output_dir / "submission_assets")
    return batch_result
