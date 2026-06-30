#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from exoplanet_pipeline.batch import BatchRunConfig, run_fits_file_batch
from exoplanet_pipeline.cnn import load_cnn_bundle
from exoplanet_pipeline.config import PipelineConfig
from exoplanet_pipeline.final_outputs import generate_submission_package_outputs
from exoplanet_pipeline.ingest import search_and_download_tess_lc
from exoplanet_pipeline.public_data import read_tic_ctl_catalog, write_tic_ctl_target_list


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the exoplanet pipeline from a TIC/CTL target catalog.")
    parser.add_argument("catalog_csv", help="Local TIC/CTL CSV, TIC .csv.gz chunk, or standardized target-list CSV")
    parser.add_argument("--catalog-type", default="auto", choices=["auto", "ctl", "tic", "xctl", "target-list"])
    parser.add_argument("--header-file", default=None)
    parser.add_argument("--nrows", type=int, default=None)
    parser.add_argument("--max-targets", type=int, default=100)
    parser.add_argument("--sector", type=int, default=None)
    parser.add_argument("--download-dir", default="data/public/lightcurves")
    parser.add_argument("--output-dir", default="outputs_tic_ctl_pipeline")
    parser.add_argument("--cnn-model", default=None, help="Optional CNN bundle directory or cnn_model.pt path")
    parser.add_argument("--n-periods", type=int, default=2000)
    parser.add_argument("--method", choices=["bls", "tls", "both"], default="bls")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.catalog_type == "target-list":
        targets = pd.read_csv(args.catalog_csv)
    else:
        targets = read_tic_ctl_catalog(args.catalog_csv, catalog_type=args.catalog_type, nrows=args.nrows, header_file=args.header_file)
    target_list_path = write_tic_ctl_target_list(targets, output_dir / "tic_ctl_target_list.csv", max_targets=args.max_targets)
    targets = pd.read_csv(target_list_path)

    download_dir = Path(args.download_dir)
    download_dir.mkdir(parents=True, exist_ok=True)
    fits_paths: list[Path] = []
    download_rows: list[dict] = []
    for _, row in targets.iterrows():
        tic_id = int(row["tic_id"])
        try:
            paths = search_and_download_tess_lc(tic_id, sector=args.sector, download_dir=download_dir)
            fits_paths.extend(paths)
            download_rows.append({"tic_id": tic_id, "n_files": len(paths), "status": "ok" if paths else "no_lightcurve"})
        except Exception as exc:
            download_rows.append({"tic_id": tic_id, "n_files": 0, "status": "download_failed", "error": repr(exc)})

    download_manifest = pd.DataFrame(download_rows)
    download_manifest.to_csv(output_dir / "tic_ctl_download_manifest.csv", index=False)
    (output_dir / "download_summary.json").write_text(
        json.dumps(
            {
                "n_targets": int(len(targets)),
                "n_fits_files": int(len(fits_paths)),
                "status_counts": download_manifest["status"].value_counts(dropna=False).to_dict() if not download_manifest.empty else {},
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    if not fits_paths:
        print(f"No TESS light-curve FITS files found/downloaded. See {output_dir / 'tic_ctl_download_manifest.csv'}")
        return

    pipeline_config = PipelineConfig(n_periods=args.n_periods, detection_method=args.method, make_plots=False, detection_use_variants=False)
    batch_config = BatchRunConfig(output_dir=output_dir, cache_dir=output_dir / "cache", resume=not args.no_resume, max_targets=None)
    cnn_bundle = load_cnn_bundle(args.cnn_model) if args.cnn_model else None
    result = run_fits_file_batch(fits_paths, cnn_bundle=cnn_bundle, pipeline_config=pipeline_config, batch_config=batch_config)
    paths = generate_submission_package_outputs(result["final_candidate_catalog"], output_dir / "submission_assets")
    print("TIC/CTL pipeline complete.")
    print(f"Targets: {len(targets)}")
    print(f"FITS files: {len(fits_paths)}")
    print(f"Candidates: {len(result['final_candidate_catalog'])}")
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
