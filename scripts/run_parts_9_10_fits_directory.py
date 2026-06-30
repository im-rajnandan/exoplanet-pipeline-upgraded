#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from exoplanet_pipeline.config import PipelineConfig
from exoplanet_pipeline.batch import BatchRunConfig, discover_fits_files, run_fits_file_batch
from exoplanet_pipeline.cnn import load_cnn_bundle
from exoplanet_pipeline.final_outputs import generate_submission_package_outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Parts 1–10 on a directory of local TESS FITS light curves.")
    parser.add_argument("fits_dir", type=str, help="Directory containing .fits/.fits.gz TESS light curves")
    parser.add_argument("--output-dir", type=str, default="outputs_parts_9_10_fits")
    parser.add_argument("--max-targets", type=int, default=None)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--n-periods", type=int, default=2000)
    parser.add_argument("--method", choices=["bls", "tls", "both"], default="bls")
    parser.add_argument("--cnn-model", type=str, default=None, help="Optional CNN bundle directory or cnn_model.pt path")
    args = parser.parse_args()

    fits_files = discover_fits_files(args.fits_dir, recursive=True)
    if args.max_targets is not None:
        fits_files = fits_files[: args.max_targets]
    print(f"Found {len(fits_files)} FITS files")
    if not fits_files:
        parser.error(f"No FITS files found under {args.fits_dir!r}")

    output_dir = Path(args.output_dir)
    pipeline_config = PipelineConfig(n_periods=args.n_periods, detection_method=args.method, make_plots=False)
    batch_config = BatchRunConfig(output_dir=output_dir, cache_dir=output_dir / "cache", resume=not args.no_resume, max_targets=args.max_targets)
    cnn_bundle = load_cnn_bundle(args.cnn_model) if args.cnn_model else None
    result = run_fits_file_batch(fits_files, cnn_bundle=cnn_bundle, pipeline_config=pipeline_config, batch_config=batch_config)
    paths = generate_submission_package_outputs(result["final_candidate_catalog"], output_dir / "submission_assets")

    print("Batch complete.")
    print(f"Targets processed: {len(result['target_summary'])}")
    print(f"Failures: {len(result['failures'])}")
    print(f"Candidates: {len(result['final_candidate_catalog'])}")
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
