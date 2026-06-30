#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path
import argparse
import sys
import os

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from exoplanet_pipeline.config import PipelineConfig
from exoplanet_pipeline.batch import BatchRunConfig, discover_fits_files, run_fits_file_batch
from exoplanet_pipeline.cnn import load_cnn_bundle
from exoplanet_pipeline.final_outputs import generate_submission_package_outputs
from exoplanet_pipeline.ml import load_model_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Parts 1–10 on a directory of local TESS FITS light curves.")
    parser.add_argument("fits_dir", type=str, help="Directory containing .fits/.fits.gz TESS light curves")
    parser.add_argument("--output-dir", type=str, default="outputs_parts_9_10_fits")
    parser.add_argument("--max-targets", type=int, default=None)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--n-periods", type=int, default=2000)
    parser.add_argument("--period-min-days", type=float, default=0.20)
    parser.add_argument("--period-max-days", type=float, default=None)
    parser.add_argument("--method", choices=["bls", "tls", "both"], default="bls")
    parser.add_argument("--detrend-method", choices=["rolling_median", "wotan_biweight", "none"], default="rolling_median")
    parser.add_argument("--quality-mask-mode", choices=["none", "minimal", "conservative", "strict"], default="conservative")
    parser.add_argument("--min-clean-points", type=int, default=500)
    parser.add_argument("--ai-model", type=str, default=None, help="Optional sklearn AI model bundle .joblib")
    parser.add_argument("--cnn-model", type=str, default=None, help="Optional CNN bundle directory or cnn_model.pt path")
    parser.add_argument("--n-workers", type=int, default=os.cpu_count() or 2, help="Number of parallel execution workers")
    parser.add_argument("--timeout-seconds", type=float, default=300.0, help="Per-FITS processing timeout; use 0 to disable")
    parser.add_argument("--validation-report", type=str, default=None, help="Optional validation_report.json to include in generated report assets")
    parser.add_argument("--no-variants", action="store_true", help="Disable searching across multiple detrending variants (speed up search by 4x)")
    args = parser.parse_args()

    fits_files = discover_fits_files(args.fits_dir, recursive=True)
    if args.max_targets is not None:
        fits_files = fits_files[: args.max_targets]
    print(f"Found {len(fits_files)} FITS files")
    if not fits_files:
        parser.error(f"No FITS files found under {args.fits_dir!r}")

    output_dir = Path(args.output_dir)
    pipeline_config = PipelineConfig(
        n_periods=args.n_periods,
        period_min_days=args.period_min_days,
        period_max_days=args.period_max_days,
        detection_method=args.method,
        detrend_method=args.detrend_method,
        quality_mask_mode=args.quality_mask_mode,
        min_clean_points=args.min_clean_points,
        make_plots=False,
        detection_use_variants=not args.no_variants
    )
    batch_config = BatchRunConfig(
        output_dir=output_dir,
        cache_dir=output_dir / "cache",
        resume=not args.no_resume,
        max_targets=args.max_targets,
        n_workers=args.n_workers,
        timeout_seconds=None if args.timeout_seconds == 0 else args.timeout_seconds,
    )
    model_bundle = load_model_bundle(args.ai_model) if args.ai_model else None
    cnn_bundle = load_cnn_bundle(args.cnn_model) if args.cnn_model else None
    result = run_fits_file_batch(fits_files, model_bundle=model_bundle, cnn_bundle=cnn_bundle, pipeline_config=pipeline_config, batch_config=batch_config)
    paths = generate_submission_package_outputs(
        result["final_candidate_catalog"],
        output_dir / "submission_assets",
        validation_report_path=args.validation_report,
    )

    print("Batch complete.")
    print(f"Targets processed: {len(result['target_summary'])}")
    print(f"Failures: {len(result['failures'])}")
    print(f"Candidates: {len(result['final_candidate_catalog'])}")
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
