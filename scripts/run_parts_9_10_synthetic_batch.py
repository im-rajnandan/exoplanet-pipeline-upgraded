#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from exoplanet_pipeline.config import PipelineConfig
from exoplanet_pipeline.batch import BatchRunConfig, run_raw_lightcurve_batch
from exoplanet_pipeline.cnn import load_cnn_bundle
from exoplanet_pipeline.synthetic import make_synthetic_transit_lc, make_synthetic_eb_lc, make_synthetic_blend_lc
from exoplanet_pipeline.final_outputs import generate_submission_package_outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Parts 1–10 on three synthetic demonstration targets.")
    parser.add_argument("--output-dir", type=str, default=str(ROOT / "outputs_parts_9_10"), help="Directory for batch outputs")
    parser.add_argument("--n-periods", type=int, default=500, help="Number of BLS trial periods for the fast demo")
    parser.add_argument("--method", choices=["bls", "tls", "both"], default="bls")
    parser.add_argument("--resume", action="store_true", help="Resume from cached per-target outputs when available")
    parser.add_argument("--make-plots", action="store_true", help="Enable per-candidate plots; off by default for speed")
    parser.add_argument("--cnn-model", type=str, default=None, help="Optional CNN bundle directory or cnn_model.pt path")
    args = parser.parse_args()

    out = Path(args.output_dir)
    raws = [
        make_synthetic_transit_lc(tic_id=910001, period_days=3.0, depth_ppm=1500, noise_ppm=300, random_seed=101),
        make_synthetic_eb_lc(tic_id=920001, period_days=4.0, primary_depth_ppm=18000, secondary_depth_ppm=5000, random_seed=201),
        make_synthetic_blend_lc(tic_id=930001, period_days=3.5, observed_depth_ppm=1200, centroid_shift_pix=0.03, random_seed=301),
    ]

    # Keep the demo fast. For real sector runs, increase n_periods and use resume=True.
    pipeline_config = PipelineConfig(
        n_periods=args.n_periods,
        detection_method=args.method,
        make_plots=args.make_plots,
        detection_use_variants=False,
    )
    batch_config = BatchRunConfig(output_dir=out, cache_dir=out / "cache", resume=args.resume, write_heartbeat_every=3)
    cnn_bundle = load_cnn_bundle(args.cnn_model) if args.cnn_model else None
    result = run_raw_lightcurve_batch(raws, cnn_bundle=cnn_bundle, pipeline_config=pipeline_config, batch_config=batch_config)
    paths = generate_submission_package_outputs(result["final_candidate_catalog"], out / "submission_assets")
    print("Parts 9–10 synthetic batch complete.")
    print(f"Targets processed: {len(result['target_summary'])}")
    print(f"Candidates found: {len(result['final_candidate_catalog'])}")
    print("Main outputs:")
    print(f"  Final catalog: {out / 'batch_final_candidate_catalog.csv'}")
    print(f"  Target summary: {out / 'batch_target_summary.csv'}")
    print(f"  Failure log: {out / 'batch_failure_log.csv'}")
    for name, path in paths.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
