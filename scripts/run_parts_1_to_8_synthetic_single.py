#!/usr/bin/env python3
"""Run one end-to-end Parts 1–8 example on a synthetic planet-like light curve."""
from __future__ import annotations

import argparse
from pathlib import Path

from exoplanet_pipeline.config import PipelineConfig
from exoplanet_pipeline.synthetic import make_synthetic_transit_lc
from exoplanet_pipeline.pipeline_parts_1_to_8 import run_parts_1_to_8_from_raw
from exoplanet_pipeline.cnn import load_cnn_bundle
from exoplanet_pipeline.diagnostics import plot_preprocessing, plot_detection


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one end-to-end Parts 1-8 example on a synthetic planet-like light curve.")
    parser.add_argument("--output-dir", default="outputs_parts_1_to_8_single")
    parser.add_argument("--cnn-model", default=None, help="Optional CNN bundle directory or cnn_model.pt path")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = make_synthetic_transit_lc(period_days=3.0, depth_ppm=1200, duration_hours=2.0, noise_ppm=300)
    config = PipelineConfig(n_periods=1200, n_durations=8, detection_use_variants=False, make_plots=False)
    cnn_bundle = load_cnn_bundle(args.cnn_model) if args.cnn_model else None
    result = run_parts_1_to_8_from_raw(raw, cnn_bundle=cnn_bundle, config=config)
    result["catalog"].to_csv(out_dir / "parts_1_to_8_single_catalog.csv", index=False)
    plot_preprocessing(result["clean"], out_dir / "single_preprocessing.png")
    if result["detection"].best_candidate is not None:
        plot_detection(result["clean"], result["detection"], out_dir / "single_detection.png")
    print(result["catalog"].T)
    print("Wrote", out_dir)


if __name__ == "__main__":
    main()
