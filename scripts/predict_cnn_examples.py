#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from exoplanet_pipeline.cnn import load_cnn_bundle, load_cnn_examples, predict_cnn_examples


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict classes for saved .npz CNN examples.")
    parser.add_argument("cnn_model", help="CNN bundle directory or cnn_model.pt path")
    parser.add_argument("examples", help="Directory containing .npz CNN examples, or one .npz file")
    parser.add_argument("--output-csv", default="outputs_cnn/cnn_predictions.csv")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    bundle = load_cnn_bundle(args.cnn_model, map_location=args.device)
    examples = load_cnn_examples(args.examples, config=bundle["config"])
    pred = predict_cnn_examples(bundle, examples, device=args.device)
    out = Path(args.output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    pred.to_csv(out, index=False)
    print(f"Wrote predictions to {out}")


if __name__ == "__main__":
    main()
