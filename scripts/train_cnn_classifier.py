#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from exoplanet_pipeline.cnn import load_cnn_examples, save_cnn_bundle, train_cnn_classifier


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the optional PyTorch CNN candidate vetter from .npz CNN examples.")
    parser.add_argument("examples", help="Directory containing .npz CNN examples, or one .npz file")
    parser.add_argument("--output-dir", default="outputs_cnn")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    examples = load_cnn_examples(args.examples)
    result = train_cnn_classifier(
        examples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        device=args.device,
    )
    paths = save_cnn_bundle(
        result.model,
        args.output_dir,
        result.config,
        result.scalar_scaler,
        result.label_map,
        {**result.metrics, "warnings": result.warnings},
    )
    print("CNN training complete.")
    print("Final metrics:", result.metrics.get("final", {}))
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
