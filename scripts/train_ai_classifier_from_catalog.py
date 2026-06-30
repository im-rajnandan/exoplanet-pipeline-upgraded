from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from exoplanet_pipeline.ml import train_ai_classifier, predict_ai_classifier, write_training_artifacts, load_model_bundle
from exoplanet_pipeline.ml_diagnostics import plot_confusion_matrix, plot_feature_importance


def parse_args():
    p = argparse.ArgumentParser(description="Train Part 6 AI classifier from a labeled candidate catalog CSV.")
    p.add_argument("catalog_csv", help="CSV containing Parts 1-5 features plus a label column")
    p.add_argument("--label-col", default="label", help="Name of the label column")
    p.add_argument("--output-dir", default="outputs_part6_ai", help="Directory for model and reports")
    p.add_argument("--model-type", default="random_forest", choices=["random_forest", "extra_trees", "hist_gbdt"])
    p.add_argument("--no-calibration", action="store_true", help="Disable probability calibration")
    p.add_argument("--test-size", type=float, default=0.25)
    p.add_argument("--random-state", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.catalog_csv)
    result = train_ai_classifier(
        df,
        label_col=args.label_col,
        model_type=args.model_type,
        calibrate=not args.no_calibration,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    paths = write_training_artifacts(result, out_dir)
    plot_confusion_matrix(result.confusion_matrix, result.class_names, out_dir / "part6_confusion_matrix.png")
    plot_feature_importance(result.feature_importance, out_dir / "part6_feature_importance.png")
    print("Training complete.")
    for name, path in paths.items():
        print(f"{name}: {path}")
    print({k: v for k, v in result.test_metrics.items() if k != "classification_report"})


if __name__ == "__main__":
    main()
