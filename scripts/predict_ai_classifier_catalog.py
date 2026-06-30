from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from exoplanet_pipeline.ml import load_model_bundle, predict_ai_classifier
from exoplanet_pipeline.ml_diagnostics import plot_prediction_probability_bars


def parse_args():
    p = argparse.ArgumentParser(description="Predict Part 6 AI classes for an unlabeled candidate catalog CSV.")
    p.add_argument("model_joblib", help="Model bundle produced by train_ai_classifier_from_catalog.py")
    p.add_argument("catalog_csv", help="Unlabeled Parts 1-5 candidate catalog CSV")
    p.add_argument("--output-csv", default="outputs_part6_ai/part6_predictions.csv")
    p.add_argument("--no-guardrails", action="store_true")
    p.add_argument("--rule-weight", type=float, default=0.25)
    return p.parse_args()


def main():
    args = parse_args()
    model = load_model_bundle(args.model_joblib)
    df = pd.read_csv(args.catalog_csv)
    pred = predict_ai_classifier(
        model,
        df,
        apply_physical_guardrails=not args.no_guardrails,
        rule_weight=args.rule_weight,
    )
    out = Path(args.output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    pred.to_csv(out, index=False)
    try:
        plot_prediction_probability_bars(pred, out.with_suffix(".png"))
    except Exception:
        pass
    print(f"Wrote predictions to {out}")


if __name__ == "__main__":
    main()
