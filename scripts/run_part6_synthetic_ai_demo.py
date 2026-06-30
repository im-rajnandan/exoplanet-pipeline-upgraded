from pathlib import Path
import pandas as pd

from exoplanet_pipeline.ml import train_ai_classifier, predict_ai_classifier, write_training_artifacts
from exoplanet_pipeline.ml_synthetic import make_synthetic_ml_feature_catalog
from exoplanet_pipeline.ml_diagnostics import plot_confusion_matrix, plot_feature_importance, plot_prediction_probability_bars


def main():
    out_dir = Path("outputs_part6_ai")
    out_dir.mkdir(exist_ok=True)

    catalog = make_synthetic_ml_feature_catalog(n_per_class=45, random_seed=2026)
    catalog_path = out_dir / "part6_synthetic_labeled_feature_catalog.csv"
    catalog.to_csv(catalog_path, index=False)

    result = train_ai_classifier(
        catalog,
        label_col="label",
        model_type="random_forest",
        calibrate=False,
        test_size=0.25,
        random_state=2026,
    )
    paths = write_training_artifacts(result, out_dir)

    plot_confusion_matrix(
        result.confusion_matrix,
        result.class_names,
        out_dir / "part6_confusion_matrix.png",
    )
    plot_feature_importance(
        result.feature_importance,
        out_dir / "part6_feature_importance.png",
    )

    # Demonstrate prediction on an unlabeled copy of a small subset.
    predict_input = catalog.sample(n=24, random_state=7).drop(columns=["label"])
    predictions = predict_ai_classifier(result.model_bundle, predict_input)
    pred_path = out_dir / "part6_demo_predictions.csv"
    predictions.to_csv(pred_path, index=False)
    plot_prediction_probability_bars(predictions, out_dir / "part6_prediction_probabilities.png")

    print("Part 6 synthetic AI demo complete.")
    print(f"Labeled catalog: {catalog_path}")
    print(f"Model: {paths['model']}")
    print(f"Metrics: {paths['metrics']}")
    print(f"Predictions: {pred_path}")
    print("Test metrics:")
    print({k: v for k, v in result.test_metrics.items() if k != "classification_report"})
    if result.warnings:
        print("Warnings:", result.warnings)


if __name__ == "__main__":
    main()
