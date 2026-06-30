#!/usr/bin/env python3
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from exoplanet_pipeline.cnn import load_cnn_bundle, load_cnn_example_npz, predict_cnn_candidate_views

def run_diagnostics():
    print("============================================================")
    print("           THERMONUCLEAR DEEP DIVE CNN MODEL AUDIT          ")
    print("============================================================")

    cnn_dir = Path("outputs_cnn")
    examples_dir = Path("data/public/cnn_examples")

    if not cnn_dir.exists():
        print(f"Error: {cnn_dir} directory does not exist. Train the model first!")
        sys.exit(1)

    if not examples_dir.exists():
        print(f"Error: {examples_dir} directory does not exist. Examples must exist!")
        sys.exit(1)

    # 1. Load the model bundle
    print("\n[Step 1] Loading CNN Model Bundle...")
    try:
        bundle = load_cnn_bundle(cnn_dir)
        print("✔ Successfully loaded model bundle.")
        print(f"  Model type: {type(bundle['model'])}")
        print(f"  Classes: {bundle['config'].canonical_classes}")
        print(f"  Scalars to expect: {len(bundle['config'].scalar_feature_names)}")
    except Exception as e:
        print(f"❌ Failed to load model bundle: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    model = bundle["model"]
    config = bundle["config"]

    # 2. Check Model Weights & Sanity
    print("\n[Step 2] Auditing Model Weights for NaNs/Infs...")
    nans = 0
    infs = 0
    total_params = 0
    weight_norms = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            total_params += param.numel()
            val = param.data.cpu().numpy()
            nans += np.isnan(val).sum()
            infs += np.isinf(val).sum()
            weight_norms.append((name, float(np.linalg.norm(val))))

    print(f"  Total trainable parameters: {total_params:,}")
    if nans > 0 or infs > 0:
        print(f"❌ WARNING: Found {nans} NaNs and {infs} Infs in parameters!")
    else:
        print("✔ Weights are healthy. No NaNs or Infs found.")
    
    # 3. Load metrics
    print("\n[Step 3] Loading Training History Metrics...")
    metrics_path = cnn_dir / "cnn_metrics.json"
    if metrics_path.exists():
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
        print(f"  Total examples used: {metrics.get('n_examples')}")
        print(f"  Train/Val split: {metrics.get('n_train')} / {metrics.get('n_validation')}")
        history = metrics.get("history", [])
        if history:
            print(f"  Initial Train Loss: {history[0].get('train_loss'):.4f} | Initial Val Acc: {history[0].get('val_accuracy'):.4f}")
            print(f"  Final Train Loss: {history[-1].get('train_loss'):.4f} | Final Val Acc: {history[-1].get('val_accuracy'):.4f}")
            # check if loss decreased
            loss_diff = history[0].get('train_loss', 0) - history[-1].get('train_loss', 0)
            if loss_diff > 0:
                print(f"✔ Model learned: Train loss decreased by {loss_diff:.4f} during training.")
            else:
                print("⚠ Model Warning: Train loss did not decrease. Check learning rate / initialization.")
        else:
            print("⚠ No history available in metrics.")
    else:
        print("⚠ cnn_metrics.json not found.")

    # 4. Load examples and perform batch inference
    print("\n[Step 4] Running Inference on Compiled CNN Examples...")
    example_files = list(examples_dir.glob("*.npz"))
    print(f"  Found {len(example_files)} compiled examples.")
    if not example_files:
        print("❌ Error: No compiled example files (.npz) found.")
        sys.exit(1)

    all_examples = []
    for p in example_files:
        try:
            ex = load_cnn_example_npz(p, config=config)
            all_examples.append(ex)
        except Exception as e:
            print(f"  Failed to load example {p.name}: {e}")

    print(f"  Successfully loaded {len(all_examples)} examples.")

    # Predict
    raw_predictions = []
    class_probabilities = []
    binary_probs = []
    true_classes = []
    
    model.eval()
    with torch.no_grad():
        for ex in all_examples:
            pred = predict_cnn_candidate_views(bundle, ex, catalog_row=ex.metadata, apply_physical_guardrails=False)
            raw_predictions.append(pred)
            
            # Extract probability array for the classes
            probs = [pred[f"cnn_prob_{cls}"] for cls in config.canonical_classes]
            class_probabilities.append(probs)
            binary_probs.append(pred["cnn_binary_planet_probability"])
            
            if ex.canonical_label:
                true_classes.append(ex.canonical_label)
            else:
                true_classes.append("UNKNOWN")

    class_probabilities = np.array(class_probabilities)  # (N, C)
    binary_probs = np.array(binary_probs)  # (N,)

    # 5. Output Diversity Audit
    print("\n[Step 5] Checking Prediction Diversity & Entropy...")
    
    # Calculate variance of predictions across examples for each class
    prob_variances = np.var(class_probabilities, axis=0)
    prob_means = np.mean(class_probabilities, axis=0)
    
    print("  Class Mean Probabilities & Variances:")
    for idx, cls in enumerate(config.canonical_classes):
        print(f"    - {cls:32s}: Mean={prob_means[idx]:.4f}, Var={prob_variances[idx]:.6e}")

    # Check for collapse
    max_var = np.max(prob_variances)
    if max_var < 1e-6:
        print("❌ ALERT: Model predictions have collapsed! The variance is near zero across all classes.")
        print("          The model is predicting the same constant value regardless of input.")
    else:
        print(f"✔ Predictions are diverse. Max variance is {max_var:.6f} (Model is responsive to inputs).")

    # Binary classification check
    bin_var = np.var(binary_probs)
    print(f"  Binary Planet Probability: Mean={np.mean(binary_probs):.4f}, Var={bin_var:.6e}, Min={np.min(binary_probs):.4f}, Max={np.max(binary_probs):.4f}")
    if bin_var < 1e-6:
        print("❌ ALERT: Binary probability predictions have collapsed to a constant value!")
    else:
        print("✔ Binary probability predictions are diverse and responsive.")

    # 6. Evaluation metrics if labels exist
    print("\n[Step 6] Evaluating Model Accuracy against Ground Truth Labels...")
    unique_true, counts_true = np.unique(true_classes, return_counts=True)
    print("  Ground truth class distribution in dataset:")
    for tc, count in zip(unique_true, counts_true):
        print(f"    - {tc:32s}: {count}")

    if len(unique_true) > 1 and "UNKNOWN" not in unique_true:
        preds = [pred["cnn_predicted_class"] for pred in raw_predictions]
        correct = sum(1 for t, p in zip(true_classes, preds) if t == p)
        acc = correct / len(true_classes)
        print(f"  Empirical Accuracy on this dataset: {acc:.4%} ({correct}/{len(true_classes)})")
        
        # Display confusion matrix
        df_cm = pd.crosstab(
            pd.Series(true_classes, name='Actual'),
            pd.Series(preds, name='Predicted')
        )
        print("\n  Confusion Matrix:")
        print(df_cm.to_string())
    else:
        print("  Note: Ground truth labels are uniform or unknown. Skipping confusion matrix evaluation.")

    # Generate Markdown Report
    report_lines = [
        "# CNN Model Deep Audit Diagnostic Report",
        "",
        "## Summary",
        f"- **Model Status**: {'HEALTHY' if max_var >= 1e-6 and nans == 0 else 'CRITICAL (Collapsed or NaN weights)'}",
        f"- **Total Trainable Parameters**: {total_params:,}",
        f"- **Weight NaNs/Infs**: {nans} / {infs}",
        f"- **Dataset Size Checked**: {len(all_examples)} examples",
        f"- **Prediction Variety Check**: Max class variance = {max_var:.6f}",
        "",
        "## Training History Snapshot",
    ]
    
    if metrics_path.exists():
        report_lines.append(f"- **Total Epochs**: {metrics.get('epochs')}")
        report_lines.append(f"- **Final Training Loss**: {metrics.get('history', [{}])[-1].get('train_loss', 'N/A')}")
        report_lines.append(f"- **Final Validation Accuracy**: {metrics.get('history', [{}])[-1].get('val_accuracy', 'N/A')}")
    else:
        report_lines.append("No training history JSON found.")

    report_lines.extend([
        "",
        "## Prediction Statistics per Class",
        "| Class | Mean Prob | Variance |",
        "| :--- | :--- | :--- |"
    ])
    for idx, cls in enumerate(config.canonical_classes):
        report_lines.append(f"| {cls} | {prob_means[idx]:.4f} | {prob_variances[idx]:.6e} |")

    report_lines.extend([
        "",
        "## Weight Norms per Layer",
        "| Layer Name | L2 Norm |",
        "| :--- | :--- |"
    ])
    for name, norm in weight_norms[:15]:  # Top 15 layers
        report_lines.append(f"| {name} | {norm:.4f} |")
    if len(weight_norms) > 15:
        report_lines.append(f"| ... and {len(weight_norms)-15} more layers | |")

    # Save to disk
    report_path = cnn_dir / "diagnostic_report.md"
    report_path.write_text("\n".join(report_lines))
    print(f"\n✔ Diagnostic report written to: {report_path}")
    print("============================================================")

if __name__ == "__main__":
    run_diagnostics()
