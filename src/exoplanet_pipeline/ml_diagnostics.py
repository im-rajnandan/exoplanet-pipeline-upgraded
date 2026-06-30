from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_confusion_matrix(cm: list[list[int]] | np.ndarray, labels: list[str], output_path: str | Path) -> None:
    cm_arr = np.asarray(cm)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(max(7, 0.8 * len(labels)), max(6, 0.75 * len(labels))))
    im = ax.imshow(cm_arr)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title("Part 6 AI classifier confusion matrix")
    for i in range(cm_arr.shape[0]):
        for j in range(cm_arr.shape[1]):
            ax.text(j, i, str(cm_arr[i, j]), ha="center", va="center")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_feature_importance(feature_importance: pd.DataFrame, output_path: str | Path, top_n: int = 20) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    top = feature_importance.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, max(5, 0.35 * len(top))))
    ax.barh(top["feature"], top["importance"])
    ax.set_xlabel("Random-forest feature importance")
    ax.set_title(f"Top {len(top)} Part 6 classifier features")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_prediction_probability_bars(predictions: pd.DataFrame, output_path: str | Path, max_rows: int = 12) -> None:
    """Compact probability summary for a small prediction table."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prob_cols = [c for c in predictions.columns if c.startswith("final_prob_")]
    if not prob_cols:
        raise ValueError("No final_prob_* columns found in predictions dataframe.")
    df = predictions.head(max_rows).copy()
    labels = [f"TIC {int(x)}" if pd.notna(x) else f"row {i}" for i, x in enumerate(df.get("tic_id", range(len(df))))]
    bottom = np.zeros(len(df))
    fig, ax = plt.subplots(figsize=(10, max(4, 0.45 * len(df))))
    y = np.arange(len(df))
    for col in prob_cols:
        vals = df[col].to_numpy(dtype=float)
        ax.barh(y, vals, left=bottom, label=col.replace("final_prob_", ""))
        bottom += vals
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Final class probability")
    ax.set_title("Part 6 final AI + guardrail probabilities")
    ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
