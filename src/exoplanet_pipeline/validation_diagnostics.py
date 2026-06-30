from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


def plot_confusion_matrix_from_report(report_dict: dict, output_path: str | Path) -> None:
    import matplotlib.pyplot as plt
    cls = report_dict.get("classification_metrics", {})
    if not cls.get("available") or "confusion_matrix" not in cls:
        return
    cm = np.asarray(cls["confusion_matrix"], dtype=float)
    labels = cls.get("labels", [])
    if cm.size == 0:
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, aspect="auto")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Classification confusion matrix")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.8)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_parameter_recovery(catalog: pd.DataFrame, output_path: str | Path) -> None:
    import matplotlib.pyplot as plt
    candidates = [
        ("true_period_days", "fit_period_days", "Period (days)"),
        ("true_depth_ppm", "fit_depth_ppm", "Depth (ppm)"),
        ("true_duration_hours", "fit_duration_days", "Duration (hours)"),
    ]
    rows = []
    for true_col, pred_col, label in candidates:
        if true_col not in catalog.columns or pred_col not in catalog.columns:
            continue
        true = pd.to_numeric(catalog[true_col], errors="coerce")
        pred = pd.to_numeric(catalog[pred_col], errors="coerce")
        if label.startswith("Duration"):
            pred = pred * 24.0
        keep = true.notna() & pred.notna()
        if keep.sum() == 0:
            continue
        rows.append((true[keep].to_numpy(), pred[keep].to_numpy(), label))
    if not rows:
        return
    fig, axes = plt.subplots(1, len(rows), figsize=(5 * len(rows), 4))
    if len(rows) == 1:
        axes = [axes]
    for ax, (true, pred, label) in zip(axes, rows):
        ax.scatter(true, pred, s=22, alpha=0.7)
        lo = np.nanmin([true.min(), pred.min()])
        hi = np.nanmax([true.max(), pred.max()])
        ax.plot([lo, hi], [lo, hi], linestyle="--")
        ax.set_xlabel(f"True {label}")
        ax.set_ylabel(f"Recovered {label}")
        ax.set_title(label)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_reliability_diagram(report_dict: dict, output_path: str | Path) -> None:
    import matplotlib.pyplot as plt
    rel = report_dict.get("reliability_metrics", {})
    if not rel.get("available"):
        return
    bins = rel.get("bins", [])
    if not bins:
        return
    x = [b["mean_confidence"] for b in bins]
    y = [b["accuracy"] for b in bins]
    n = [b["n"] for b in bins]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.scatter(x, y, s=[max(30, 10 * nn) for nn in n], alpha=0.8)
    for xi, yi, ni in zip(x, y, n):
        ax.text(xi, yi, str(ni), fontsize=8, ha="center", va="bottom")
    ax.set_xlabel("Mean reported confidence")
    ax.set_ylabel("Empirical accuracy")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Confidence reliability")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_injection_recovery_heatmap(catalog: pd.DataFrame, output_path: str | Path) -> None:
    import matplotlib.pyplot as plt
    if not {"true_depth_ppm", "noise_ppm", "detected"}.issubset(catalog.columns):
        return
    df = catalog.copy()
    df["depth_bin"] = pd.cut(pd.to_numeric(df["true_depth_ppm"], errors="coerce"), bins=5)
    df["noise_bin"] = pd.cut(pd.to_numeric(df["noise_ppm"], errors="coerce"), bins=5)
    pivot = df.pivot_table(values="detected", index="noise_bin", columns="depth_bin", aggfunc="mean", observed=False)
    if pivot.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", origin="lower", vmin=0, vmax=1)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([str(c) for c in pivot.columns], rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([str(i) for i in pivot.index], fontsize=8)
    ax.set_xlabel("Injected depth ppm")
    ax.set_ylabel("Noise ppm")
    ax.set_title("Injection recovery rate")
    fig.colorbar(im, ax=ax, label="Detection fraction")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
