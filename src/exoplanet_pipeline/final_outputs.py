from __future__ import annotations

from pathlib import Path
import json
import textwrap
import numpy as np
import pandas as pd

from .final_catalog import (
    harmonize_candidate_catalog,
    summarize_final_catalog,
    validate_final_catalog_schema,
    PLANET_CLASS,
    EB_CLASS,
    BLEND_CLASS,
    UNCERTAIN_CLASS,
)


def _format_markdown_cell(value) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        text = f"{value:.6g}"
    else:
        text = str(value)
    return text.replace("\n", "<br>").replace("|", "\\|")


def _dataframe_to_markdown_table(df: pd.DataFrame) -> str:
    """Small dependency-free markdown table writer.

    pandas.DataFrame.to_markdown requires the optional tabulate package. The
    submission writer should work from the declared core dependencies alone.
    """
    if df.empty:
        return ""
    columns = [str(c) for c in df.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(_format_markdown_cell(row[c]) for c in df.columns) + " |")
    return "\n".join(lines)


def plot_final_class_distribution(catalog: pd.DataFrame, output_path: str | Path) -> None:
    import matplotlib.pyplot as plt
    df = harmonize_candidate_catalog(catalog) if "final_science_class" not in catalog.columns else catalog.copy()
    if df.empty or "final_science_class" not in df:
        return
    counts = df["final_science_class"].value_counts()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(range(len(counts)), counts.values)
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels(counts.index, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Number of candidates")
    ax.set_title("Final candidate class distribution")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_final_confidence_distribution(catalog: pd.DataFrame, output_path: str | Path) -> None:
    import matplotlib.pyplot as plt
    df = harmonize_candidate_catalog(catalog) if "final_science_class" not in catalog.columns else catalog.copy()
    if df.empty or "final_science_confidence" not in df:
        return
    conf = pd.to_numeric(df["final_science_confidence"], errors="coerce").dropna()
    if conf.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(conf, bins=np.linspace(0, 1, 11), edgecolor="black")
    ax.set_xlabel("Final confidence")
    ax.set_ylabel("Number of candidates")
    ax.set_title("Candidate confidence distribution")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_priority_scatter(catalog: pd.DataFrame, output_path: str | Path) -> None:
    import matplotlib.pyplot as plt
    df = harmonize_candidate_catalog(catalog) if "science_priority_score" not in catalog.columns else catalog.copy()
    if df.empty:
        return
    x = pd.to_numeric(df.get("period_days"), errors="coerce")
    y = pd.to_numeric(df.get("depth_ppm"), errors="coerce")
    s = pd.to_numeric(df.get("science_priority_score"), errors="coerce").fillna(0)
    conf = pd.to_numeric(df.get("final_science_confidence"), errors="coerce").fillna(0.0)
    keep = x.notna() & y.notna()
    if keep.sum() == 0:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    scatter = ax.scatter(x[keep], y[keep], s=np.clip(s[keep] - s[keep].min() + 20, 20, 200), c=conf[keep], alpha=0.75)
    ax.set_xlabel("Period (days)")
    ax.set_ylabel("Depth (ppm)")
    ax.set_title("Candidate period-depth-priority map")
    fig.colorbar(scatter, ax=ax, label="Final confidence")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def make_final_visual_summary(catalog: pd.DataFrame, output_dir: str | Path) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "class_distribution": output_dir / "final_class_distribution.png",
        "confidence_distribution": output_dir / "final_confidence_distribution.png",
        "priority_scatter": output_dir / "final_period_depth_priority.png",
    }
    if catalog is None or catalog.empty:
        return paths
    plot_final_class_distribution(catalog, paths["class_distribution"])
    plot_final_confidence_distribution(catalog, paths["confidence_distribution"])
    plot_priority_scatter(catalog, paths["priority_scatter"])
    return paths


def write_final_candidate_review_markdown(catalog: pd.DataFrame, output_path: str | Path, top_n: int = 20) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = harmonize_candidate_catalog(catalog) if "science_priority_rank" not in catalog.columns else catalog.copy()
    summary = summarize_final_catalog(df)
    lines: list[str] = []
    lines.append("# Final Candidate Review Table")
    lines.append("")
    lines.append(f"Total candidates: **{summary.get('n_candidates', 0)}**")
    lines.append(f"Unique targets: **{summary.get('n_unique_targets', 'NA')}**")
    lines.append("")
    lines.append("## Class counts")
    lines.append("")
    for k, v in summary.get("class_counts", {}).items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append(f"## Top {top_n} candidates by review priority")
    lines.append("")
    cols = [c for c in ["science_priority_rank", "tic_id", "sector", "candidate_id", "final_science_class", "final_science_confidence", "period_days", "duration_hours", "depth_ppm", "effective_snr", "recommended_action"] if c in df]
    if cols and not df.empty:
        lines.append(_dataframe_to_markdown_table(df.head(top_n)[cols]))
    else:
        lines.append("No candidate rows available.")
    lines.append("")
    lines.append("## Interpretation note")
    lines.append("")
    lines.append("The priority rank is not an astronomical confirmation. It is a triage score for human review that combines detection strength, parameter stability, classification confidence, data quality, and false-positive risk indicators such as secondary eclipses, odd/even mismatch, centroid motion, and crowding.")
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def generate_three_page_report_markdown(
    output_path: str | Path,
    catalog: pd.DataFrame | None = None,
    validation_report: dict | None = None,
    project_title: str = "AI-enabled Detection of Exoplanets from Noisy TESS Light Curves",
) -> Path:
    """Write a concise submission-ready report draft in Markdown.

    The content is intentionally compact enough to be adapted to the required
    three-page limit. It states assumptions and uncertainty handling honestly.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary = summarize_final_catalog(harmonize_candidate_catalog(catalog)) if catalog is not None and not catalog.empty else {"n_candidates": 0, "class_counts": {}}
    val = validation_report or {}
    det = val.get("detection_metrics", {}) if isinstance(val, dict) else {}
    cls = val.get("classification_metrics", {}) if isinstance(val, dict) else {}
    par = val.get("parameter_metrics", {}) if isinstance(val, dict) else {}

    text = f"""
# {project_title}

## 1. Methodology
We implemented a hybrid physics-informed and AI-driven pipeline for TESS light curves. The pipeline first ingests SAP/PDCSAP light curves, applies TESS quality masking, selects the safest flux source, normalizes the flux, performs conservative detrending, and records quality-control metrics such as cadence, baseline, noise, gap fraction, CROWDSAP, FLFRCSAP, and centroid availability. Synthetic fallback is disabled for real science data so failed downloads cannot become false successes.

Periodic transit-like dips are detected with a BLS/TLS-style candidate search. Each candidate is represented by period, epoch, duration, depth, SNR, periodogram strength, number of observed transits, and detection status. Candidate detection is deliberately separated from classification: the detector only answers whether a periodic dip exists, while later vetting decides whether it resembles a planet, eclipsing binary, blend, stellar variability, systematic, or uncertain signal.

For each candidate we refine period, epoch, duration, and depth using a local box-profile fit and event-by-event depth estimates. We then extract physically motivated vetting features: odd/even depth mismatch, secondary-eclipse strength at phase 0.5, centroid shift significance, crowding/dilution risk, V-shape score, harmonic risk, red-noise proxy, and data-quality score. These features feed both a transparent rule-based classifier and an optional supervised classifier trained on curated labeled candidate catalogs.

## 2. Classification and uncertainty
The classifier outputs probabilities for planetary transit candidate, eclipsing binary, blend/contaminated signal, stellar variability, instrumental/systematic, no significant signal, and uncertain transit-like signal. Physical guardrails prevent the AI model from overcalling planet candidates when strong secondary eclipses, odd/even mismatch, centroid shifts, or poor data quality are present.

Signal significance is reported mainly through local transit SNR, periodogram strength, and effective SNR after red-noise inflation. Parameter uncertainties combine local photometric scatter, residual bootstrap depth uncertainty, event-to-event depth scatter, ephemeris-grid curvature, and multi-detrender stability. Final confidence is a weighted combination of detection confidence, parameter confidence, and classification confidence; it is a triage confidence, not a formal astronomical validation probability.

## 3. Validation and outputs
The validation layer supports curated labeled data and synthetic injection-recovery experiments. Detection is evaluated with precision, recall, specificity, and F1. Classification is evaluated with accuracy, balanced accuracy, macro F1, weighted F1, and a confusion matrix. Parameter recovery is measured through absolute and relative errors in period, depth, and duration. Confidence calibration is checked through reliability bins comparing reported confidence with empirical correctness.

Current candidate-catalog summary: **{summary.get('n_candidates', 0)} candidates** across **{summary.get('n_unique_targets', 'NA')} targets**. Class counts: `{json.dumps(summary.get('class_counts', {}))}`.

Validation snapshot, when labels are available: detection metrics `{json.dumps(det)[:700]}`; classification metrics `{json.dumps({k: cls.get(k) for k in ['accuracy','balanced_accuracy','macro_f1','weighted_f1'] if isinstance(cls, dict) and k in cls})}`. Parameter metrics are stored for period, depth, and duration recovery when ground truth columns are available.

## Assumptions and limitations
We assume that periodic dips are approximately stable over the observed baseline and that detrending windows are longer than the transit duration. Transit depth may be biased by dilution in crowded apertures; CROWDSAP is therefore treated as a risk feature rather than an automatic rejection. Planet radius is only physically meaningful when reliable stellar radius is available. Crowded-field and high-value candidates should receive follow-up inspection using target-pixel difference imaging and nearby-source checks before being treated as validated planets.
"""
    output_path.write_text(textwrap.dedent(text).strip() + "\n", encoding="utf-8")
    return output_path


def generate_submission_package_outputs(
    catalog: pd.DataFrame,
    output_dir: str | Path,
    validation_report_path: str | Path | None = None,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    final_catalog = harmonize_candidate_catalog(catalog)
    issues = validate_final_catalog_schema(final_catalog)
    if issues:
        raise ValueError("Final catalog schema validation failed: " + "; ".join(issues))
    final_catalog_path = output_dir / "submission_final_candidate_catalog.csv"
    final_catalog.to_csv(final_catalog_path, index=False)
    visual_paths = make_final_visual_summary(final_catalog, output_dir)
    review_path = write_final_candidate_review_markdown(final_catalog, output_dir / "submission_candidate_review.md")
    validation_report = None
    if validation_report_path is not None and Path(validation_report_path).exists():
        with open(validation_report_path, "r", encoding="utf-8") as f:
            validation_report = json.load(f)
    report_path = generate_three_page_report_markdown(output_dir / "submission_three_page_report_draft.md", final_catalog, validation_report)
    summary_path = output_dir / "submission_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summarize_final_catalog(final_catalog), f, indent=2)
    return {
        "final_catalog": final_catalog_path,
        "candidate_review": review_path,
        "three_page_report_draft": report_path,
        "summary_json": summary_path,
        **visual_paths,
    }
