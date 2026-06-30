#!/usr/bin/env python3
"""Run Parts 7–8 synthetic injection-recovery validation.

This script proves that the pipeline can produce not only detections/classes, but
also uncertainty estimates and validation metrics before the organizer's curated
labeled dataset is available.
"""
from __future__ import annotations

import json
from pathlib import Path

from exoplanet_pipeline.config import PipelineConfig
from exoplanet_pipeline.injection_recovery import run_injection_recovery_grid, summarize_injection_recovery, compact_injection_demo_grid
from exoplanet_pipeline.validation import validate_candidate_catalog, validation_report_to_markdown
from exoplanet_pipeline.validation_diagnostics import (
    plot_confusion_matrix_from_report,
    plot_parameter_recovery,
    plot_reliability_diagram,
    plot_injection_recovery_heatmap,
)


def main() -> None:
    out_dir = Path("outputs_parts_7_8")
    out_dir.mkdir(parents=True, exist_ok=True)

    config = PipelineConfig(
        detection_method="bls",
        n_periods=300,
        n_durations=5,
        min_clean_points=300,
        detection_use_variants=False,
        make_plots=False,
    )
    # Use a compact class-balanced grid for a fast demo. Increase this in the final run.
    specs = compact_injection_demo_grid(random_seed=42, n_per_class=2)
    catalog = run_injection_recovery_grid(specs=specs, config=config, n_bootstrap=50)
    catalog.to_csv(out_dir / "parts_7_8_injection_recovery_catalog.csv", index=False)

    summary = summarize_injection_recovery(catalog)
    with open(out_dir / "parts_7_8_injection_recovery_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    report = validate_candidate_catalog(catalog, label_col="label", pred_col="final_predicted_class")
    report.save_json(out_dir / "parts_7_8_validation_report.json")
    (out_dir / "parts_7_8_validation_report.md").write_text(validation_report_to_markdown(report), encoding="utf-8")

    report_dict = report.to_dict()
    plot_confusion_matrix_from_report(report_dict, out_dir / "parts_7_8_confusion_matrix.png")
    plot_parameter_recovery(catalog, out_dir / "parts_7_8_parameter_recovery.png")
    plot_reliability_diagram(report_dict, out_dir / "parts_7_8_reliability_diagram.png")
    plot_injection_recovery_heatmap(catalog, out_dir / "parts_7_8_injection_heatmap.png")

    print("Wrote Parts 7–8 validation outputs to", out_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
