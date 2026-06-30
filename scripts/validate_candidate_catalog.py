#!/usr/bin/env python3
"""Validate a labeled candidate catalog from the organizer/curated dataset."""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from exoplanet_pipeline.validation import validate_candidate_catalog, validation_report_to_markdown
from exoplanet_pipeline.validation_diagnostics import (
    plot_confusion_matrix_from_report,
    plot_parameter_recovery,
    plot_reliability_diagram,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", required=True, help="CSV file containing predictions and labels")
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--pred-col", default=None)
    ap.add_argument("--out-dir", default="outputs_validation")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.catalog)
    report = validate_candidate_catalog(df, label_col=args.label_col, pred_col=args.pred_col)
    report.save_json(out_dir / "validation_report.json")
    (out_dir / "validation_report.md").write_text(validation_report_to_markdown(report), encoding="utf-8")
    rd = report.to_dict()
    plot_confusion_matrix_from_report(rd, out_dir / "confusion_matrix.png")
    plot_parameter_recovery(df, out_dir / "parameter_recovery.png")
    plot_reliability_diagram(rd, out_dir / "reliability_diagram.png")
    print("Validation report written to", out_dir)


if __name__ == "__main__":
    main()
