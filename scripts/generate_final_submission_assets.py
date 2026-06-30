#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path
import argparse
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from exoplanet_pipeline.final_outputs import generate_submission_package_outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate final plots, candidate review table, and 3-page report draft from a candidate catalog.")
    parser.add_argument("catalog_csv", type=str)
    parser.add_argument("--output-dir", type=str, default="submission_assets")
    parser.add_argument("--validation-report", type=str, default=None)
    args = parser.parse_args()
    catalog = pd.read_csv(args.catalog_csv)
    paths = generate_submission_package_outputs(catalog, args.output_dir, validation_report_path=args.validation_report)
    print("Generated submission assets:")
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
