#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from exoplanet_pipeline.public_data import read_tic_ctl_catalog, write_tic_ctl_target_list


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a pipeline target list from STScI TIC/CTL catalog CSV files.")
    parser.add_argument("catalog_csv", help="Local TIC/CTL CSV or TIC .csv.gz chunk from STScI")
    parser.add_argument("--catalog-type", default="auto", choices=["auto", "ctl", "tic", "xctl"])
    parser.add_argument("--header-file", default=None, help="Optional STScI header file for full xCTL/TIC reads")
    parser.add_argument("--nrows", type=int, default=None, help="Read only the first N catalog rows")
    parser.add_argument("--max-targets", type=int, default=None, help="Write only the first/sorted N targets")
    parser.add_argument("--output-csv", default="data/public/tic_ctl_targets.csv")
    parser.add_argument("--full-tic", action="store_true", help="Read all TIC columns; requires --header-file and much more memory")
    args = parser.parse_args()

    targets = read_tic_ctl_catalog(
        args.catalog_csv,
        catalog_type=args.catalog_type,
        nrows=args.nrows,
        minimal_tic=not args.full_tic,
        header_file=args.header_file,
    )
    path = write_tic_ctl_target_list(targets, args.output_csv, max_targets=args.max_targets)
    print(f"targets: {path}")
    print(f"rows_written: {sum(1 for _ in open(path, encoding='utf-8')) - 1}")


if __name__ == "__main__":
    main()
