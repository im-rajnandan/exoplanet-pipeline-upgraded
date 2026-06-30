#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from exoplanet_pipeline.public_data import download_exoplanet_archive_metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Download official public metadata from NASA Exoplanet Archive TAP.")
    parser.add_argument("--source", required=True, choices=["tess-toi", "kepler-dr25"])
    parser.add_argument("--output-dir", default="data/public")
    parser.add_argument("--top", type=int, default=None, help="Optional TOP N for smoke tests or small notebook runs")
    parser.add_argument("--where", default=None, help="Optional ADQL WHERE clause, for example: tfopwg_disp is not null")
    args = parser.parse_args()

    paths = download_exoplanet_archive_metadata(
        args.source,
        output_dir=args.output_dir,
        top=args.top,
        where=args.where,
    )
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
