#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from exoplanet_pipeline.public_data import read_public_metadata, supported_public_sources, write_public_metadata_artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize public Kepler/TESS metadata for CNN vetter dataset construction.")
    parser.add_argument("metadata_csv", help="Local CSV/parquet metadata file from a supported public source")
    parser.add_argument("--source", required=True, choices=supported_public_sources())
    parser.add_argument("--output-dir", default="data/public", help="Ignored output directory for normalized metadata/manifests")
    args = parser.parse_args()

    df = read_public_metadata(args.metadata_csv, source=args.source)
    paths = write_public_metadata_artifacts(df, args.output_dir, source=args.source)
    print(f"Rows: {len(df)}")
    print("Canonical labels:", df["canonical_label"].fillna("UNLABELED").value_counts().to_dict())
    print("Binary labels:", df["binary_label"].fillna("UNLABELED").value_counts().to_dict())
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
