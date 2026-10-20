#!/usr/bin/env python3
"""Check that the selected CICIDS-style CSV files can be loaded."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from online_incremental_ids.dataset import load_dataset  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate dataset files and print their labels/features.")
    parser.add_argument("--data", default="data/raw", help="CSV file or directory of extracted CSV files.")
    parser.add_argument("--rows-per-file", type=int, default=0, help="Optional preview limit per file; 0 reads all rows.")
    args = parser.parse_args()
    data, metadata = load_dataset(args.data, rows_per_file=args.rows_per_file)
    labels = data["__target_label__"].astype(str).value_counts()
    print("\nDataset check passed.")
    print(f"Rows: {len(data):,}")
    print(f"Usable numeric features: {len(metadata['feature_columns'])}")
    print(f"Timestamp coverage: {metadata['timestamp_valid_fraction']:.1%}")
    print(f"Ordering: {metadata['sort_method']}")
    print("\nLabel counts:")
    print(labels.to_string())
    if metadata["files_skipped"]:
        print("\nSkipped CSV files:")
        for item in metadata["files_skipped"]:
            print(f"- {Path(item['file']).name}: {item['reason']}")


if __name__ == "__main__":
    main()
