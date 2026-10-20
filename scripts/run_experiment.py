#!/usr/bin/env python3
"""Entry point for the online incremental IDS experiment."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from online_incremental_ids.experiment import run_experiment  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare a frozen static IDS model against online incremental learning on chronological traffic batches."
    )
    parser.add_argument("--data", default="data/raw", help="CSV file or directory of extracted CICIDS2017 CSV files (default: data/raw).")
    parser.add_argument("--output", default="outputs/latest", help="Directory for metrics, plots, and saved models (default: outputs/latest).")
    parser.add_argument("--task", choices=["binary", "multiclass"], default="binary", help="Binary benign-vs-attack detection or multiclass attack-type classification.")
    parser.add_argument("--initial-fraction", type=float, default=0.30, help="Fraction of earliest chronological rows used for initial training (0.05-0.80; default 0.30).")
    parser.add_argument("--batch-size", type=int, default=50_000, help="Number of traffic rows per evaluation/update batch (default: 50000).")
    parser.add_argument("--initial-epochs", type=int, default=5, help="Warm-up passes for both models (default: 5).")
    parser.add_argument("--rows-per-file", type=int, default=0, help="Optional quick-run limit: read only the first N rows of each CSV; 0 means all rows.")
    parser.add_argument("--no-balance", action="store_true", help="Disable inverse-frequency sample weighting based only on warm-up labels.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        run_experiment(
            data_path=args.data,
            output_dir=args.output,
            task=args.task,
            initial_fraction=args.initial_fraction,
            batch_size=args.batch_size,
            initial_epochs=args.initial_epochs,
            rows_per_file=args.rows_per_file,
            balance=not args.no_balance,
            random_seed=args.seed,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
