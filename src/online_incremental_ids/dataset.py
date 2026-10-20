"""Load CICIDS-style CSV files while keeping only usable numeric features."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd

LABEL_NAMES = {"label", "class", "target", "attack", "attack type", "category"}
TIME_NAMES = {"timestamp", "time", "datetime", "date time", "date/time"}
IDENTIFIER_NAMES = {
    "flow id",
    "source ip",
    "destination ip",
    "src ip",
    "dst ip",
    "source address",
    "destination address",
}


def clean_column_name(value: Any) -> str:
    """Normalize whitespace without changing the feature's meaning."""
    return re.sub(r"\s+", " ", str(value).strip())


def _normalized_name(value: Any) -> str:
    return clean_column_name(value).lower()


def _find_column(columns: list[Any], candidates: set[str]) -> Any | None:
    for column in columns:
        if _normalized_name(column) in candidates:
            return column
    return None


def _read_one_csv(path: Path, rows_per_file: int = 0) -> tuple[pd.DataFrame | None, str | None]:
    """Read one CSV and return internal target/time columns plus numeric features."""
    try:
        header = list(pd.read_csv(path, nrows=0, encoding_errors="replace").columns)
    except Exception as exc:  # clear per-file diagnostic; other files may still be usable
        return None, f"Could not read header ({type(exc).__name__}: {exc})"

    label_column = _find_column(header, LABEL_NAMES)
    if label_column is None:
        return None, "Skipped: no recognized target column (expected Label/Class/Target)."

    timestamp_column = _find_column(header, TIME_NAMES)
    excluded = IDENTIFIER_NAMES | TIME_NAMES
    selected = []
    for column in header:
        normalized = _normalized_name(column)
        if column == label_column or (timestamp_column is not None and column == timestamp_column):
            selected.append(column)
        elif normalized in excluded or normalized in LABEL_NAMES:
            continue
        elif normalized.startswith("unnamed:"):
            continue
        else:
            selected.append(column)

    read_kwargs: dict[str, Any] = {
        "usecols": selected,
        "low_memory": False,
        "encoding_errors": "replace",
    }
    if rows_per_file > 0:
        read_kwargs["nrows"] = rows_per_file
    try:
        frame = pd.read_csv(path, **read_kwargs)
    except Exception as exc:
        return None, f"Could not read CSV ({type(exc).__name__}: {exc})"

    # CICIDS2017 often has leading/trailing whitespace in column headings.
    original_columns = list(frame.columns)
    clean_columns = [clean_column_name(column) for column in original_columns]
    frame.columns = clean_columns
    label_name = clean_column_name(label_column)
    timestamp_name = clean_column_name(timestamp_column) if timestamp_column is not None else None

    if label_name not in frame.columns:
        return None, "Skipped: label column could not be identified after header normalization."

    labels = frame[label_name].astype("string").str.strip().str.replace(r"\s+", " ", regex=True)
    labels = labels.replace("", pd.NA)
    timestamps = (
        pd.to_datetime(frame[timestamp_name], errors="coerce", dayfirst=True)
        if timestamp_name and timestamp_name in frame.columns
        else pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns]")
    )

    feature_frames: dict[str, pd.Series] = {}
    for column in frame.columns:
        if column == label_name or (timestamp_name and column == timestamp_name):
            continue
        if _normalized_name(column) in IDENTIFIER_NAMES | TIME_NAMES | LABEL_NAMES:
            continue
        if _normalized_name(column).startswith("unnamed:"):
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        # Completely nonnumeric columns (for example HTTP request strings) are not model features.
        if not values.notna().any():
            continue
        values = values.replace([np.inf, -np.inf], np.nan).astype("float32")
        feature_frames[column] = values

    if not feature_frames:
        return None, "Skipped: no numeric traffic features were found."

    result = pd.DataFrame(feature_frames, index=frame.index)
    result["__target_label__"] = labels
    result["__timestamp__"] = timestamps
    result = result.loc[result["__target_label__"].notna()].copy()
    if result.empty:
        return None, "Skipped: no rows with a valid target label."
    return result, None


def load_dataset(data_path: str | Path, rows_per_file: int = 0) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load a CSV file or all CSV files in a directory, sorted by timestamp if available.

    Args:
        data_path: A CSV file or a directory containing extracted dataset CSVs.
        rows_per_file: Optional quick-run limit, taking the first N rows from each CSV.

    Returns:
        (dataframe, metadata). Internal columns are ``__target_label__`` and
        ``__timestamp__``; all remaining columns are numeric model features.
    """
    path = Path(data_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset path does not exist: {path}\n"
            "Put the extracted CICIDS2017 CSV files in data/raw, or pass --data PATH."
        )

    if path.is_file():
        if path.suffix.lower() != ".csv":
            raise ValueError(f"Expected a CSV file or directory of CSV files, got: {path}")
        csv_files = [path]
    else:
        csv_files = sorted(p for p in path.rglob("*.csv") if p.is_file())

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found under: {path}\n"
            "Extract the CICIDS2017 ZIP file first; this project does not read ZIP archives directly."
        )

    frames: list[pd.DataFrame] = []
    loaded_files: list[str] = []
    skipped_files: list[dict[str, str]] = []
    for csv_path in csv_files:
        frame, error = _read_one_csv(csv_path, rows_per_file=rows_per_file)
        if frame is None:
            skipped_files.append({"file": str(csv_path), "reason": error or "Unknown reason"})
            continue
        frames.append(frame)
        loaded_files.append(str(csv_path))
        print(f"Loaded {len(frame):,} labelled rows from {csv_path.name}")

    if not frames:
        reasons = "\n".join(f"- {item['file']}: {item['reason']}" for item in skipped_files[:10])
        raise ValueError(
            "No usable labelled CSV files were found. Check that the CICIDS2017 flow CSV files "
            f"are extracted and contain a Label column.\n{reasons}"
        )

    # Concatenate only after identifiers/text columns have been removed. Numeric features are float32.
    data = pd.concat(frames, axis=0, ignore_index=True, sort=False)
    data["__target_label__"] = data["__target_label__"].astype("string").str.strip()
    data["__timestamp__"] = pd.to_datetime(data["__timestamp__"], errors="coerce")
    data = data.replace([np.inf, -np.inf], np.nan)

    feature_columns = [
        column for column in data.columns if column not in {"__target_label__", "__timestamp__"}
    ]
    # Drop columns that are completely empty across all supplied files.
    feature_columns = [column for column in feature_columns if data[column].notna().any()]
    if not feature_columns:
        raise ValueError("No usable numeric features remain after cleaning the supplied CSV files.")
    data = data[feature_columns + ["__target_label__", "__timestamp__"]]

    timestamp_valid_fraction = float(data["__timestamp__"].notna().mean())
    if timestamp_valid_fraction > 0:
        # Stable sorting preserves original file/row order when timestamps tie or are missing.
        data = data.sort_values("__timestamp__", kind="stable", na_position="last").reset_index(drop=True)
        sort_method = "timestamp (ascending; unparseable timestamps placed last)"
    else:
        data = data.reset_index(drop=True)
        sort_method = "CSV/file row order (no usable timestamp column)"
        print("WARNING: no usable timestamps found; evaluation will follow CSV/file row order.")

    # Drop rows with no values at all for any model feature. Other missing cells are imputed from warm-up data.
    data = data.loc[data[feature_columns].notna().any(axis=1)].reset_index(drop=True)
    data["__target_label__"] = data["__target_label__"].astype("category")

    metadata = {
        "data_path": str(path),
        "files_found": len(csv_files),
        "files_loaded": loaded_files,
        "files_skipped": skipped_files,
        "rows": int(len(data)),
        "feature_columns": feature_columns,
        "timestamp_valid_fraction": timestamp_valid_fraction,
        "sort_method": sort_method,
        "rows_per_file_limit": int(rows_per_file),
    }
    return data, metadata
