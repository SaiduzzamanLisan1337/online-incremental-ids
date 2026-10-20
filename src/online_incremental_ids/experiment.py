"""Run a time-aware static-vs-online intrusion-detection experiment."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import json
import time
from typing import Any

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

from .dataset import load_dataset

BENIGN_LABELS = {"BENIGN", "NORMAL", "NORMAL TRAFFIC", "0"}


def _encode_targets(labels: pd.Series, task: str) -> tuple[np.ndarray, list[str], dict[str, int]]:
    label_values = labels.astype("string").fillna("UNKNOWN").str.strip()
    if task == "binary":
        normalized = label_values.str.upper()
        y = (~normalized.isin(BENIGN_LABELS)).astype("int32").to_numpy()
        mapping = {"BENIGN": 0, "ATTACK": 1}
        return y, ["BENIGN", "ATTACK"], mapping

    encoder = LabelEncoder()
    y = encoder.fit_transform(label_values.astype(str)).astype("int32")
    mapping = {str(label): int(index) for index, label in enumerate(encoder.classes_)}
    return y, [str(label) for label in encoder.classes_], mapping


def _metrics(y_true: np.ndarray, y_pred: np.ndarray, task: str) -> dict[str, float]:
    if task == "binary":
        precision = precision_score(y_true, y_pred, pos_label=1, average="binary", zero_division=0)
        recall = recall_score(y_true, y_pred, pos_label=1, average="binary", zero_division=0)
        f1 = f1_score(y_true, y_pred, pos_label=1, average="binary", zero_division=0)
        macro_f1 = f1_score(y_true, y_pred, labels=[0, 1], average="macro", zero_division=0)
    else:
        precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
        recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "macro_f1": float(macro_f1),
    }


def _sample_weights(y: np.ndarray, classes: np.ndarray, balance: bool) -> tuple[np.ndarray, dict[int, float]]:
    if not balance:
        return np.ones(len(y), dtype=np.float32), {int(c): 1.0 for c in classes}
    unique, counts = np.unique(y, return_counts=True)
    n_present = max(1, len(unique))
    count_by_class = {int(c): int(n) for c, n in zip(unique, counts)}
    weights = {
        int(c): min(20.0, len(y) / (n_present * count_by_class[int(c)]))
        if int(c) in count_by_class else 1.0
        for c in classes
    }
    row_weights = np.asarray([weights.get(int(label), 1.0) for label in y], dtype=np.float32)
    return row_weights, weights


def _new_classifier(seed: int) -> SGDClassifier:
    # log_loss provides probabilities if users later extend the project; SGDClassifier supports partial_fit.
    return SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=0.0001,
        random_state=np.random.RandomState(seed),
        shuffle=True,
        tol=None,
        average=True,
    )


def _as_numeric_frame(data: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    numeric = data.loc[:, columns].copy()
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    for column in numeric.columns:
        numeric[column] = pd.to_numeric(numeric[column], errors="coerce").astype("float32")
    return numeric


def _make_plots(batch_df: pd.DataFrame, output_dir: Path) -> list[str]:
    generated = []
    if batch_df.empty:
        return generated

    for metric, title, ylabel, filename in [
        ("f1", "Static vs Online F1 by Traffic Batch", "F1 score", "f1_over_time.png"),
        ("recall", "Static vs Online Recall by Traffic Batch", "Recall", "recall_over_time.png"),
        ("accuracy", "Static vs Online Accuracy by Traffic Batch", "Accuracy", "accuracy_over_time.png"),
    ]:
        fig, ax = plt.subplots(figsize=(11, 5.5))
        for model_name in ["static", "online"]:
            subset = batch_df[batch_df["model"] == model_name]
            ax.plot(subset["batch"], subset[metric], marker="o", markersize=3, linewidth=1.5, label=model_name.title())
        ax.set_title(title)
        ax.set_xlabel("Chronological traffic batch")
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, 1.02)
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        target = output_dir / filename
        fig.savefig(target, dpi=160)
        plt.close(fig)
        generated.append(str(target))
    return generated


def run_experiment(
    data_path: str | Path,
    output_dir: str | Path,
    task: str = "binary",
    initial_fraction: float = 0.30,
    batch_size: int = 50_000,
    initial_epochs: int = 5,
    rows_per_file: int = 0,
    balance: bool = True,
    random_seed: int = 42,
) -> dict[str, Any]:
    if not 0.05 <= initial_fraction <= 0.80:
        raise ValueError("--initial-fraction must be between 0.05 and 0.80.")
    if batch_size < 1:
        raise ValueError("--batch-size must be a positive integer.")
    if initial_epochs < 1:
        raise ValueError("--initial-epochs must be at least 1.")

    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    print("\n[1/5] Loading dataset and sorting rows by timestamp...")
    data, data_info = load_dataset(data_path, rows_per_file=rows_per_file)
    if len(data) < 100:
        raise ValueError(f"Only {len(data)} usable rows were found. At least 100 rows are required.")

    feature_columns = list(data_info["feature_columns"])
    y, class_names, label_mapping = _encode_targets(data["__target_label__"], task)
    n_rows = len(data)
    start_index = min(max(int(n_rows * initial_fraction), 2), n_rows - 1)
    max_start = min(int(n_rows * 0.80), n_rows - 1)
    classes = np.arange(len(class_names), dtype=np.int32) if task == "multiclass" else np.array([0, 1], dtype=np.int32)

    # Keep the initial period chronological. If it contains only one target class, extend it gradually.
    while len(np.unique(y[:start_index])) < 2 and start_index < max_start:
        start_index = min(max_start, start_index + max(1, int(n_rows * 0.05)))
    if len(np.unique(y[:start_index])) < 2:
        raise ValueError(
            "The chronological warm-up contains only one target class. Increase --initial-fraction, "
            "or check whether your CSV files contain both benign and attack records."
        )
    if start_index >= n_rows:
        raise ValueError("No stream rows remain after the initial warm-up. Reduce --initial-fraction.")

    warmup_fraction_actual = start_index / n_rows
    print(f"Loaded {n_rows:,} rows and {len(feature_columns)} candidate numeric features.")
    print(f"Task: {task}; class labels: {class_names}")
    print(f"Initial chronological warm-up: {start_index:,} rows ({warmup_fraction_actual:.1%}); stream: {n_rows - start_index:,} rows.")
    if data_info["timestamp_valid_fraction"] < 0.95:
        print(f"WARNING: only {data_info['timestamp_valid_fraction']:.1%} of rows have parseable timestamps.")

    # Fit preprocessing on the warm-up only; never update the scaler using future batches.
    print("\n[2/5] Fitting warm-up-only imputer/scaler...")
    initial_features = _as_numeric_frame(data.iloc[:start_index], feature_columns)
    usable_features = [column for column in initial_features.columns if initial_features[column].notna().any()]
    if not usable_features:
        raise ValueError("All candidate features are missing in the initial warm-up.")
    feature_columns = usable_features
    initial_raw = initial_features[feature_columns].to_numpy(dtype=np.float32, copy=True)
    del initial_features
    imputer = SimpleImputer(strategy="median")
    x_initial_imputed = imputer.fit_transform(initial_raw).astype(np.float32, copy=False)
    scaler = StandardScaler()
    x_initial = scaler.fit_transform(x_initial_imputed).astype(np.float32, copy=False)
    y_initial = y[:start_index]
    initial_sample_weights, class_weight_map = _sample_weights(y_initial, classes, balance)

    print("\n[3/5] Initializing two equivalent SGD classifiers...")
    static_model = _new_classifier(random_seed)
    online_model = _new_classifier(random_seed)
    init_start = time.perf_counter()
    # Identical warm-up data, class space, epoch count, and algorithm make the frozen baseline comparable.
    for epoch in range(initial_epochs):
        if epoch == 0:
            static_model.partial_fit(x_initial, y_initial, classes=classes, sample_weight=initial_sample_weights)
            online_model.partial_fit(x_initial, y_initial, classes=classes, sample_weight=initial_sample_weights)
        else:
            static_model.partial_fit(x_initial, y_initial, sample_weight=initial_sample_weights)
            online_model.partial_fit(x_initial, y_initial, sample_weight=initial_sample_weights)
    initial_train_seconds = time.perf_counter() - init_start
    del initial_raw, x_initial_imputed, x_initial, initial_sample_weights

    print("\n[4/5] Predicting each future batch before updating the online model...")
    batch_records: list[dict[str, Any]] = []
    y_true_all: list[np.ndarray] = []
    static_predictions: list[np.ndarray] = []
    online_predictions: list[np.ndarray] = []
    online_update_total = 0.0
    static_prediction_total = 0.0
    online_prediction_total = 0.0
    batch_id = 0
    for batch_start in range(start_index, n_rows, batch_size):
        batch_end = min(batch_start + batch_size, n_rows)
        batch = data.iloc[batch_start:batch_end]
        x_raw = _as_numeric_frame(batch, feature_columns).to_numpy(dtype=np.float32, copy=True)
        x_batch = scaler.transform(imputer.transform(x_raw)).astype(np.float32, copy=False)
        y_batch = y[batch_start:batch_end]

        tick = time.perf_counter()
        static_pred = static_model.predict(x_batch)
        static_seconds = time.perf_counter() - tick
        tick = time.perf_counter()
        online_pred = online_model.predict(x_batch)
        online_seconds = time.perf_counter() - tick

        stamp_values = batch["__timestamp__"].dropna()
        start_stamp = stamp_values.iloc[0].isoformat() if len(stamp_values) else ""
        end_stamp = stamp_values.iloc[-1].isoformat() if len(stamp_values) else ""
        common = {
            "batch": batch_id + 1,
            "stream_start_row": batch_start,
            "stream_end_row_exclusive": batch_end,
            "n_samples": int(len(batch)),
            "start_timestamp": start_stamp,
            "end_timestamp": end_stamp,
        }
        for model_name, prediction, pred_seconds in [
            ("static", static_pred, static_seconds),
            ("online", online_pred, online_seconds),
        ]:
            metrics = _metrics(y_batch, prediction, task)
            batch_records.append({
                **common,
                "model": model_name,
                **metrics,
                "prediction_seconds": float(pred_seconds),
                "update_seconds": 0.0,
            })

        # Prequential protocol: first score the batch, then train on its labels.
        tick = time.perf_counter()
        stream_weights = np.asarray([class_weight_map.get(int(label), 1.0) for label in y_batch], dtype=np.float32)
        online_model.partial_fit(x_batch, y_batch, sample_weight=stream_weights)
        update_seconds = time.perf_counter() - tick
        batch_records[-1]["update_seconds"] = float(update_seconds)

        y_true_all.append(y_batch.copy())
        static_predictions.append(static_pred.copy())
        online_predictions.append(online_pred.copy())
        static_prediction_total += static_seconds
        online_prediction_total += online_seconds
        online_update_total += update_seconds
        batch_id += 1
        if batch_id == 1 or batch_id % 5 == 0 or batch_end == n_rows:
            online_f1 = batch_records[-1]["f1"]
            static_f1 = batch_records[-2]["f1"]
            print(
                f"  Batch {batch_id:>3}: rows {batch_start:,}-{batch_end - 1:,} | "
                f"static F1={static_f1:.4f}, online F1={online_f1:.4f} | "
                f"online update={update_seconds:.2f}s"
            )
        del x_raw, x_batch, y_batch, static_pred, online_pred

    if not batch_records:
        raise ValueError("No evaluation batches were produced. Increase the dataset size or reduce warm-up fraction.")

    print("\n[5/5] Saving metrics, plots, and model artifacts...")
    batch_df = pd.DataFrame(batch_records)
    batch_csv = out / "batch_metrics.csv"
    batch_df.to_csv(batch_csv, index=False)

    y_stream = np.concatenate(y_true_all)
    y_static = np.concatenate(static_predictions)
    y_online = np.concatenate(online_predictions)
    summary_records = []
    for model_name, prediction, prediction_seconds, update_seconds in [
        ("static", y_static, static_prediction_total, 0.0),
        ("online", y_online, online_prediction_total, online_update_total),
    ]:
        summary_records.append({
            "model": model_name,
            **_metrics(y_stream, prediction, task),
            "evaluation_samples": int(len(y_stream)),
            "initial_train_seconds_shared": float(initial_train_seconds),
            "prediction_seconds": float(prediction_seconds),
            "avg_prediction_ms_per_sample": float(prediction_seconds * 1000 / max(1, len(y_stream))),
            "update_seconds": float(update_seconds),
        })
    summary_df = pd.DataFrame(summary_records)
    summary_csv = out / "summary_metrics.csv"
    summary_df.to_csv(summary_csv, index=False)

    plot_files = _make_plots(batch_df, out)
    model_dir = out / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    common_artifact = {
        "imputer": imputer,
        "scaler": scaler,
        "feature_columns": feature_columns,
        "task": task,
        "class_names": class_names,
        "label_mapping": label_mapping,
        "warmup_rows": int(start_index),
        "dataset_sort_method": data_info["sort_method"],
    }
    joblib.dump({**common_artifact, "estimator": static_model, "model_type": "static"}, model_dir / "static_model.joblib")
    joblib.dump({**common_artifact, "estimator": online_model, "model_type": "online_incremental"}, model_dir / "online_model_final.joblib")

    label_counts = pd.Series(data["__target_label__"].astype(str)).value_counts().to_dict()
    run_info = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "task": task,
        "class_names": class_names,
        "label_mapping": label_mapping,
        "rows": int(len(data)),
        "candidate_feature_count": int(len(data_info["feature_columns"])),
        "used_feature_count": int(len(feature_columns)),
        "used_features": feature_columns,
        "warmup_rows": int(start_index),
        "warmup_fraction_requested": float(initial_fraction),
        "warmup_fraction_actual": float(warmup_fraction_actual),
        "stream_rows": int(n_rows - start_index),
        "batch_size": int(batch_size),
        "batch_count": int(batch_id),
        "initial_epochs": int(initial_epochs),
        "balance_initial_classes": bool(balance),
        "class_weights_from_warmup": {str(k): float(v) for k, v in class_weight_map.items()},
        "initial_train_seconds": float(initial_train_seconds),
        "label_counts": {str(k): int(v) for k, v in label_counts.items()},
        "dataset": data_info,
        "outputs": {
            "batch_metrics_csv": str(batch_csv),
            "summary_metrics_csv": str(summary_csv),
            "plots": plot_files,
            "static_model": str(model_dir / "static_model.joblib"),
            "online_model_final": str(model_dir / "online_model_final.joblib"),
        },
    }
    (out / "run_metadata.json").write_text(json.dumps(run_info, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== OVERALL STREAM RESULTS ===")
    with pd.option_context("display.max_columns", None, "display.width", 160):
        print(summary_df.round(4).to_string(index=False))
    print(f"\nSaved results to: {out}")
    print(f"  - {batch_csv.name}: per-batch metrics for time-aware analysis")
    print(f"  - {summary_csv.name}: overall stream comparison")
    print("  - f1_over_time.png, recall_over_time.png, accuracy_over_time.png")
    print("  - models/static_model.joblib and models/online_model_final.joblib")

    return run_info
