# Online / Incremental Intrusion Detection

**GitHub repository name:** `online-incremental-ids`

A beginner-friendly research prototype that compares a **static intrusion-detection model** with an **online/incremental model** on network-traffic batches from CICIDS2017. It is designed to run locally in PyCharm on Linux, Windows, or macOS.

## What the project does

1. Loads the labelled CICIDS-style traffic CSV files and cleans column names, numeric features, missing values, and infinite values.
2. Orders the records by `Timestamp` when usable timestamps are present.
3. Trains two matching `SGDClassifier` models on the same initial chronological warm-up period.
4. Processes the remaining traffic in batches. Both models predict each batch, metrics are recorded, and **only the online model** is then updated with that batch's labels using `partial_fit()`.
5. Compares the models across time and saves CSV metrics, figures, run metadata, and model artifacts.

The default task is **binary detection**: `BENIGN` versus `ATTACK`. An optional multiclass task predicts the dataset's original label categories.

## Repository structure

```text
online-incremental-ids/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/       # Put extracted CICIDS2017 CSV files here
│   └── demo/      # Generated synthetic smoke-test files (not real research data)
├── outputs/       # Metrics, plots, model artifacts, and run metadata
├── scripts/
│   ├── check_dataset.py
│   ├── generate_demo_data.py
│   └── run_experiment.py
└── src/
    └── online_incremental_ids/
        ├── __init__.py
        ├── dataset.py
        └── experiment.py
```

## 1. Set up the project in PyCharm

1. Download or clone this repository, then open the `online-incremental-ids` folder in PyCharm.
2. Open **Settings → Project → Python Interpreter** and create a new virtual environment (`.venv`) using Python 3.10 or newer.
3. Open the PyCharm terminal at the project root and install dependencies:

   ```bash
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

4. Run configurations can target `scripts/check_dataset.py`, `scripts/generate_demo_data.py`, or `scripts/run_experiment.py`. Their paths are resolved relative to the project root, so run them with the project root as the working directory.

## 2. Put CICIDS2017 in the correct folder

Download CICIDS2017 from the [Canadian Institute for Cybersecurity dataset page](https://www.unb.ca/cic/datasets/ids-2017.html), or use the dataset archive supplied by your university. Extract the CSV files first; ZIP files are not read directly by this project.

Put the **CSV traffic files** in:

```text
data/raw/
```

For example, these common CICIDS2017 flow CSV files can sit directly in `data/raw/` or in a subfolder under it:

```text
 data/raw/
 ├── Monday-WorkingHours.pcap_ISCX.csv
 ├── Tuesday-WorkingHours.pcap_ISCX.csv
 ├── Wednesday-workingHours.pcap_ISCX.csv
 ├── Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
 └── ...
```

You do not need to rename the files. The loader searches subfolders recursively and looks for a label column such as `Label`.

**Important:** place only one copy/variant of the dataset in the selected directory. Do not put both original and already-merged CSVs together, because duplicate records may be counted twice. Do not commit the full dataset to GitHub; `data/raw/` is ignored by Git.

## 3. Check the dataset

From the project root:

```bash
python scripts/check_dataset.py --data data/raw
```

The script prints the number of labelled rows, usable numeric features, timestamp coverage, and label counts. If files are skipped, it explains why. If no CSVs are found, make sure the archive has been extracted and that `--data` points to the correct folder.

For a quick preview of each CSV, you can limit ingestion to its first 50,000 rows:

```bash
python scripts/check_dataset.py --data data/raw --rows-per-file 50000
```

That limit is for debugging only; the subset may not represent the entire dataset.

## 4. Run the experiment

Recommended default run:

```bash
python scripts/run_experiment.py --data data/raw
```

It defaults to binary detection, a 30% initial chronological warm-up, 50,000 rows per stream batch, and five warm-up passes. If the initial period contains only one target class, the program extends that period in 5-percentage-point steps (up to 80%) until it finds both benign and attack records. This uses only the labels in the initial chronological prefix.

A smaller, faster debugging run (first 100,000 rows from **each** CSV file):

```bash
python scripts/run_experiment.py --data data/raw --rows-per-file 100000 --batch-size 10000
```

The `--rows-per-file` option limits ingestion per file; it is not a statistically representative sample and should not be used for final research claims.

Optional multiclass attack-label evaluation:

```bash
python scripts/run_experiment.py --data data/raw --task multiclass --batch-size 50000
```

Useful options:

| Option | Default | Meaning |
|---|---:|---|
| `--data` | `data/raw` | CSV file or directory containing extracted CSV files |
| `--output` | `outputs/latest` | Folder for this run's outputs |
| `--task` | `binary` | `binary` (BENIGN vs ATTACK) or `multiclass` (original labels) |
| `--initial-fraction` | `0.30` | Requested fraction reserved for chronological warm-up (0.05–0.80) |
| `--batch-size` | `50000` | Number of rows scored together before the online update |
| `--initial-epochs` | `5` | Initial training passes applied equally to both models |
| `--rows-per-file` | `0` | First N rows per CSV for quick debugging; `0` reads all rows |
| `--no-balance` | off | Disables sample weighting based on warm-up class frequencies |
| `--seed` | `42` | Random seed |

### Optional smoke test without downloading CICIDS2017

This makes small synthetic traffic CSV files in `data/demo/`. They are only for verifying that code and plotting work; they are **not CICIDS2017 data** and must not be reported as research results.

```bash
python scripts/generate_demo_data.py
python scripts/check_dataset.py --data data/demo
python scripts/run_experiment.py --data data/demo --batch-size 2000 --initial-fraction 0.30
```

## 5. Where to find the results

After a successful run, open `outputs/latest/`:

| File | Contents |
|---|---|
| `summary_metrics.csv` | Overall stream results for static and online models |
| `batch_metrics.csv` | Accuracy, precision, recall, F1, prediction time, and online update time per chronological batch |
| `f1_over_time.png` | F1 comparison across stream batches |
| `recall_over_time.png` | Recall comparison across batches (attack recall in binary mode) |
| `accuracy_over_time.png` | Accuracy comparison across batches |
| `run_metadata.json` | Dataset files, feature list, label counts, warm-up configuration, and output paths |
| `models/static_model.joblib` | Frozen classifier plus preprocessing objects and feature names |
| `models/online_model_final.joblib` | Final incrementally updated classifier plus preprocessing objects and feature names |

For a time-aware discussion, inspect both the overall table and the per-batch plots. An online model is not guaranteed to outperform a static model; adaptation can help or hurt depending on label quality, the kind of drift, batch size, and update strategy.

## Evaluation design

- **Chronological evaluation:** records are sorted by the parsed timestamp where available. If no usable timestamp exists, the loader warns and preserves CSV/file row order instead.
- **Warm-up / stream separation:** the initial prefix trains both models; later records form the evaluation stream.
- **Matched baseline:** both models use the same `SGDClassifier`, initial observations, preprocessing, number of warm-up passes, and seed.
- **Prequential evaluation:** each future batch is first predicted and scored, then its labels are used to update the online model. The static model is never updated.
- **Preprocessing is frozen:** the median imputer and standard scaler are fit on warm-up data only, preventing future-batch statistics from leaking into evaluation.
- **Class imbalance:** by default, sample weights based on warm-up label frequencies are used for the warm-up and subsequent online updates. Classes not observed in warm-up receive neutral weight until learned from stream labels.

### Metrics

For binary detection, `precision`, `recall`, and `f1` refer to the positive **ATTACK** class. For multiclass mode, those metrics are weighted averages. `macro_f1` is also recorded to reveal performance differences across labels. Accuracy, prediction time, online-update time, and per-batch metrics are saved as well.

## Limitations and research cautions

- CICIDS2017 is an offline benchmark dataset. Incremental `partial_fit()` over its timestamp-ordered records simulates a stream; it is not a live network sensor.
- The online update uses the batch's ground-truth labels after scoring that batch. This assumes labels become available for online learning. In a real organization, alert verification can be delayed, incomplete, or wrong; a later project extension could model delayed analyst feedback.
- CICIDS2017 traffic and labels have known dataset-specific limitations. Report the exact files, filtering, chronological split, warm-up size, batch size, and evaluation protocol in any thesis or paper.
- The current baseline is a compact linear model (SGD with logistic loss), chosen because it supports efficient incremental updates. It is not expected to match every offline ensemble model.
- Avoid selecting hyperparameters based on the final test stream. For stronger research evidence, use a separate validation period or repeat experiments with more than one chronological split and seed.
- If timestamps are missing or malformed in many rows, chronological conclusions become weaker; check `timestamp_valid_fraction` in `run_metadata.json`.

## Suggested GitHub description

> Compare static and online incremental intrusion detection on CICIDS2017 using chronological traffic batches, prequential evaluation, and time-aware performance plots.

## License

Add a license file before publishing if you want others to reuse the code. Check the dataset provider's terms separately before redistributing any dataset files.
