
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "MachineLearningCSV"
RESULTS_DIR = ROOT / "results"
MODELS_DIR = ROOT / "models"

HELDOUT_FILE = "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
SAMPLE_PER_FILE = 20000
RANDOM_STATE = 42

EXCLUDED_COLUMNS = [
    "Label",
    "target",
    "Flow ID",
    "Source IP",
    "Destination IP",
    "Timestamp",
]

THRESHOLDS = np.round(np.arange(0.05, 0.96, 0.05), 2)


def load_csv(filename, sample=True):
    path = DATA_DIR / filename

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.strip()

    if "Label" not in df.columns:
        raise ValueError(f"Label column missing from {filename}")

    df["Label"] = df["Label"].astype(str).str.strip()
    df["target"] = (
        df["Label"].str.upper() != "BENIGN"
    ).astype(int)

    if sample and len(df) > SAMPLE_PER_FILE:
        df = df.sample(
            n=SAMPLE_PER_FILE,
            random_state=RANDOM_STATE,
        )

    return df.reset_index(drop=True)


def prepare_features(df):
    X = df.drop(
        columns=[
            col for col in EXCLUDED_COLUMNS
            if col in df.columns
        ],
        errors="ignore",
    ).copy()

    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    return X.replace([np.inf, -np.inf], np.nan)


def row_hashes(X):
    X = X.astype("float64")
    return pd.util.hash_pandas_object(X, index=False)


def make_pipeline():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        (
            "model",
            RandomForestClassifier(
                n_estimators=100,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                class_weight="balanced",
                max_features="sqrt",
            ),
        ),
    ])


def get_metrics(y_true, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(int)

    return {
        "threshold": float(threshold),
        "accuracy": accuracy_score(y_true, predictions),
        "precision": precision_score(
            y_true, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_true, predictions, zero_division=0
        ),
        "f1": f1_score(
            y_true, predictions, zero_division=0
        ),
    }


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading held-out Friday DDoS file...")
    heldout = load_csv(HELDOUT_FILE, sample=False)
    y_test = heldout["target"].to_numpy()
    X_test = prepare_features(heldout)

    training_files = sorted(
        path.name
        for path in DATA_DIR.glob("*.csv")
        if path.name != HELDOUT_FILE
    )

    if len(training_files) != 7:
        raise ValueError(
            f"Expected 7 training files, found {len(training_files)}"
        )

    print("\nLoading seven training files...")
    frames = []

    for filename in training_files:
        df = load_csv(filename)
        frames.append(df)
        print(f"{filename}: {len(df):,} rows")

    training = pd.concat(frames, ignore_index=True)
    y_all = training["target"].reset_index(drop=True)
    X_all = prepare_features(training)

    common_features = [
        col for col in X_all.columns if col in X_test.columns
    ]

    X_all = X_all[common_features]
    X_test = X_test[common_features]

    all_missing = X_all.columns[X_all.isna().all()].tolist()

    if all_missing:
        print("\nRemoving all-missing training features:", all_missing)
        X_all = X_all.drop(columns=all_missing)
        X_test = X_test.drop(columns=all_missing)

    feature_cols = X_all.columns.tolist()

    print("\nRemoving exact overlapping feature rows...")
    test_hashes = set(row_hashes(X_test).tolist())
    overlap_mask = row_hashes(X_all).isin(test_hashes)

    print(f"Overlapping training rows removed: {overlap_mask.sum():,}")

    keep = ~overlap_mask
    X_all = X_all.loc[keep].reset_index(drop=True)
    y_all = y_all.loc[keep].reset_index(drop=True)

    print(f"Training rows remaining: {len(X_all):,}")
    print(f"Held-out test rows: {len(X_test):,}")
    print(f"Features: {len(feature_cols)}")

    X_train, X_val, y_train, y_val = train_test_split(
        X_all,
        y_all,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y_all,
    )

    print("\nTraining model for threshold selection...")
    validation_model = make_pipeline()
    validation_model.fit(X_train, y_train)

    positive_class_index = list(
        validation_model.named_steps["model"].classes_
    ).index(1)

    val_probabilities = validation_model.predict_proba(X_val)[
        :, positive_class_index
    ]

    print("\n=== VALIDATION THRESHOLD COMPARISON ===")
    validation_results = []

    for threshold in THRESHOLDS:
        metrics = get_metrics(
            y_val,
            val_probabilities,
            threshold,
        )
        validation_results.append(metrics)

    results_df = pd.DataFrame(validation_results)
    results_df.to_csv(
        RESULTS_DIR / "threshold_comparison.csv",
        index=False,
    )

    print(
        results_df.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    # Choose the threshold with the highest validation F1.
    # If tied, prefer higher recall, then higher precision.
    best_row = max(
        validation_results,
        key=lambda row: (
            row["f1"],
            row["recall"],
            row["precision"],
        ),
    )

    best_threshold = best_row["threshold"]

    print(f"\nSelected validation threshold: {best_threshold:.2f}")
    print(f"Validation F1: {best_row['f1']:.4f}")
    print(f"Validation recall: {best_row['recall']:.4f}")
    print(f"Validation precision: {best_row['precision']:.4f}")

    # Refit using all seven training files.
    # The Friday file remains excluded from model fitting.
    print("\nRetraining final model on all seven training files...")
    final_model = make_pipeline()
    final_model.fit(X_all, y_all)

    # Evaluate the selected threshold on the held-out file.
    positive_class_index = list(
        final_model.named_steps["model"].classes_
    ).index(1)

    test_probabilities = final_model.predict_proba(X_test)[
        :, positive_class_index
    ]

    test_metrics = get_metrics(
        y_test,
        test_probabilities,
        best_threshold,
    )

    test_predictions = (
        test_probabilities >= best_threshold
    ).astype(int)

    print("\n=== FINAL HELD-OUT RESULTS ===")
    print(f"Threshold: {best_threshold:.2f}")
    print(f"Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"Precision: {test_metrics['precision']:.4f}")
    print(f"Recall:    {test_metrics['recall']:.4f}")
    print(f"F1 score:  {test_metrics['f1']:.4f}")

    print("\nConfusion matrix (rows=actual, columns=predicted):")
    print(confusion_matrix(y_test, test_predictions, labels=[0, 1]))

    print("\nClass order: 0 = BENIGN, 1 = MALICIOUS")

    model_path = MODELS_DIR / "cloudnids_binary_tuned.joblib"

    joblib.dump(
        {
            "pipeline": final_model,
            "feature_cols": feature_cols,
            "class_names": ["BENIGN", "MALICIOUS"],
            "threshold": best_threshold,
        },
        model_path,
    )

    summary = {
        "training_files": training_files,
        "heldout_file": HELDOUT_FILE,
        "training_rows": int(len(X_all)),
        "heldout_rows": int(len(X_test)),
        "features": int(len(feature_cols)),
        "selected_threshold": float(best_threshold),
        "validation_metrics": best_row,
        "heldout_metrics": {
            key: float(value)
            for key, value in test_metrics.items()
        },
        "confusion_matrix": confusion_matrix(
            y_test,
            test_predictions,
            labels=[0, 1],
        ).tolist(),
    }

    with open(
        RESULTS_DIR / "threshold_tuning_metrics.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(summary, file, indent=4)

    print(f"\nSaved tuned model: {model_path}")
    print(
        "Saved validation thresholds: "
        f"{RESULTS_DIR / 'threshold_comparison.csv'}"
    )
    print(
        "Saved metrics: "
        f"{RESULTS_DIR / 'threshold_tuning_metrics.json'}"
    )


if __name__ == "__main__":
    main()
