
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
    classification_report,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "MachineLearningCSV"
RESULTS_DIR = ROOT / "results"

HELDOUT_FILE = (
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
)

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


def load_csv(filename):
    path = DATA_DIR / filename

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.strip()

    if "Label" not in df.columns:
        raise ValueError(f"'Label' column not found in {filename}")

    df["Label"] = df["Label"].astype(str).str.strip()
    df["target"] = (
        df["Label"].str.upper() != "BENIGN"
    ).astype(int)

    if len(df) > SAMPLE_PER_FILE and filename != HELDOUT_FILE:
        df = df.sample(
            n=SAMPLE_PER_FILE,
            random_state=RANDOM_STATE,
        )

    return df.reset_index(drop=True)


def prepare_features(df, feature_cols=None):
    excluded = [
        column
        for column in EXCLUDED_COLUMNS
        if column in df.columns
    ]

    X = df.drop(columns=excluded, errors="ignore").copy()

    for column in X.columns:
        X[column] = pd.to_numeric(
            X[column],
            errors="coerce",
        )

    X = X.replace([np.inf, -np.inf], np.nan)

    if feature_cols is not None:
        X = X.reindex(columns=feature_cols)

    return X


def row_hashes(X):
    normalized = X.copy()

    for column in normalized.columns:
        normalized[column] = normalized[column].astype("float64")

    return pd.util.hash_pandas_object(
        normalized,
        index=False,
    )


def median_summary(X, mask):
    if mask.sum() == 0:
        return pd.Series(np.nan, index=X.columns)

    return X.loc[mask].median()


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading held-out DDoS data...")

    heldout = load_csv(HELDOUT_FILE)
    y_test = heldout["target"].reset_index(drop=True)

    X_test_raw = prepare_features(heldout)

    print("\nLoading the other seven training files...")

    csv_files = sorted(
        path.name
        for path in DATA_DIR.glob("*.csv")
        if path.name != HELDOUT_FILE
    )

    if len(csv_files) != 7:
        raise ValueError(
            f"Expected 7 training CSV files, found {len(csv_files)}. "
            f"Check the files in {DATA_DIR}"
        )

    training_frames = []

    for filename in csv_files:
        df = load_csv(filename)
        training_frames.append(df)
        print(f"Loaded {filename}: {len(df):,} sampled rows")

    training = pd.concat(
        training_frames,
        ignore_index=True,
    )

    y_train_all = training["target"].reset_index(drop=True)
    X_train_raw = prepare_features(training)

    common_features = [
        column
        for column in X_train_raw.columns
        if column in X_test_raw.columns
    ]

    X_train_raw = X_train_raw[common_features]
    X_test_raw = X_test_raw[common_features]

    all_missing = X_train_raw.columns[
        X_train_raw.isna().all()
    ].tolist()

    if all_missing:
        print(
            "\nRemoving features entirely missing in training:",
            all_missing,
        )

        X_train_raw = X_train_raw.drop(columns=all_missing)
        X_test_raw = X_test_raw.drop(columns=all_missing)

    feature_cols = X_train_raw.columns.tolist()

    print(f"\nFeatures before overlap removal: {len(feature_cols)}")

    print("\nRemoving exact feature-row overlaps...")

    training_hashes = row_hashes(X_train_raw)
    heldout_hash_set = set(row_hashes(X_test_raw).tolist())

    overlap_mask = training_hashes.isin(heldout_hash_set)

    print(
        "Exact overlapping training rows removed:",
        f"{overlap_mask.sum():,}",
    )

    keep_mask = ~overlap_mask

    X_train_raw = X_train_raw.loc[keep_mask].reset_index(drop=True)
    y_train_all = y_train_all.loc[keep_mask].reset_index(drop=True)

    print(f"Remaining training rows: {len(X_train_raw):,}")
    print(f"Held-out test rows: {len(X_test_raw):,}")
    print("\nTraining class counts:")
    print(
        y_train_all.map({0: "BENIGN", 1: "MALICIOUS"})
        .value_counts()
        .to_string()
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_raw,
        y_train_all,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y_train_all,
    )

    pipeline = Pipeline([
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

    print("\nTraining a fresh Random Forest on the seven files...")
    pipeline.fit(X_train, y_train)

    print("\nEvaluating internal validation split...")
    val_pred = pipeline.predict(X_val)

    print(f"Validation accuracy:  {accuracy_score(y_val, val_pred):.4f}")
    print(
        f"Validation precision: "
        f"{precision_score(y_val, val_pred, zero_division=0):.4f}"
    )
    print(
        f"Validation recall:    "
        f"{recall_score(y_val, val_pred, zero_division=0):.4f}"
    )
    print(
        f"Validation F1:         "
        f"{f1_score(y_val, val_pred, zero_division=0):.4f}"
    )

    print("\nEvaluating the held-out Friday DDoS file...")

    test_pred = pipeline.predict(X_test_raw)

    print(f"Test accuracy:  {accuracy_score(y_test, test_pred):.4f}")
    print(
        f"Test precision: "
        f"{precision_score(y_test, test_pred, zero_division=0):.4f}"
    )
    print(
        f"Test recall:    "
        f"{recall_score(y_test, test_pred, zero_division=0):.4f}"
    )
    print(
        f"Test F1:        "
        f"{f1_score(y_test, test_pred, zero_division=0):.4f}"
    )

    print("\nConfusion matrix (rows=actual, columns=predicted):")
    print(confusion_matrix(y_test, test_pred, labels=[0, 1]))

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            test_pred,
            labels=[0, 1],
            target_names=["BENIGN", "MALICIOUS"],
            zero_division=0,
        )
    )

    malicious_test = y_test.to_numpy() == 1
    detected = malicious_test & (test_pred == 1)
    missed = malicious_test & (test_pred == 0)

    malicious_training = y_train_all.to_numpy() == 1

    print("\n=== ATTACK COUNTS ===")
    print(f"Actual malicious rows: {malicious_test.sum():,}")
    print(f"Detected attacks:      {detected.sum():,}")
    print(f"Missed attacks:        {missed.sum():,}")

    train_medians = median_summary(
        X_train_raw,
        malicious_training,
    )
    detected_medians = median_summary(
        X_test_raw,
        detected,
    )
    missed_medians = median_summary(
        X_test_raw,
        missed,
    )

    comparison = pd.DataFrame({
        "feature": feature_cols,
        "Malicious training median": train_medians.reindex(feature_cols).values,
        "Detected DDoS median": detected_medians.reindex(feature_cols).values,
        "Missed DDoS median": missed_medians.reindex(feature_cols).values,
    })

    comparison["Missed_minus_detected_median"] = (
        comparison["Missed DDoS median"]
        - comparison["Detected DDoS median"]
    ).abs()

    comparison["Missed_vs_detected_ratio"] = (
        comparison["Missed DDoS median"].abs()
        / comparison["Detected DDoS median"].abs().replace(0, np.nan)
    )

    comparison = comparison.sort_values(
        "Missed_minus_detected_median",
        ascending=False,
    )

    comparison_path = (
        RESULTS_DIR / "missed_attack_feature_comparison.csv"
    )
    comparison.to_csv(comparison_path, index=False)

    print("\n=== TOP 15 MEDIAN DIFFERENCES: MISSED VS DETECTED ===")
    print(
        comparison.head(15).to_string(
            index=False,
            float_format=lambda value: f"{value:.4g}",
        )
    )

    model = pipeline.named_steps["model"]

    importance = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    importance_path = RESULTS_DIR / "feature_importance.csv"
    importance.to_csv(importance_path, index=False)

    print("\n=== TOP 15 RANDOM FOREST FEATURE IMPORTANCES ===")
    print(
        importance.head(15).to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )

    metrics = {
        "training_files": csv_files,
        "heldout_file": HELDOUT_FILE,
        "training_rows_after_overlap_removal": int(len(X_train_raw)),
        "heldout_rows": int(len(X_test_raw)),
        "features": int(len(feature_cols)),
        "overlap_rows_removed": int(overlap_mask.sum()),
        "accuracy": float(accuracy_score(y_test, test_pred)),
        "precision": float(
            precision_score(y_test, test_pred, zero_division=0)
        ),
        "recall": float(
            recall_score(y_test, test_pred, zero_division=0)
        ),
        "f1": float(
            f1_score(y_test, test_pred, zero_division=0)
        ),
        "confusion_matrix": confusion_matrix(
            y_test,
            test_pred,
            labels=[0, 1],
        ).tolist(),
        "actual_malicious": int(malicious_test.sum()),
        "detected_malicious": int(detected.sum()),
        "missed_malicious": int(missed.sum()),
    }

    metrics_path = RESULTS_DIR / "missed_attack_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    print("\nSaved results:")
    print(comparison_path)
    print(importance_path)
    print(metrics_path)


if __name__ == "__main__":
    main()
