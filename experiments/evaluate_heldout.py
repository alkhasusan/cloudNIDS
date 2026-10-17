
import os
import glob
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


DATA_DIR = "data/MachineLearningCSV"
HELDOUT_FILE = os.path.join(
    DATA_DIR,
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
)

SAMPLE_PER_FILE = 20000
RANDOM_STATE = 42


def load_csv(path, sample_size=None):
    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.strip()

    if sample_size is not None and len(df) > sample_size:
        df = df.sample(
            n=sample_size,
            random_state=RANDOM_STATE
        )

    df["Label"] = df["Label"].astype(str).str.strip()
    return df


def prepare_features(df, feature_cols=None):
    excluded = {
        "Label",
        "target",
        "Flow ID",
        "Source IP",
        "Destination IP",
        "Timestamp"
    }

    if feature_cols is None:
        feature_cols = [
            col for col in df.columns
            if col not in excluded
        ]

    X = df.reindex(columns=feature_cols).copy()

    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    X = X.replace([np.inf, -np.inf], np.nan)

    return X, feature_cols


def make_target(df):
    return (
        df["Label"].str.upper() != "BENIGN"
    ).astype(int)


def show_metrics(name, y_true, y_pred):
    print(f"\n{name}")
    print("-" * len(name))
    print(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
    print(
        f"Precision: "
        f"{precision_score(y_true, y_pred, zero_division=0):.4f}"
    )
    print(
        f"Recall:    "
        f"{recall_score(y_true, y_pred, zero_division=0):.4f}"
    )
    print(
        f"F1 Score:  "
        f"{f1_score(y_true, y_pred, zero_division=0):.4f}"
    )
    print("Confusion matrix:")
    print(confusion_matrix(y_true, y_pred))


def main():
    if not os.path.exists(HELDOUT_FILE):
        raise FileNotFoundError(
            f"Held-out file not found: {HELDOUT_FILE}"
        )

    all_files = sorted(
        glob.glob(os.path.join(DATA_DIR, "*.csv"))
    )

    train_files = [
        path for path in all_files
        if os.path.abspath(path) != os.path.abspath(HELDOUT_FILE)
    ]

    if not train_files:
        raise RuntimeError("No training CSV files found.")

    print("Loading training files...")

    train_frames = []

    for path in train_files:
        df = load_csv(path, SAMPLE_PER_FILE)
        train_frames.append(df)
        print(f"{os.path.basename(path)}: {len(df)} rows")

    train_df = pd.concat(train_frames, ignore_index=True)

    print("\nLoading held-out DDoS file...")
    test_df = load_csv(HELDOUT_FILE)

    y_all = make_target(train_df)
    y_test = make_target(test_df)

    X_all, feature_cols = prepare_features(train_df)
    X_test, _ = prepare_features(test_df, feature_cols)

    # Remove columns that contain no usable training values.
    usable_cols = X_all.columns[X_all.notna().any()].tolist()
    X_all = X_all[usable_cols]
    X_test = X_test[usable_cols]
    feature_cols = usable_cols

    # Remove exact feature-row overlaps with the held-out file.
    print("\nChecking for exact feature-row overlaps...")

    test_hashes = set(
        pd.util.hash_pandas_object(
            X_test, index=False
        ).to_numpy()
    )

    train_hashes = pd.util.hash_pandas_object(
        X_all, index=False
    ).to_numpy()

    overlap_mask = np.isin(train_hashes, list(test_hashes))
    overlap_count = int(overlap_mask.sum())

    keep_mask = ~overlap_mask

    X_all = X_all.loc[keep_mask].reset_index(drop=True)
    y_all = y_all.loc[keep_mask].reset_index(drop=True)

    print(f"Exact overlapping training rows removed: {overlap_count}")
    print(f"Remaining training rows: {len(X_all)}")
    print(f"Held-out test rows: {len(X_test)}")
    print(f"Features: {len(feature_cols)}")

    print("\nTraining label distribution:")
    print(y_all.map({0: "BENIGN", 1: "MALICIOUS"}).value_counts())

    print("\nHeld-out label distribution:")
    print(y_test.map({0: "BENIGN", 1: "MALICIOUS"}).value_counts())

    # One fixed train/validation split.
    X_train, X_val, y_train, y_val = train_test_split(
        X_all,
        y_all,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y_all
    )

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(
            n_estimators=100,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            class_weight="balanced",
            max_features="sqrt"
        ))
    ])

    print("\nTraining Random Forest once...")
    model.fit(X_train, y_train)

    # Validation results from the training files.
    val_pred = model.predict(X_val)
    show_metrics("VALIDATION RESULTS", y_val, val_pred)

    # Final evaluation on the separate DDoS file.
    test_pred = model.predict(X_test)
    show_metrics("HELD-OUT DDOS RESULTS", y_test, test_pred)

    print("\nHeld-out classification report:")
    print(
        classification_report(
            y_test,
            test_pred,
            labels=[0, 1],
            target_names=["BENIGN", "MALICIOUS"],
            zero_division=0
        )
    )

    print(
        "\nNote: this evaluates generalization to a separate DDoS CSV. "
        "It does not guarantee performance on live network traffic."
    )


    # Inspect missed malicious flows.
    missed_mask = (y_test.to_numpy() == 1) & (test_pred == 0)
    missed = X_test.loc[missed_mask]

    print("\nMISSED MALICIOUS FLOWS")
    print(f"Count: {len(missed)}")

    if len(missed) > 0:
        comparison = pd.DataFrame({
            "Missed malicious": missed.mean(),
            "Detected malicious": X_test.loc[
                (y_test.to_numpy() == 1) & (test_pred == 1)
            ].mean()
        })

        comparison["Absolute difference"] = (
            comparison["Missed malicious"]
            - comparison["Detected malicious"]
        ).abs()

        print("\nFeatures with the largest mean differences:")
        print(
            comparison.sort_values(
                "Absolute difference", ascending=False
            ).head(15).round(3).to_string()
        )


    # Compare distributions of missed and detected DDoS flows.
    detected_mask = (
        (y_test.to_numpy() == 1) & (test_pred == 1)
    )

    detected = X_test.loc[detected_mask]

    important_features = [
        "Flow Duration",
        "Fwd IAT Total",
        "Idle Max",
        "Flow IAT Max",
        "Fwd IAT Max",
        "Idle Mean",
        "Packet Length Variance",
    ]

    print("\nDDoS FEATURE DISTRIBUTION COMPARISON")

    for feature in important_features:
        if feature not in X_test.columns:
            continue

        print(f"\n{feature}")

        print("Missed:")
        print(
            missed[feature]
            .quantile([0.00, 0.25, 0.50, 0.75, 0.90, 0.99])
            .round(3)
            .to_string()
        )

        print("Detected:")
        print(
            detected[feature]
            .quantile([0.00, 0.25, 0.50, 0.75, 0.90, 0.99])
            .round(3)
            .to_string()
        )


if __name__ == "__main__":
    main()
