
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
    confusion_matrix
)


DATA_DIR = "data/MachineLearningCSV"
HELDOUT_FILE = os.path.join(
    DATA_DIR,
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
)

SAMPLE_SIZES = [20000, 40000]
RANDOM_STATE = 42


def load_csv(path):
    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.strip()
    df["Label"] = df["Label"].astype(str).str.strip()
    return df


def stratified_sample(df, size):
    if len(df) <= size:
        return df.copy()

    sampled, _ = train_test_split(
        df,
        train_size=size,
        random_state=RANDOM_STATE,
        stratify=df["Label"]
    )

    return sampled


def prepare_features(df, columns=None):
    excluded = {
        "Label", "target", "Flow ID",
        "Source IP", "Destination IP", "Timestamp"
    }

    if columns is None:
        columns = [
            col for col in df.columns
            if col not in excluded
        ]

    X = df.reindex(columns=columns).copy()

    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    X = X.replace([np.inf, -np.inf], np.nan)

    return X, list(X.columns)


def make_target(df):
    return (
        df["Label"].str.upper() != "BENIGN"
    ).astype(int)


def evaluate_model(name, model, X_test, y_test):
    predictions = model.predict(X_test)

    tn, fp, fn, tp = confusion_matrix(
        y_test, predictions, labels=[0, 1]
    ).ravel()

    result = {
        "Training size per file": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "F1": f1_score(
            y_test, predictions, zero_division=0
        ),
        "False positives": int(fp),
        "Missed attacks": int(fn),
        "True negatives": int(tn),
        "True positives": int(tp)
    }

    return result


def main():
    if not os.path.exists(HELDOUT_FILE):
        raise FileNotFoundError(HELDOUT_FILE)

    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))
    train_files = [
        p for p in files
        if os.path.abspath(p) != os.path.abspath(HELDOUT_FILE)
    ]

    if not train_files:
        raise RuntimeError("No training files found.")

    print("Loading held-out DDoS file...")
    test_df = load_csv(HELDOUT_FILE)
    y_test = make_target(test_df).to_numpy()

    X_test, columns = prepare_features(test_df)

    usable = X_test.columns[X_test.notna().any()].tolist()
    X_test = X_test[usable]

    print("Loading training files...")
    raw_frames = {}

    for path in train_files:
        df = load_csv(path)
        raw_frames[path] = df
        print(f"{os.path.basename(path)}: {len(df)} rows available")

    test_hashes = set(
        pd.util.hash_pandas_object(
            X_test, index=False
        ).to_numpy()
    )

    results = []

    for sample_size in SAMPLE_SIZES:
        print(f"\n{'=' * 55}")
        print(f"EXPERIMENT: {sample_size} rows per training file")
        print("=" * 55)

        sampled_frames = []

        for path, df in raw_frames.items():
            sampled = stratified_sample(df, sample_size)
            sampled_frames.append(sampled)

        train_df = pd.concat(
            sampled_frames, ignore_index=True
        )

        y_train = make_target(train_df).reset_index(drop=True)
        X_train, _ = prepare_features(train_df, usable)

        X_train = X_train.replace([np.inf, -np.inf], np.nan)

        train_hashes = pd.util.hash_pandas_object(
            X_train, index=False
        ).to_numpy()

        overlap = np.isin(train_hashes, list(test_hashes))
        removed = int(overlap.sum())

        keep = ~overlap
        X_train = X_train.loc[keep].reset_index(drop=True)
        y_train = y_train.loc[keep].reset_index(drop=True)

        print(f"Rows after overlap removal: {len(X_train)}")
        print(f"Exact overlaps removed: {removed}")
        print("Training classes:")
        print(y_train.map({0: "BENIGN", 1: "MALICIOUS"}).value_counts())

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

        print("Training Random Forest...")
        model.fit(X_train, y_train)

        result = evaluate_model(
            sample_size, model, X_test, y_test
        )

        results.append(result)

        print("\nHeld-out DDoS results:")
        for key, value in result.items():
            print(f"{key}: {value}")

    table = pd.DataFrame(results)

    print("\nFINAL TRAINING-SIZE COMPARISON")
    print(table.round(4).to_string(index=False))

    os.makedirs("results", exist_ok=True)
    table.to_csv(
        "results/training_size_comparison.csv",
        index=False
    )

    print("\nSaved results/training_size_comparison.csv")


if __name__ == "__main__":
    main()
