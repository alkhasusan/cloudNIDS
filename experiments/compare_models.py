
import os
import glob
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
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

SAMPLE_PER_FILE = 20000
RANDOM_STATE = 42


def load_csv(path, sample_size=None):
    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.strip()
    df["Label"] = df["Label"].astype(str).str.strip()

    if sample_size is not None and len(df) > sample_size:
        df = df.sample(
            n=sample_size,
            random_state=RANDOM_STATE
        )

    return df


def prepare_features(df, columns=None):
    excluded = {
        "Label",
        "target",
        "Flow ID",
        "Source IP",
        "Destination IP",
        "Timestamp"
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


def log_transform(X):
    return np.sign(X) * np.log1p(np.abs(X))


def report(name, y_true, probabilities, threshold=0.5):
    predictions = (probabilities >= threshold).astype(int)

    return {
        "Model": name,
        "Threshold": threshold,
        "Accuracy": accuracy_score(y_true, predictions),
        "Precision": precision_score(
            y_true, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_true, predictions, zero_division=0
        ),
        "F1": f1_score(
            y_true, predictions, zero_division=0
        ),
        "Confusion Matrix": confusion_matrix(
            y_true, predictions
        )
    }


def main():
    if not os.path.exists(HELDOUT_FILE):
        raise FileNotFoundError(HELDOUT_FILE)

    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))

    train_files = [
        path for path in files
        if os.path.abspath(path) != os.path.abspath(HELDOUT_FILE)
    ]

    if not train_files:
        raise RuntimeError("No training files found.")

    print("Loading training files...")

    frames = []

    for path in train_files:
        df = load_csv(path, SAMPLE_PER_FILE)
        frames.append(df)
        print(f"{os.path.basename(path)}: {len(df)} rows")

    train_df = pd.concat(frames, ignore_index=True)

    print("\nLoading held-out DDoS file...")
    test_df = load_csv(HELDOUT_FILE)

    y_all = make_target(train_df).reset_index(drop=True)
    y_test = make_target(test_df).to_numpy()

    X_all, columns = prepare_features(train_df)
    X_test, _ = prepare_features(test_df, columns)

    usable = X_all.columns[X_all.notna().any()].tolist()
    X_all = X_all[usable]
    X_test = X_test[usable]

    print("\nRemoving exact feature-row overlaps...")

    test_hashes = set(
        pd.util.hash_pandas_object(
            X_test, index=False
        ).to_numpy()
    )

    train_hashes = pd.util.hash_pandas_object(
        X_all, index=False
    ).to_numpy()

    overlap = np.isin(train_hashes, list(test_hashes))
    removed = int(overlap.sum())

    keep = ~overlap
    X_all = X_all.loc[keep].reset_index(drop=True)
    y_all = y_all.loc[keep].reset_index(drop=True)

    print(f"Exact overlapping rows removed: {removed}")
    print(f"Training rows remaining: {len(X_all)}")
    print(f"Held-out rows: {len(X_test)}")
    print(f"Features: {len(usable)}")

    X_train, X_val, y_train, y_val = train_test_split(
        X_all,
        y_all,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y_all
    )

    models = {
        "Random Forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", RandomForestClassifier(
                n_estimators=100,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                class_weight="balanced",
                max_features="sqrt"
            ))
        ]),

        "Log Random Forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("log", FunctionTransformer(log_transform)),
            ("classifier", RandomForestClassifier(
                n_estimators=100,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                class_weight="balanced",
                max_features="sqrt"
            ))
        ]),

        "Extra Trees": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", ExtraTreesClassifier(
                n_estimators=100,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                class_weight="balanced",
                max_features="sqrt"
            ))
        ])
    }

    results = []

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)

        val_prob = model.predict_proba(X_val)[:, 1]
        test_prob = model.predict_proba(X_test)[:, 1]

        results.append(
            report(name, y_val.to_numpy(), val_prob)
        )

        results.append(
            report(name, y_test, test_prob)
        )

    table = pd.DataFrame([
        {
            "Model": result["Model"],
            "Dataset": (
                "Validation"
                if i % 2 == 0
                else "Held-out DDoS"
            ),
            "Accuracy": result["Accuracy"],
            "Precision": result["Precision"],
            "Recall": result["Recall"],
            "F1": result["F1"]
        }
        for i, result in enumerate(results)
    ])

    print("\nMODEL COMPARISON")
    print(table.round(4).to_string(index=False))

    print("\nHELD-OUT CONFUSION MATRICES")

    for result in results[1::2]:
        print(f"\n{result['Model']}")
        print(result["Confusion Matrix"])

    os.makedirs("results", exist_ok=True)
    table.to_csv("results/model_comparison.csv", index=False)
    print("\nSaved results to results/model_comparison.csv")


if __name__ == "__main__":
    main()
