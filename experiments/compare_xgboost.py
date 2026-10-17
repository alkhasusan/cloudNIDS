
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
from xgboost import XGBClassifier


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


def show_metrics(name, y_true, probabilities):
    predictions = (probabilities >= 0.5).astype(int)

    print(f"\n{name}")
    print("-" * len(name))
    print(f"Accuracy:  {accuracy_score(y_true, predictions):.4f}")
    print(
        f"Precision: "
        f"{precision_score(y_true, predictions, zero_division=0):.4f}"
    )
    print(
        f"Recall:    "
        f"{recall_score(y_true, predictions, zero_division=0):.4f}"
    )
    print(
        f"F1 Score:  "
        f"{f1_score(y_true, predictions, zero_division=0):.4f}"
    )
    print("Confusion matrix:")
    print(confusion_matrix(y_true, predictions))


def main():
    if not os.path.exists(HELDOUT_FILE):
        raise FileNotFoundError(HELDOUT_FILE)

    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))

    train_files = [
        path for path in files
        if os.path.abspath(path) != os.path.abspath(HELDOUT_FILE)
    ]

    print("Loading training files...")
    frames = []

    for path in train_files:
        df = load_csv(path, SAMPLE_PER_FILE)
        frames.append(df)
        print(f"{os.path.basename(path)}: {len(df)} rows")

    train_df = pd.concat(frames, ignore_index=True)
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
    print(f"Exact overlapping rows removed: {int(overlap.sum())}")

    keep = ~overlap
    X_all = X_all.loc[keep].reset_index(drop=True)
    y_all = y_all.loc[keep].reset_index(drop=True)

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

        "XGBoost": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="binary:logistic",
                eval_metric="logloss",
                tree_method="hist",
                random_state=RANDOM_STATE,
                n_jobs=-1
            ))
        ])
    }

    results = []

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)

        for dataset_name, X, y in [
            ("Validation", X_val, y_val.to_numpy()),
            ("Held-out DDoS", X_test, y_test)
        ]:
            probabilities = model.predict_proba(X)[:, 1]
            predictions = (probabilities >= 0.5).astype(int)

            results.append({
                "Model": name,
                "Dataset": dataset_name,
                "Accuracy": accuracy_score(y, predictions),
                "Precision": precision_score(
                    y, predictions, zero_division=0
                ),
                "Recall": recall_score(
                    y, predictions, zero_division=0
                ),
                "F1": f1_score(
                    y, predictions, zero_division=0
                )
            })

            show_metrics(
                f"{name} - {dataset_name}",
                y,
                probabilities
            )

    table = pd.DataFrame(results)

    print("\nFINAL COMPARISON")
    print(table.round(4).to_string(index=False))

    os.makedirs("results", exist_ok=True)
    table.to_csv("results/xgboost_comparison.csv", index=False)
    print("\nSaved results/xgboost_comparison.csv")


if __name__ == "__main__":
    main()
