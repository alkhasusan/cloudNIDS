
from pathlib import Path
import json
import time

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "MachineLearningCSV"
MODEL_DIR = BASE_DIR / "models"
RESULT_DIR = BASE_DIR / "results"

MODEL_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)

SAMPLES_PER_FILE = 20000
RANDOM_STATE = 42


def load_data():
    files = list(DATA_DIR.glob("*.csv"))

    if not files:
        raise FileNotFoundError(
            f"No CSV files found in {DATA_DIR}"
        )

    frames = []

    for file in files:
        print(f"Reading {file.name}")

        df = pd.read_csv(file, low_memory=False)
        df.columns = df.columns.str.strip()

        label_col = next(
            (c for c in df.columns if c.lower() == "label"),
            None
        )

        if label_col is None:
            print(f"Skipping {file.name}: no Label column")
            continue

        if len(df) > SAMPLES_PER_FILE:
            df = df.sample(
                n=SAMPLES_PER_FILE,
                random_state=RANDOM_STATE
            )

        df = df.rename(columns={label_col: "Label"})
        frames.append(df)

    if not frames:
        raise ValueError("No usable labelled CSV files found.")

    data = pd.concat(frames, ignore_index=True)
    data["Label"] = data["Label"].astype(str).str.strip()

    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna(subset=["Label"])
    data = data.drop_duplicates()

    data["target"] = (
        data["Label"].str.upper() != "BENIGN"
    ).astype(int)

    print("\nOriginal label distribution:")
    print(data["Label"].value_counts().to_string())

    print("\nBinary class distribution:")
    print(data["target"].value_counts().to_string())

    return data


def main():
    start = time.time()
    data = load_data()

    excluded = {
        "Label",
        "target",
        "Flow ID",
        "Source IP",
        "Destination IP",
        "Timestamp"
    }

    feature_cols = [
        col for col in data.columns
        if col not in excluded
    ]

    X = data[feature_cols].apply(
        pd.to_numeric, errors="coerce"
    )
    X = X.replace([np.inf, -np.inf], np.nan)

    valid_cols = X.columns[X.notna().any()].tolist()
    X = X[valid_cols]
    feature_cols = valid_cols

    y = data["target"]

    if y.nunique() != 2:
        raise ValueError(
            "Expected both benign and malicious records."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y
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

    print(f"\nTraining records: {len(X_train)}")
    print(f"Testing records: {len(X_test)}")
    print(f"Features: {len(feature_cols)}")

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(
            precision_score(y_test, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y_test, predictions, zero_division=0)
        ),
        "f1_score": float(
            f1_score(y_test, predictions, zero_division=0)
        ),
        "training_records": len(X_train),
        "testing_records": len(X_test),
        "features": len(feature_cols)
    }

    print("\nEvaluation results")
    for key, value in metrics.items():
        print(f"{key}: {value}")

    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, predictions, labels=[0, 1]))

    print("\nClassification report:")
    print(classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["BENIGN", "MALICIOUS"],
        zero_division=0
    ))

    joblib.dump(
        {
            "pipeline": model,
            "feature_cols": feature_cols,
            "class_names": {
                0: "BENIGN",
                1: "MALICIOUS"
            }
        },
        MODEL_DIR / "cloudnids_binary.joblib"
    )

    with open(RESULT_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print("\nModel saved to models/cloudnids_binary.joblib")
    print("Metrics saved to results/metrics.json")
    print(f"Elapsed seconds: {time.time() - start:.2f}")


if __name__ == "__main__":
    main()
