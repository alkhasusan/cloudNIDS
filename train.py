from pathlib import Path
import json
import time

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "MachineLearningCSV"
MODEL_DIR = BASE_DIR / "models"
RESULT_DIR = BASE_DIR / "results"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLES_PER_FILE = 20000
RANDOM_STATE = 42
N_FEATURES = 30


def normalize_label(label):
    """Convert original dataset labels into multiclass categories."""
    label = str(label).strip()
    label = label.replace("\u2013", "-").replace("\u2014", "-")
    label = " ".join(label.split())
    key = label.casefold()

    if key == "benign":
        return "BENIGN"

    dos_variants = {
        "dos hulk", "dos goldeneye", "dos slowloris", "dos slowhttptest"
    }
    if key in dos_variants:
        return "DoS"

    known_labels = {
        "ddos": "DDoS",
        "portscan": "PortScan",
        "ftp-patator": "FTP-Patator",
        "ssh-patator": "SSH-Patator",
        "bot": "Bot",
        "infiltration": "Infiltration",
        "heartbleed": "Heartbleed",
        "web attack - brute force": "Web Attack - Brute Force",
        "web attack - xss": "Web Attack - XSS",
        "web attack - sql injection": "Web Attack - SQL Injection"
    }
    return known_labels.get(key, label)


def load_data():
    files = sorted(DATA_DIR.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {DATA_DIR}")

    frames = []
    for file in files:
        print(f"\nReading {file.name}")
        df = pd.read_csv(file, low_memory=False)
        df.columns = df.columns.str.strip()

        label_col = next(
            (c for c in df.columns if c.casefold() == "label"), None
        )
        if label_col is None:
            print("Skipping file: no Label column")
            continue

        df = df.rename(columns={label_col: "Label"})

        if len(df) > SAMPLES_PER_FILE:
            sampled = df.sample(n=SAMPLES_PER_FILE, random_state=RANDOM_STATE)
            counts = df["Label"].value_counts()
            rare_labels = counts[counts < 100].index
            rare_rows = df[df["Label"].isin(rare_labels)]
            df = pd.concat([sampled, rare_rows], ignore_index=True).drop_duplicates()

        frames.append(df)

    if not frames:
        raise ValueError("No usable labelled CSV files found.")

    data = pd.concat(frames, ignore_index=True)
    data["Label"] = data["Label"].astype(str).str.strip()
    data = data.dropna(subset=["Label"])
    data = data[data["Label"] != ""]
    data["target"] = data["Label"].apply(normalize_label)
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.drop_duplicates().reset_index(drop=True)

    print("\nOriginal dataset labels:")
    print(data["Label"].value_counts().to_string())
    print("\nMulticlass label distribution:")
    print(data["target"].value_counts().to_string())
    print(f"\nTotal records: {len(data)}")
    print(f"Total classes: {data['target'].nunique()}")
    return data


def make_classifier():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(
            n_estimators=200,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            class_weight="balanced_subsample",
            max_features="sqrt"
        ))
    ])


def main():
    start = time.time()
    data = load_data()

    excluded = {
        "Label", "target", "Flow ID", "Source IP",
        "Destination IP", "Timestamp", "Unnamed: 0"
    }
    feature_cols = [col for col in data.columns if col not in excluded]

    X = data[feature_cols].apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    feature_cols = X.columns[X.notna().any()].tolist()
    X = X[feature_cols]
    y = data["target"]

    if y.nunique() < 2:
        raise ValueError("Multiclass training requires at least two classes.")

    counts = y.value_counts()
    singleton_mask = y.map(counts) < 2
    X_regular, y_regular = X.loc[~singleton_mask], y.loc[~singleton_mask]
    X_rare, y_rare = X.loc[singleton_mask], y.loc[singleton_mask]

    if y_regular.nunique() < 2:
        raise ValueError("Not enough examples per class for a reliable split.")

    # Reserve a test set before feature selection. Selector is fit only on training.
    X_train, X_test, y_train, y_test = train_test_split(
        X_regular, y_regular, test_size=0.20,
        random_state=RANDOM_STATE, stratify=y_regular
    )

    if len(X_rare):
        X_train = pd.concat([X_train, X_rare])
        y_train = pd.concat([y_train, y_rare])
        print(
            f"\nNote: {len(X_rare)} records from classes with only one "
            "example were placed in training and cannot be evaluated on test data."
        )

    n_selected = min(N_FEATURES, len(feature_cols))
    # Mutual information scores are calculated after fitting imputation on training only.
    selection_imputer = SimpleImputer(strategy="median")
    X_train_imputed = selection_imputer.fit_transform(X_train)

    selector = SelectKBest(
        score_func=lambda features, labels: mutual_info_classif(
            features, labels, random_state=RANDOM_STATE
        ),
        k=n_selected
    )
    selector.fit(X_train_imputed, y_train)
    selected_features = X.columns[selector.get_support()].tolist()

    print(f"\nOriginal features: {len(feature_cols)}")
    print(f"Selected features: {len(selected_features)}")
    print("Selected feature names:")
    for feature in selected_features:
        print(f"- {feature}")

    model = make_classifier()
    print(f"\nTraining records: {len(X_train)}")
    print(f"Testing records: {len(X_test)}")
    print(f"Features used by classifier: {len(selected_features)}")
    print(f"Classes: {y.nunique()}")
    print("\nTraining reduced-feature multiclass model...")

    model.fit(X_train[selected_features], y_train)
    predictions = model.predict(X_test[selected_features])
    class_names = sorted(y.unique())
    evaluated_labels = sorted(set(y_test.unique()) | set(predictions))

    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(
        y_test, predictions, labels=evaluated_labels,
        output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_test, predictions, labels=evaluated_labels)

    metrics = {
        "task": "multiclass intrusion detection",
        "accuracy": float(accuracy),
        "accuracy_percent": round(float(accuracy) * 100, 2),
        "macro_precision": float(precision_score(
            y_test, predictions, average="macro", zero_division=0
        )),
        "macro_recall": float(recall_score(
            y_test, predictions, average="macro", zero_division=0
        )),
        "macro_f1": float(f1_score(
            y_test, predictions, average="macro", zero_division=0
        )),
        "weighted_f1": float(f1_score(
            y_test, predictions, average="weighted", zero_division=0
        )),
        "training_records": int(len(X_train)),
        "testing_records": int(len(X_test)),
        "original_features": int(len(feature_cols)),
        "features": int(len(selected_features)),
        "selected_features": selected_features,
        "classes": class_names,
        "test_class_distribution": {
            str(k): int(v) for k, v in y_test.value_counts().items()
        },
        "classification_report": report
    }

    print("\n========== MULTICLASS EVALUATION ==========")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print(f"Weighted F1: {metrics['weighted_f1']:.4f}")
    print("\nPer-class classification report:")
    print(classification_report(
        y_test, predictions, labels=evaluated_labels,
        target_names=evaluated_labels, zero_division=0
    ))
    print("\nConfusion matrix:")
    print(cm)

    classifier = model.named_steps["classifier"]
    importance_df = pd.DataFrame({
        "Feature": selected_features,
        "Importance": classifier.feature_importances_
    }).sort_values("Importance", ascending=False)
    importance_df.to_csv(RESULT_DIR / "feature_importance.csv", index=False)

    model_path = MODEL_DIR / "cloudnids_multiclass.joblib"
    joblib.dump({
        "pipeline": model,
        "feature_cols": selected_features,
        "class_names": class_names,
        "original_feature_count": len(feature_cols),
        "selected_feature_count": len(selected_features),
        "label_mapping": {
            "DoS Hulk": "DoS",
            "DoS GoldenEye": "DoS",
            "DoS slowloris": "DoS",
            "DoS Slowhttptest": "DoS"
        }
    }, model_path)

    metrics_path = RESULT_DIR / "metrics_multiclass.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    pd.DataFrame(cm, index=evaluated_labels, columns=evaluated_labels).to_csv(
        RESULT_DIR / "confusion_matrix_multiclass.csv"
    )

    print("\nSaved multiclass model:", model_path)
    print("Saved metrics:", metrics_path)
    print("Saved confusion matrix:", RESULT_DIR / "confusion_matrix_multiclass.csv")
    print("Saved feature importance:", RESULT_DIR / "feature_importance.csv")
    print(f"Elapsed seconds: {time.time() - start:.2f}")


if __name__ == "__main__":
    main()
