
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "MachineLearningCSV"

saved = joblib.load(
    BASE_DIR / "models" / "cloudnids_binary.joblib"
)

model = saved["pipeline"]
feature_cols = saved["feature_cols"]

csv_files = sorted(DATA_DIR.glob("*.csv"))

if len(csv_files) < 2:
    raise FileNotFoundError("At least two dataset CSV files are required.")

print("Available dataset files:")
for i, file in enumerate(csv_files):
    print(f"{i}: {file.name}")

test_index = int(input("\nEnter the index of the file to test on: "))

test_file = csv_files[test_index]
print(f"\nTesting on: {test_file.name}")

df = pd.read_csv(test_file, low_memory=False)
df.columns = df.columns.str.strip()

label_col = next(
    (col for col in df.columns if col.lower() == "label"),
    None
)

if label_col is None:
    raise ValueError("No Label column found in the selected CSV.")

df = df.rename(columns={label_col: "Label"})
df["Label"] = df["Label"].astype(str).str.strip()

y_true = (
    df["Label"].str.upper() != "BENIGN"
).astype(int)

X = df.drop(
    columns=["Label", "target"],
    errors="ignore"
)

X = X.reindex(columns=feature_cols)
X = X.apply(pd.to_numeric, errors="coerce")
X = X.replace([np.inf, -np.inf], np.nan)

y_pred = model.predict(X)

print("\nActual class counts:")
print(y_true.map({0: "BENIGN", 1: "MALICIOUS"}).value_counts())

print("\nEvaluation results:")
print(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
print(f"Precision: {precision_score(y_true, y_pred, zero_division=0):.4f}")
print(f"Recall:    {recall_score(y_true, y_pred, zero_division=0):.4f}")
print(f"F1 Score:  {f1_score(y_true, y_pred, zero_division=0):.4f}")

print("\nConfusion matrix:")
print(confusion_matrix(y_true, y_pred, labels=[0, 1]))

print("\nClassification report:")
print(classification_report(
    y_true,
    y_pred,
    labels=[0, 1],
    target_names=["BENIGN", "MALICIOUS"],
    zero_division=0
))
