
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "cloudnids_binary_tuned.joblib"
DATA_DIR = BASE_DIR / "data" / "MachineLearningCSV"

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}. Run tune_threshold.py first."
    )

saved = joblib.load(MODEL_PATH)

model = saved["pipeline"]
feature_cols = saved["feature_cols"]
threshold = float(saved.get("threshold", 0.5))

csv_files = sorted(DATA_DIR.glob("*.csv"))

if not csv_files:
    raise FileNotFoundError(f"No dataset CSV files found in {DATA_DIR}")

print("Available CSV files:")
for i, path in enumerate(csv_files):
    print(f"{i}: {path.name}")

file_index = int(input("\nEnter CSV number to test: "))

if file_index < 0 or file_index >= len(csv_files):
    raise ValueError("Invalid CSV selection.")

df = pd.read_csv(csv_files[file_index], low_memory=False)
df.columns = df.columns.str.strip()

if df.empty:
    raise ValueError("Selected CSV contains no rows.")

print(f"\nSelected file: {csv_files[file_index].name}")
print(f"Rows available: {len(df):,}")

row_index = int(input("Enter row number to predict (starting at 0): "))

if row_index < 0 or row_index >= len(df):
    raise ValueError("Invalid row number.")

row = df.iloc[[row_index]].copy()

actual_label = (
    str(row["Label"].iloc[0]).strip()
    if "Label" in row.columns
    else "Unknown"
)

row = row.drop(
    columns=[
        "Label",
        "target",
        "Flow ID",
        "Source IP",
        "Destination IP",
        "Timestamp",
    ],
    errors="ignore",
)

row = row.reindex(columns=feature_cols)

for col in row.columns:
    row[col] = pd.to_numeric(row[col], errors="coerce")

row = row.replace([np.inf, -np.inf], np.nan)

probabilities = model.predict_proba(row)[0]
classes = list(model.named_steps["model"].classes_)

malicious_index = classes.index(1)
benign_index = classes.index(0)

malicious_probability = float(probabilities[malicious_index])
benign_probability = float(probabilities[benign_index])

prediction = (
    1 if malicious_probability >= threshold else 0
)

prediction_name = "MALICIOUS" if prediction == 1 else "BENIGN"

print("\n=== CloudNIDS Prediction ===")
print("Predicted class:", prediction_name)
print(f"Malicious probability: {malicious_probability:.4f}")
print(f"Benign probability:    {benign_probability:.4f}")
print(f"Decision threshold:    {threshold:.2f}")
print("Dataset label:", actual_label)
print(
    "Label agrees with prediction:",
    (
        prediction_name == (
            "BENIGN"
            if actual_label.upper() == "BENIGN"
            else "MALICIOUS"
        )
    ) if actual_label != "Unknown" else "Not available",
)
