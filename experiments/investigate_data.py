
import os
import glob
import numpy as np
import pandas as pd

DATA_DIR = "data/MachineLearningCSV"
HELDOUT_FILE = "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
SAMPLE_SIZE = 20000
RANDOM_STATE = 42

def load_csv(path):
    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.strip()
    df["Label"] = df["Label"].astype(str).str.strip()
    return df

def prepare_features(df):
    excluded = [
        "Label", "Flow ID", "Source IP",
        "Destination IP", "Timestamp"
    ]

    X = df.drop(
        columns=[c for c in excluded if c in df.columns]
    ).copy()

    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    X = X.replace([np.inf, -np.inf], np.nan)
    return X

def main():
    heldout_path = os.path.join(DATA_DIR, HELDOUT_FILE)

    print("Loading held-out DDoS data...")
    heldout = load_csv(heldout_path)
    heldout_X = prepare_features(heldout)

    print("\nHeld-out label distribution:")
    print(heldout["Label"].value_counts().to_string())

    print("\nLoading training data...")
    training_parts = []

    for path in glob.glob(os.path.join(DATA_DIR, "*.csv")):
        if os.path.basename(path) == HELDOUT_FILE:
            continue

        df = load_csv(path)

        if len(df) > SAMPLE_SIZE:
            df = df.sample(
                n=SAMPLE_SIZE,
                random_state=RANDOM_STATE
            )

        training_parts.append(df)
        print(f"{os.path.basename(path)}: {len(df)} sampled rows")

    training = pd.concat(training_parts, ignore_index=True)
    training_X = prepare_features(training)

    common = training_X.columns.intersection(heldout_X.columns)
    training_X = training_X[common]
    heldout_X = heldout_X[common]

    training_y = (
        training["Label"].str.upper() != "BENIGN"
    ).astype(int)

    heldout_y = (
        heldout["Label"].str.upper() != "BENIGN"
    ).astype(int)

    print("\n=== DATA QUALITY ===")
    print(f"Training rows: {len(training_X)}")
    print(f"Held-out rows: {len(heldout_X)}")
    print(f"Common features: {len(common)}")

    print("\nMissing-value percentage: largest training features")
    print(
        (training_X.isna().mean() * 100)
        .sort_values(ascending=False)
        .head(15).round(2).to_string()
    )

    constant = [
        c for c in common
        if training_X[c].nunique(dropna=True) <= 1
    ]
    print(f"\nConstant or all-missing training features: {len(constant)}")
    print(constant)

    print("\n=== BINARY CLASS DISTRIBUTION ===")
    print("Training:")
    print(training_y.value_counts().rename(
        index={0: "BENIGN", 1: "MALICIOUS"}
    ).to_string())

    print("\nHeld-out:")
    print(heldout_y.value_counts().rename(
        index={0: "BENIGN", 1: "MALICIOUS"}
    ).to_string())

    print("\n=== FEATURE DISTRIBUTION COMPARISON ===")
    rows = []

    for col in common:
        tr = training_X[col].replace([np.inf, -np.inf], np.nan).dropna()
        te = heldout_X[col].replace([np.inf, -np.inf], np.nan).dropna()

        if tr.empty or te.empty:
            continue

        train_median = tr.median()
        test_median = te.median()
        train_std = tr.std()

        median_shift = abs(test_median - train_median) / (
            abs(train_std) + 1e-9
        )

        rows.append({
            "feature": col,
            "train_median": train_median,
            "heldout_median": test_median,
            "median_shift_in_train_std": median_shift,
            "train_missing_pct": training_X[col].isna().mean() * 100,
            "heldout_missing_pct": heldout_X[col].isna().mean() * 100
        })

    comparison = pd.DataFrame(rows).sort_values(
        "median_shift_in_train_std", ascending=False
    )

    os.makedirs("results", exist_ok=True)
    comparison.to_csv(
        "results/feature_distribution_comparison.csv",
        index=False
    )

    print("\nFeatures with the largest median shifts:")
    print(comparison.head(20).round(3).to_string(index=False))

    print("\nSaved results/feature_distribution_comparison.csv")

if __name__ == "__main__":
    main()
