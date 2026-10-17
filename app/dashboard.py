from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

BASE_DIR = Path(__file__).resolve().parent
API_URL = "http://127.0.0.1:8000"
BATCH_SIZE = 100

st.set_page_config(page_title="CloudNIDS", page_icon="🛡️", layout="wide")
st.title("🛡️ CloudNIDS")
st.caption("Cloud-Based Network Intrusion Detection System")
st.info(
    "Experimental ML model. Predictions can be incorrect, "
    "and this dashboard does not monitor live network traffic."
)

with st.sidebar:
    st.header("Configuration")
    st.write("Inference API:", API_URL)
    uploaded_file = st.file_uploader("Upload CIC-IDS2017 CSV", type=["csv"])
    max_rows = st.number_input(
        "Maximum flows to analyse", min_value=1, max_value=5000,
        value=500, step=100
    )

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file, nrows=int(max_rows), low_memory=False)
        df.columns = df.columns.str.strip()
        label_col = next((c for c in df.columns if c.lower() == "label"), None)

        if label_col:
            df[label_col] = df[label_col].astype(str).str.strip()

        st.subheader("Dataset preview")
        st.write(f"Rows loaded from file: {len(df):,}")
        st.dataframe(df.head(10), use_container_width=True)

        filters = ["All flows"]
        if label_col:
            filters += ["BENIGN only", "Attack only"]

        selected_filter = st.selectbox(
            "Flows to analyse",
            filters,
            help=(
                "Filtering uses the dataset Label column; attack-only excludes BENIGN."
                if label_col else
                "This CSV has no Label column, so ground-truth evaluation is unavailable."
            ),
        )

        if label_col and selected_filter == "BENIGN only":
            analysis_df = df[df[label_col].str.upper() == "BENIGN"].copy()
        elif label_col and selected_filter == "Attack only":
            analysis_df = df[df[label_col].str.upper() != "BENIGN"].copy()
        else:
            analysis_df = df.copy()

        st.write(f"Rows selected for analysis: {len(analysis_df):,}")

        if st.button("Analyse traffic", type="primary"):
            if analysis_df.empty:
                st.error("No rows match this filter. Choose another filter or upload matching rows.")
                st.stop()

            results = []
            progress = st.progress(0)
            status = st.empty()

            for start in range(0, len(analysis_df), BATCH_SIZE):
                chunk = analysis_df.iloc[start:start + BATCH_SIZE]
                flows = []

                for _, row in chunk.iterrows():
                    features = row.drop(
                        labels=[label_col] if label_col else [],
                        errors="ignore",
                    )
                    features = features.replace([np.inf, -np.inf], np.nan)
                    features = features.astype(object).where(features.notna(), None)
                    flows.append({"features": features.to_dict()})

                try:
                    response = requests.post(
                        f"{API_URL}/predict/batch",
                        json={"flows": flows},
                        timeout=120,
                    )
                    response.raise_for_status()
                    results.extend(response.json()["results"])
                except requests.RequestException as exc:
                    if exc.response is not None:
                        st.error(f"API status: {exc.response.status_code}")
                        st.code(exc.response.text)
                    else:
                        st.error(f"Prediction request failed: {exc}")
                    st.stop()

                completed = min(start + len(chunk), len(analysis_df))
                progress.progress(completed / len(analysis_df))
                status.write(f"Analysed {completed:,} of {len(analysis_df):,} flows")

            result_df = pd.DataFrame(results).reset_index(drop=True)
            output_df = analysis_df.reset_index(drop=True).copy()
            output_df["Predicted Class"] = result_df["prediction"]
            output_df["Malicious Probability"] = result_df["malicious_probability"]
            output_df["Benign Probability"] = result_df["benign_probability"]

            st.session_state["cloudnids_results"] = output_df
            st.session_state["cloudnids_label_col"] = label_col
            st.session_state["cloudnids_filter"] = selected_filter

    except (pd.errors.ParserError, UnicodeDecodeError, ValueError) as exc:
        st.error(f"Could not read the uploaded CSV: {exc}")

if "cloudnids_results" in st.session_state:
    output_df = st.session_state["cloudnids_results"]
    label_col = st.session_state["cloudnids_label_col"]
    saved_filter = st.session_state.get("cloudnids_filter", "All flows")

    st.divider()
    st.subheader("Prediction summary")
    st.caption(f"Filter used for this analysis: {saved_filter}")

    total = len(output_df)
    malicious = int((output_df["Predicted Class"].astype(str).str.upper() == "MALICIOUS").sum())
    benign = total - malicious

    col1, col2, col3 = st.columns(3)
    col1.metric("Flows analysed", f"{total:,}")
    col2.metric("Predicted benign", f"{benign:,}")
    col3.metric("Predicted malicious", f"{malicious:,}")

    chart_df = pd.DataFrame({"Prediction": ["BENIGN", "MALICIOUS"], "Count": [benign, malicious]})
    st.plotly_chart(
        px.bar(chart_df, x="Prediction", y="Count", color="Prediction",
               title="Predicted traffic distribution"),
        use_container_width=True,
    )

    st.subheader("Prediction results")
    st.dataframe(output_df, use_container_width=True)
    st.download_button(
        "Download prediction results (CSV)",
        data=output_df.to_csv(index=False).encode("utf-8"),
        file_name="cloudnids_predictions.csv",
        mime="text/csv",
    )

    if label_col and label_col in output_df.columns:
        st.divider()
        st.subheader("Evaluation against dataset labels")
        st.caption(
            "These metrics compare predictions with labels in this uploaded sample; "
            "they are not an independent test of the model."
        )

        actual_labels = output_df[label_col].astype(str).str.strip().str.upper()
        actual_binary = (actual_labels != "BENIGN").astype(int)
        predicted_labels = output_df["Predicted Class"].astype(str).str.strip().str.upper()
        predicted_binary = (predicted_labels == "MALICIOUS").astype(int)

        cm = confusion_matrix(actual_binary, predicted_binary, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy", f"{accuracy_score(actual_binary, predicted_binary):.2%}")
        m2.metric("Precision", f"{precision_score(actual_binary, predicted_binary, zero_division=0):.2%}")
        m3.metric("Recall", f"{recall_score(actual_binary, predicted_binary, zero_division=0):.2%}")
        m4.metric("F1-score", f"{f1_score(actual_binary, predicted_binary, zero_division=0):.2%}")

        cm_df = pd.DataFrame(
            cm,
            index=["Actual BENIGN", "Actual MALICIOUS"],
            columns=["Predicted BENIGN", "Predicted MALICIOUS"],
        )
        st.markdown("**Confusion matrix**")
        st.dataframe(cm_df, use_container_width=True)
        st.write(
            f"True negatives: **{tn:,}** · False positives: **{fp:,}** · "
            f"False negatives: **{fn:,}** · True positives: **{tp:,}**"
        )

        st.markdown("**Results by original dataset label**")
        per_label = pd.DataFrame({
            "Dataset label": actual_labels,
            "Predicted malicious": predicted_binary,
        })
        per_label_summary = per_label.groupby("Dataset label").agg(
            Flows=("Predicted malicious", "size"),
            **{"Predicted malicious": ("Predicted malicious", "sum")},
        ).reset_index()
        per_label_summary["Predicted benign"] = (
            per_label_summary["Flows"] - per_label_summary["Predicted malicious"]
        )
        per_label_summary["Detection rate"] = (
            per_label_summary["Predicted malicious"] / per_label_summary["Flows"]
        ).map(lambda value: f"{value:.2%}")
        st.dataframe(per_label_summary, use_container_width=True)

        st.markdown("**Dataset labels vs predicted class**")
        comparison = pd.DataFrame({
            "Dataset label": actual_labels,
            "Predicted class": predicted_labels,
        })
        st.dataframe(
            pd.crosstab(comparison["Dataset label"], comparison["Predicted class"]),
            use_container_width=True,
        )

        if actual_binary.nunique() < 2:
            st.warning(
                "This sample contains only one actual binary class. Metrics may not "
                "show how well the model distinguishes benign from malicious traffic. "
                "Try Attack only or All flows with both classes represented."
            )
else:
    st.write("Upload a CIC-IDS2017 CSV to preview the data and run predictions.")
