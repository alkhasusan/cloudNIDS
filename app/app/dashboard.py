
import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

API_URL = "http://127.0.0.1:8002"
BATCH_SIZE = 100
CHUNK_SIZE = 50000

st.set_page_config(
    page_title="CloudNIDS Multiclass",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ CloudNIDS")
st.caption("Cloud-Based Multiclass Network Intrusion Detection System")
st.info(
    "Experimental ML model. Predictions can be incorrect. "
    "This dashboard analyses sampled CSV traffic, not live traffic."
)

with st.sidebar:
    st.header("Configuration")
    st.write("Inference API:", API_URL)

    uploaded_file = st.file_uploader(
        "Upload CIC-IDS2017 CSV",
        type=["csv"],
    )

    max_rows = st.number_input(
        "Maximum flows to analyse",
        min_value=100,
        max_value=5000,
        value=5000,
        step=500,
    )

    st.caption(
        "Rows are sampled across the complete CSV to reduce "
        "the risk of testing only its first, mostly benign rows."
    )


def normalize_label(label):
    """Map CIC-IDS2017 labels to the trained model's class names."""
    value = str(label).strip()
    key = value.casefold()
    key = key.replace("\ufffd", "-")
    key = key.replace("\u2013", "-").replace("\u2014", "-")
    key = " ".join(key.split())

    mapping = {
        "benign": "BENIGN",
        "ddos": "DDoS",
        "dos hulk": "DoS",
        "dos goldeneye": "DoS",
        "dos slowloris": "DoS",
        "dos slowhttptest": "DoS",
        "portscan": "PortScan",
        "ftp-patator": "FTP-Patator",
        "ssh-patator": "SSH-Patator",
        "bot": "Bot",
        "infiltration": "Infiltration",
        "heartbleed": "Heartbleed",
        "web attack - brute force": "Web Attack \ufffd Brute Force",
        "web attack - sql injection": "Web Attack \ufffd Sql Injection",
        "web attack - xss": "Web Attack \ufffd XSS",
    }

    return mapping.get(key, value)



def load_sample(file, sample_size):
    """Create a reproducible sample that includes rare attack classes."""
    file.seek(0)

    chunks = []
    total_rows = 0

    # Read the CSV in chunks and count all rows.
    for chunk in pd.read_csv(
        file,
        chunksize=CHUNK_SIZE,
        low_memory=False,
    ):
        chunk.columns = chunk.columns.str.strip()
        total_rows += len(chunk)
        chunks.append(chunk)

    if not chunks:
        return pd.DataFrame(), 0

    df = pd.concat(chunks, ignore_index=True)

    # Find the actual label column.
    label_col = next(
        (col for col in df.columns if col.strip().casefold() == "label"),
        None,
    )

    # If labels are unavailable, use ordinary random sampling.
    if label_col is None:
        sample = df.sample(
            n=min(sample_size, len(df)),
            random_state=42,
        )
        return sample.reset_index(drop=True), total_rows

    # Sample each class separately so rare attacks are included.
    groups = list(df.groupby(label_col, dropna=False, sort=False))
    if not groups:
        return df.head(sample_size).reset_index(drop=True), total_rows

    per_class = max(1, sample_size // len(groups))
    samples = []

    for _, group in groups:
        samples.append(
            group.sample(
                n=min(per_class, len(group)),
                random_state=42,
            )
        )

    result = pd.concat(samples, axis=0)

    # Fill any remaining sample slots without duplicating rows.
    remaining_count = sample_size - len(result)

    if remaining_count > 0:
        remaining_df = df.loc[~df.index.isin(result.index)]

        if not remaining_df.empty:
            extra = remaining_df.sample(
                n=min(remaining_count, len(remaining_df)),
                random_state=42,
            )
            result = pd.concat([result, extra], axis=0)

    result = result.sample(frac=1, random_state=42).reset_index(drop=True)

    return result, total_rows


if uploaded_file is not None:
    if st.session_state.get("cloudnids_filename") != uploaded_file.name:
        st.session_state.pop("cloudnids_results", None)
        st.session_state.pop("cloudnids_label_col", None)
        st.session_state["cloudnids_filename"] = uploaded_file.name

    try:
        with st.spinner("Reading and sampling the complete CSV..."):
            df, total_file_rows = load_sample(
                uploaded_file,
                int(max_rows),
            )

        label_col = next(
            (col for col in df.columns if col.casefold() == "label"),
            None,
        )

        st.subheader("Dataset preview")
        st.write(f"Total rows in uploaded CSV: {total_file_rows:,}")
        st.write(f"Rows sampled for analysis: {len(df):,}")
        st.dataframe(df.head(10), use_container_width=True)

        if label_col:
            df[label_col] = df[label_col].astype(str).str.strip()

            st.subheader("Actual labels in sampled rows")
            label_counts = (
                df[label_col]
                .value_counts(dropna=False)
                .rename_axis("Actual Label")
                .reset_index(name="Number of Rows")
            )
            st.dataframe(label_counts, use_container_width=True)

            normalized = df[label_col].map(normalize_label)
            df["_CloudNIDS_Normalized_Label"] = normalized

            filters = ["All flows", "BENIGN only", "Attack only"]
        else:
            st.warning(
                "No Label column found. Predictions can run, "
                "but accuracy and F1 evaluation are unavailable."
            )
            filters = ["All flows"]

        selected_filter = st.selectbox(
            "Flows to analyse",
            filters,
        )

        if label_col and selected_filter == "BENIGN only":
            analysis_df = df[
                df["_CloudNIDS_Normalized_Label"] == "BENIGN"
            ].copy()
        elif label_col and selected_filter == "Attack only":
            analysis_df = df[
                df["_CloudNIDS_Normalized_Label"] != "BENIGN"
            ].copy()
        else:
            analysis_df = df.copy()

        st.write(f"Rows selected: {len(analysis_df):,}")

        if st.button("Analyse traffic", type="primary"):
            if analysis_df.empty:
                st.error("No rows match the selected filter.")
                st.stop()

            results = []
            progress = st.progress(0)
            status = st.empty()

            # Do not send the dataset label to the prediction API.
            excluded_cols = [label_col] if label_col else []
            excluded_cols.append("_CloudNIDS_Normalized_Label")

            for start in range(0, len(analysis_df), BATCH_SIZE):
                chunk = analysis_df.iloc[start:start + BATCH_SIZE]
                flows = []

                for _, row in chunk.iterrows():
                    features = row.drop(
                        labels=excluded_cols,
                        errors="ignore",
                    )
                    features = features.replace(
                        [np.inf, -np.inf], np.nan
                    )
                    features = features.astype(object).where(
                        features.notna(), None
                    )
                    flows.append({"features": features.to_dict()})

                try:
                    response = requests.post(
                        f"{API_URL}/predict/batch",
                        json={"flows": flows},
                        timeout=180,
                    )
                    response.raise_for_status()
                    batch_results = response.json()["results"]

                    if len(batch_results) != len(flows):
                        st.error(
                            "API returned a different number of "
                            "predictions than input flows."
                        )
                        st.stop()

                    results.extend(batch_results)

                except requests.RequestException as exc:
                    if exc.response is not None:
                        st.error(
                            f"API error {exc.response.status_code}"
                        )
                        st.code(exc.response.text)
                    else:
                        st.error(f"Could not contact API: {exc}")
                    st.stop()

                completed = min(
                    start + len(chunk),
                    len(analysis_df),
                )
                progress.progress(completed / len(analysis_df))
                status.write(
                    f"Analysed {completed:,} of "
                    f"{len(analysis_df):,} flows"
                )

            result_df = pd.DataFrame(results).reset_index(drop=True)
            output_df = analysis_df.reset_index(drop=True).copy()

            output_df["Predicted Class"] = result_df["prediction"]
            output_df["Confidence"] = result_df["confidence"]

            probabilities = pd.DataFrame(
                result_df["class_probabilities"].tolist()
            ).add_prefix("Probability: ")

            output_df = pd.concat(
                [output_df, probabilities],
                axis=1,
            )

            st.session_state["cloudnids_results"] = output_df
            st.session_state["cloudnids_label_col"] = label_col
            st.session_state["cloudnids_filter"] = selected_filter
            st.session_state["cloudnids_filename"] = uploaded_file.name

    except (
        pd.errors.ParserError,
        UnicodeDecodeError,
        ValueError,
        OSError,
    ) as exc:
        st.error(f"Could not read the uploaded CSV: {exc}")


if "cloudnids_results" in st.session_state:
    output_df = st.session_state["cloudnids_results"]
    label_col = st.session_state.get("cloudnids_label_col")
    saved_filter = st.session_state.get(
        "cloudnids_filter", "All flows"
    )

    st.divider()
    st.subheader("Multiclass prediction summary")
    st.caption(f"Filter used: {saved_filter}")

    total = len(output_df)
    counts = output_df["Predicted Class"].value_counts()
    benign_count = int(counts.get("BENIGN", 0))
    attack_count = total - benign_count

    c1, c2, c3 = st.columns(3)
    c1.metric("Flows analysed", f"{total:,}")
    c2.metric("Predicted BENIGN", f"{benign_count:,}")
    c3.metric("Predicted attack flows", f"{attack_count:,}")

    distribution = (
        counts.rename_axis("Predicted Class")
        .reset_index(name="Count")
    )

    st.plotly_chart(
        px.bar(
            distribution,
            x="Predicted Class",
            y="Count",
            color="Predicted Class",
            title="Predicted traffic by class",
        ),
        use_container_width=True,
    )

    st.plotly_chart(
        px.histogram(
            output_df,
            x="Confidence",
            nbins=20,
            title="Prediction confidence distribution",
        ),
        use_container_width=True,
    )

    st.subheader("Prediction results")
    st.dataframe(output_df, use_container_width=True)

    st.download_button(
        "Download prediction results (CSV)",
        data=output_df.to_csv(index=False).encode("utf-8"),
        file_name="cloudnids_multiclass_predictions.csv",
        mime="text/csv",
    )

    if label_col and label_col in output_df.columns:
        st.divider()
        st.subheader("Evaluation against dataset labels")
        st.caption(
            "These scores describe only the sampled rows and are not "
            "an independent test-set evaluation."
        )

        actual = output_df[label_col].map(normalize_label)
        predicted = output_df["Predicted Class"].astype(str)
        labels = sorted(set(actual) | set(predicted))

        accuracy = accuracy_score(actual, predicted)
        macro_f1 = f1_score(
            actual, predicted,
            labels=labels,
            average="macro",
            zero_division=0,
        )
        weighted_f1 = f1_score(
            actual, predicted,
            labels=labels,
            average="weighted",
            zero_division=0,
        )

        m1, m2, m3 = st.columns(3)
        m1.metric("Accuracy", f"{accuracy:.2%}")
        m2.metric("Macro F1-score", f"{macro_f1:.2%}")
        m3.metric("Weighted F1-score", f"{weighted_f1:.2%}")

        if actual.nunique() < 2:
            st.warning(
                "This sample contains only one actual class. "
                "These metrics do not demonstrate detection across "
                "multiple attack types."
            )

        st.markdown("**Classification report**")
        report = classification_report(
            actual,
            predicted,
            labels=labels,
            output_dict=True,
            zero_division=0,
        )
        st.dataframe(
            pd.DataFrame(report).transpose(),
            use_container_width=True,
        )

        st.markdown("**Confusion matrix**")
        matrix = confusion_matrix(actual, predicted, labels=labels)
        matrix_df = pd.DataFrame(
            matrix,
            index=[f"Actual: {name}" for name in labels],
            columns=[f"Predicted: {name}" for name in labels],
        )
        st.dataframe(matrix_df, use_container_width=True)

        st.markdown("**Actual vs predicted classes**")
        comparison = pd.crosstab(
            actual.rename("Actual class"),
            predicted.rename("Predicted class"),
        )
        st.dataframe(comparison, use_container_width=True)
        st.write("Unique raw labels:", sorted(df[label_col].unique().tolist()))
        st.write("Normalized label counts:")
        st.write(df["_CloudNIDS_Normalized_Label"].value_counts(dropna=False))
else:
    st.write(
        "Upload a CIC-IDS2017 CSV to preview the data "
        "and run multiclass predictions."
    )
