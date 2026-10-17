
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "cloudnids_binary_tuned.joblib"

saved = joblib.load(MODEL_PATH)
model = saved["pipeline"]
feature_cols = saved["feature_cols"]
threshold = float(saved.get("threshold", 0.5))

classes = list(model.named_steps["model"].classes_)
benign_index = classes.index(0)
malicious_index = classes.index(1)

app = FastAPI(title="CloudNIDS API")


class FlowData(BaseModel):
    features: dict[str, float | int | None]


class BatchFlowData(BaseModel):
    flows: list[FlowData] = Field(min_length=1, max_length=100)


def predict_flows(flows):
    rows = []

    for flow in flows:
        if not flow.features:
            raise HTTPException(
                status_code=400,
                detail="Each flow must contain at least one feature."
            )

        row = pd.DataFrame([flow.features])
        row = row.reindex(columns=feature_cols)
        row = row.apply(pd.to_numeric, errors="coerce")
        row = row.replace([np.inf, -np.inf], np.nan)
        rows.append(row.iloc[0])

    data = pd.DataFrame(rows, columns=feature_cols)
    probabilities = model.predict_proba(data)

    results = []

    for probability in probabilities:
        malicious = float(probability[malicious_index])
        benign = float(probability[benign_index])

        results.append({
            "prediction": (
                "MALICIOUS" if malicious >= threshold else "BENIGN"
            ),
            "malicious_probability": round(malicious, 4),
            "benign_probability": round(benign, 4),
            "threshold": threshold,
            "warning": "Experimental model; predictions may be incorrect."
        })

    return results


@app.get("/")
def home():
    return {
        "project": "CloudNIDS",
        "status": "running",
        "feature_count": len(feature_cols)
    }


@app.post("/predict")
def predict(flow: FlowData):
    return predict_flows([flow])[0]


@app.post("/predict/batch")
def predict_batch(batch: BatchFlowData):
    return {"results": predict_flows(batch.flows)}
