
from pathlib import Path
from typing import Optional, Union

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "cloudnids_multiclass.joblib"

saved = joblib.load(MODEL_PATH)
model = saved["pipeline"]
feature_cols = saved["feature_cols"]
class_names = list(saved["class_names"])

app = FastAPI(title="CloudNIDS Multiclass API")


class FlowData(BaseModel):
    features: dict[str, Optional[Union[float, int]]]


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
    predictions = model.predict(data)

    results = []

    for prediction, probability in zip(predictions, probabilities):
        class_probabilities = {
            str(name): round(float(probability[index]), 6)
            for index, name in enumerate(model.classes_)
        }

        results.append({
            "prediction": str(prediction),
            "confidence": round(float(np.max(probability)), 4),
            "class_probabilities": class_probabilities,
            "warning": (
                "Experimental model; predictions may be incorrect. "
                "Not validated for real-world security monitoring."
            )
        })

    return results


@app.get("/")
def home():
    return {
        "project": "CloudNIDS",
        "status": "running",
        "classification": "multiclass",
        "class_count": len(class_names),
        "classes": class_names,
        "feature_count": len(feature_cols)
    }


@app.post("/predict")
def predict(flow: FlowData):
    return predict_flows([flow])[0]


@app.post("/predict/batch")
def predict_batch(batch: BatchFlowData):
    return {"results": predict_flows(batch.flows)}
