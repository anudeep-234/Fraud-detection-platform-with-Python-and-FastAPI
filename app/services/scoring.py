import joblib
import numpy as np
import pandas as pd
import shap

from app.config import settings

ARTIFACT_DIR = settings.artifact_dir
MODEL_VERSION = "v1"

# Loaded once when the module is imported (at app startup), not per request
model = joblib.load(f"{ARTIFACT_DIR}/model.joblib")
scaler = joblib.load(f"{ARTIFACT_DIR}/scaler.joblib")
features = joblib.load(f"{ARTIFACT_DIR}/features.joblib")
review_threshold = joblib.load(f"{ARTIFACT_DIR}/threshold.joblib")
block_threshold = 0.5
explainer = shap.TreeExplainer(model)


def build_row(
    payload: dict,
    time_since_last: float,
    tx_count_last_10: float,
    profile_features: dict[str, float] | None = None,
) -> pd.DataFrame:
    row = {f"V{i}": payload["v_features"][f"V{i}"] for i in range(1, 29)}
    row["hour"] = (payload["time"] // 3600) % 24
    row["log_amount"] = np.log1p(payload["amount"])
    row["time_since_last"] = time_since_last
    row["tx_count_last_10"] = tx_count_last_10
    row.update(profile_features or {})
    df = pd.DataFrame([row])[features]
    return pd.DataFrame(scaler.transform(df), columns=features)


def decide(score: float) -> str:
    if score >= block_threshold:
        return "block"
    if score >= review_threshold:
        return "review"
    return "approve"


def score_transaction(
    payload: dict,
    time_since_last: float,
    tx_count_last_10: float,
    profile_features: dict[str, float] | None = None,
) -> dict:
    row = build_row(payload, time_since_last, tx_count_last_10, profile_features)
    risk = float(model.predict_proba(row)[0, 1])

    values = explainer.shap_values(row)
    if isinstance(values, list):
        values = values[1][0]
    elif values.ndim == 3:
        values = values[0, :, 1]
    else:
        values = values[0]

    top = np.argsort(-np.abs(values))[:3]
    reasons = [{"feature": features[i], "contribution": round(float(values[i]), 4)} for i in top]

    return {
        "risk_score": round(risk, 4),
        "decision": decide(risk),
        "reasons": reasons,
        "model_version": MODEL_VERSION,
    }
