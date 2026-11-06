import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split

# Make the project root importable so "app" can be found when run as ml/retrain.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_prep import load_data, build_features, split_and_scale
from evaluate import pick_threshold
from app.database import SessionLocal
from app import models_db as m

V1_DIR = "ml/artifacts/v1"
V2_DIR = "ml/artifacts/v2"


def load_reviewed_rows(feature_names):
    """Turn reviewed alerts into raw feature rows and labels (1 = fraud, 0 = legitimate)."""
    db = SessionLocal()
    try:
        rows = (
            db.query(m.Review, m.Transaction)
            .join(m.Alert, m.Review.alert_id == m.Alert.id)
            .join(m.Prediction, m.Alert.prediction_id == m.Prediction.id)
            .join(m.Transaction, m.Prediction.transaction_id == m.Transaction.id)
            .all()
        )
    finally:
        db.close()

    records, labels = [], []
    for review, tx in rows:
        if not tx.features or "time_since_last" not in tx.features:
            continue
        f = json.loads(tx.features)
        row = {f"V{i}": f["v_features"][f"V{i}"] for i in range(1, 29)}
        row["hour"] = (f["time"] // 3600) % 24
        row["log_amount"] = np.log1p(f["amount"])
        row["time_since_last"] = f["time_since_last"]
        # Stored as tx_count_last_hour; the model's feature name is tx_count_last_10
        row["tx_count_last_10"] = f["tx_count_last_hour"]
        records.append(row)
        labels.append(1 if review.label == "fraud" else 0)

    X_raw = pd.DataFrame(records, columns=feature_names)
    y = pd.Series(labels, dtype=int)
    return X_raw, y


if __name__ == "__main__":
    features = joblib.load(f"{V1_DIR}/features.joblib")
    model_v1 = joblib.load(f"{V1_DIR}/model.joblib")

    # Same split and seed as v1, so the test set is identical for both models
    df = load_data()
    X, y = build_features(df)
    X_train, X_test, y_train, y_test, scaler = split_and_scale(X, y)
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
    )

    X_db_raw, y_db = load_reviewed_rows(features)
    print(f"Reviewed rows available: {len(y_db)} "
          f"(fraud: {int(y_db.sum())}, legitimate: {int((y_db == 0).sum())})")
    if len(y_db) == 0:
        print("No reviewed transactions yet. Nothing to retrain on.")
        sys.exit(0)

    X_db = pd.DataFrame(scaler.transform(X_db_raw), columns=features)
    X_new = pd.concat([X_fit, X_db], ignore_index=True)
    y_new = pd.concat([y_fit, y_db], ignore_index=True)

    model_v2 = RandomForestClassifier(
        n_estimators=200, class_weight="balanced", n_jobs=-1, random_state=42
    )
    print("Training v2 on Kaggle training data plus reviewed rows...")
    model_v2.fit(X_new, y_new)

    # Threshold chosen on the validation split, which only contains Kaggle data
    threshold_v2 = pick_threshold(y_val, model_v2.predict_proba(X_val)[:, 1], min_precision=0.80)

    # Compare both models on the same untouched test set
    pr_v1 = average_precision_score(y_test, model_v1.predict_proba(X_test)[:, 1])
    pr_v2 = average_precision_score(y_test, model_v2.predict_proba(X_test)[:, 1])
    print(f"Test PR-AUC  v1: {pr_v1:.4f}   v2: {pr_v2:.4f}")

    MIN_REVIEWS = 200
    MIN_IMPROVEMENT = 0.01
    if len(y_db) >= MIN_REVIEWS and pr_v2 >= pr_v1 + MIN_IMPROVEMENT:
        os.makedirs(V2_DIR, exist_ok=True)
        joblib.dump(model_v2, f"{V2_DIR}/model.joblib")
        joblib.dump(scaler, f"{V2_DIR}/scaler.joblib")
        joblib.dump(features, f"{V2_DIR}/features.joblib")
        joblib.dump(float(threshold_v2), f"{V2_DIR}/threshold.joblib")
        print(f"v2 is better. Saved to {V2_DIR}/ with threshold {threshold_v2:.3f}")
    else:
        print("Promotion blocked: needs at least MIN_REVIEWS labels and MIN_IMPROVEMENT gain. Keeping v1.")




