import joblib
import numpy as np
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from app.config import settings
from data_prep import load_data, build_features, split_and_scale

ARTIFACT_DIR = settings.artifact_dir


def load_explainer():
    model = joblib.load(f"{ARTIFACT_DIR}/model.joblib")
    features = joblib.load(f"{ARTIFACT_DIR}/features.joblib")
    explainer = shap.TreeExplainer(model)
    return model, explainer, features


def fraud_shap_values(explainer, data):
    # Returns SHAP values for the fraud class, shape (rows, features)
    shap_values = explainer.shap_values(data)
    if isinstance(shap_values, list):
        return shap_values[1]
    if shap_values.ndim == 3:
        return shap_values[:, :, 1]
    return shap_values


def explain(row_df, explainer, features, top_n=3):
    # row_df: a one-row DataFrame with the same feature columns, already scaled
    values = fraud_shap_values(explainer, row_df)[0]
    order = np.argsort(-np.abs(values))[:top_n]
    return [
        {"feature": features[i], "contribution": round(float(values[i]), 4)}
        for i in order
    ]


if __name__ == "__main__":
    df = load_data()
    X, y = build_features(df)
    X_train, X_test, y_train, y_test, scaler = split_and_scale(X, y)
    model, explainer, features = load_explainer()

    # Explain one fraud case from the test set
    fraud_idx = y_test[y_test == 1].index[0]
    row = X_test.loc[[fraud_idx]]
    print("Top reasons for one fraud transaction:")
    for r in explain(row, explainer, features):
        print(f"  {r['feature']}: {r['contribution']:+.4f}")

    # Save a summary plot for the README
    sample = X_test.sample(500, random_state=42)
    vals = fraud_shap_values(explainer, sample)
    plt.figure()
    shap.summary_plot(vals, sample, show=False)
    plt.tight_layout()
    plt.savefig("docs/shap_summary.png", dpi=150, bbox_inches="tight")
    print("Saved docs/shap_summary.png")