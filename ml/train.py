import joblib
import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import average_precision_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

from data_prep import load_data, build_features, split_and_scale


def evaluate(name, model, X_val, y_val):
    proba = model.predict_proba(X_val)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "model": name,
        "pr_auc": average_precision_score(y_val, proba),
        "precision": precision_score(y_val, pred, zero_division=0),
        "recall": recall_score(y_val, pred),
        "f1": f1_score(y_val, pred),
    }, proba


if __name__ == "__main__":
    df = load_data()
    X, y = build_features(df)
    X_train, X_test, y_train, y_test, scaler = split_and_scale(X, y)

    # Hold out part of the training set for threshold tuning later
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
    )

    models = {
        "logistic_regression": LogisticRegression(class_weight="balanced", max_iter=1000),
        "random_forest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced", n_jobs=-1, random_state=42
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            class_weight="balanced", random_state=42
        ),
    }

    results = []
    fitted = {}
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_fit, y_fit)
        metrics, _ = evaluate(name, model, X_val, y_val)
        results.append(metrics)
        fitted[name] = model

    print("\nValidation results:")
    print(f"{'model':<24}{'pr_auc':>8}{'precision':>11}{'recall':>8}{'f1':>8}")
    for r in sorted(results, key=lambda r: r["pr_auc"], reverse=True):
        print(f"{r['model']:<24}{r['pr_auc']:>8.3f}{r['precision']:>11.3f}{r['recall']:>8.3f}{r['f1']:>8.3f}")

    best_name = max(results, key=lambda r: r["pr_auc"])["model"]
    print(f"\nBest model by PR-AUC: {best_name}")

    os.makedirs("ml/artifacts/v1", exist_ok=True)
    joblib.dump(fitted[best_name], "ml/artifacts/v1/model.joblib")
    joblib.dump(scaler, "ml/artifacts/v1/scaler.joblib")
    joblib.dump(list(X.columns), "ml/artifacts/v1/features.joblib")
    print("Saved model, scaler, and feature list to ml/artifacts/v1/")