import joblib
import numpy as np
from sklearn.metrics import precision_recall_curve, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split

from data_prep import load_data, build_features, split_and_scale


def pick_threshold(y_true, proba, min_precision=0.80):
    # Lowest threshold that keeps precision at or above min_precision,
    # which maximizes recall under that constraint
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    ok = np.where(precision[:-1] >= min_precision)[0]
    if len(ok) == 0:
        return 0.5
    best = ok[np.argmax(recall[:-1][ok])]
    return thresholds[best]


def report(name, y_true, proba, threshold):
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    print(f"\n{name} at threshold {threshold:.3f}")
    print(f"  precision: {precision_score(y_true, pred, zero_division=0):.3f}")
    print(f"  recall:    {recall_score(y_true, pred):.3f}")
    print(f"  f1:        {f1_score(y_true, pred):.3f}")
    print(f"  caught fraud: {tp}  missed fraud: {fn}  false alarms: {fp}")


if __name__ == "__main__":
    df = load_data()
    X, y = build_features(df)
    X_train, X_test, y_train, y_test, scaler = split_and_scale(X, y)
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
    )

    model = joblib.load("ml/artifacts/v1/model.joblib")

    val_proba = model.predict_proba(X_val)[:, 1]
    threshold = pick_threshold(y_val, val_proba, min_precision=0.80)
    print(f"Chosen threshold from validation set: {threshold:.3f}")

    # Test set is used once, only to report final numbers
    test_proba = model.predict_proba(X_test)[:, 1]
    report("Validation", y_val, val_proba, threshold)
    report("Test (held out)", y_test, test_proba, threshold)
    joblib.dump(float(threshold), "ml/artifacts/v1/threshold.joblib")
    print("Saved threshold to ml/artifacts/v1/threshold.joblib")

