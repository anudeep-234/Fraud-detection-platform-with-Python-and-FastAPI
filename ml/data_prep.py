import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(path="data/creditcard.csv"):
    df = pd.read_csv(path)
    # Time is seconds from the first transaction, so sort by it
    df = df.sort_values("Time").reset_index(drop=True)
    return df


def build_features(df):
    df = df.copy()

    # Hour of day from elapsed seconds (the dataset spans about two days)
    df["hour"] = (df["Time"] // 3600) % 24

    # Log-transform the amount, since it is heavily right-skewed
    df["log_amount"] = np.log1p(df["Amount"])

    # Time since the previous transaction
    df["time_since_last"] = df["Time"].diff().fillna(0)

    # Count of transactions in the previous 10 rows (a simple velocity proxy)
    df["tx_count_last_10"] = (
        pd.Series(1, index=df.index).shift(1).rolling(10, min_periods=1).sum().fillna(0)
    )

    feature_cols = [c for c in df.columns if c.startswith("V")] + [
        "hour", "log_amount", "time_since_last", "tx_count_last_10"
    ]
    X = df[feature_cols]
    y = df["Class"]
    return X, y


def split_and_scale(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns, index=X_train.index)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=X.columns, index=X_test.index)
    return X_train, X_test, y_train, y_test, scaler


if __name__ == "__main__":
    df = load_data()
    X, y = build_features(df)
    X_train, X_test, y_train, y_test, scaler = split_and_scale(X, y)
    print("Features:", X.shape[1])
    print("Train fraud rate:", round(y_train.mean(), 5))
    print("Test fraud rate:", round(y_test.mean(), 5))