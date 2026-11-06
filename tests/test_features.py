import pandas as pd

from ml.data_prep import build_features


def test_build_features_returns_expected_columns():
    df = pd.DataFrame({
        "Time": [0.0, 10.0, 4000.0],
        "Amount": [10.0, 20.0, 30.0],
        "Class": [0, 0, 1],
        **{f"V{i}": [0.1, 0.2, 0.3] for i in range(1, 29)},
    })
    X, y = build_features(df)
    assert X.shape[1] == 32
    assert list(y) == [0, 0, 1]
    assert not X.isnull().any().any()


def test_time_since_last_is_gap_between_rows():
    df = pd.DataFrame({
        "Time": [0.0, 10.0, 25.0],
        "Amount": [1.0, 1.0, 1.0],
        "Class": [0, 0, 0],
        **{f"V{i}": [0.0, 0.0, 0.0] for i in range(1, 29)},
    })
    X, _ = build_features(df)
    assert list(X["time_since_last"]) == [0.0, 10.0, 15.0]
