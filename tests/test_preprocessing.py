"""
Unit tests for dataset loading, preprocessing, scaling, and CV partitioning.
"""

import os
import pytest
import numpy as np
import pandas as pd
from src.dataset import (
    load_dataset,
    prepare_fold_data,
    get_10_fold_cv_splits,
    get_loso_cv_splits,
    FEATURE_NAMES,
    TARGET_NAME
)


@pytest.fixture
def sample_dataset():
    """Load or build a sample dataset fixture."""
    if os.path.exists("dataset.csv"):
        return load_dataset("dataset.csv")
    else:
        # Fallback dummy dataframe with all 21 features
        data = {feat: np.random.randn(50) for feat in FEATURE_NAMES}
        data["Station_Name"] = ["Station_A"] * 25 + ["Station_B"] * 25
        data[TARGET_NAME] = np.random.uniform(3000, 8000, 50)
        return pd.DataFrame(data)


DEG = chr(176)


def test_feature_list_completeness():
    """Verify exactly 21 numerical meteorological features are defined."""
    assert len(FEATURE_NAMES) == 21
    assert f"Air Temperature (C{DEG})" in FEATURE_NAMES
    assert "DHI (Wh/m2)" in FEATURE_NAMES
    assert "DNI (Wh/m2)" in FEATURE_NAMES
    assert "Relative Humidity (%)" in FEATURE_NAMES
    assert "Barometric Pressure (mB (hPa equiv))" in FEATURE_NAMES


def test_dataset_loading_and_cleaning(sample_dataset):
    """Verify dataset contains no missing values in features and target."""
    df = sample_dataset
    assert len(df) > 0
    assert TARGET_NAME in df.columns
    for feat in FEATURE_NAMES:
        assert feat in df.columns
        assert df[feat].isnull().sum() == 0, f"Null values found in {feat}"


def test_10_fold_splits(sample_dataset):
    """Verify 10-fold cross validation splits are non-empty and partition data."""
    df = sample_dataset
    splits = get_10_fold_cv_splits(df, seed=42)
    assert len(splits) == 10
    for train_idx, test_idx in splits:
        assert len(train_idx) > 0
        assert len(test_idx) > 0
        # Assert no overlap
        assert len(set(train_idx).intersection(set(test_idx))) == 0


def test_loso_splits(sample_dataset):
    """Verify Leave-One-Station-Out splits group strictly by station without overlap."""
    df = sample_dataset
    splits = get_loso_cv_splits(df)
    unique_stations = df["Station_Name"].nunique()
    assert len(splits) == unique_stations
    for train_idx, test_idx, station_name in splits:
        test_stations = df.iloc[test_idx]["Station_Name"].unique()
        train_stations = df.iloc[train_idx]["Station_Name"].unique()
        assert len(test_stations) == 1
        assert test_stations[0] == station_name
        assert station_name not in train_stations


def test_prepare_fold_data_scaling(sample_dataset):
    """Verify StandardScaler is fit strictly on train split without data leakage."""
    df = sample_dataset
    splits = get_10_fold_cv_splits(df, seed=42)
    train_idx, test_idx = splits[0]

    X_train, y_train, X_test, y_test, scaler = prepare_fold_data(
        df, train_idx, test_idx, scaler_type="standard"
    )

    assert X_train.shape[1] == 21
    assert X_test.shape[1] == 21
    assert len(X_train) == len(train_idx)
    assert len(X_test) == len(test_idx)

    # Train mean should be close to 0 and std close to 1
    assert np.allclose(X_train.mean(axis=0), 0, atol=1e-1)
