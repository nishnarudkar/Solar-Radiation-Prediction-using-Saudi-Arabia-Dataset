"""
Dataset loading, preprocessing, scaling, and cross-validation partitioners.
Aligned with Springer Nature research paper specifications (10-fold CV and 43-fold LOSO CV).
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Tuple, List, Generator, Optional
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import KFold, LeaveOneGroupOut

DEG = chr(176)

# 21 Numerical Meteorological Features (Table 1 of the paper)
FEATURE_NAMES = [
    f"Air Temperature (C{DEG})",
    f"Air Temperature Uncertainty (C{DEG})",
    f"Wind Direction at 3m ({DEG}N)",
    f"Wind Direction at 3m Uncertainty ({DEG}N)",
    "Wind Speed at 3m (m/s)",
    "Wind Speed at 3m Uncertainty (m/s)",
    "Wind Speed at 3m (std dev) (m/s)",
    "DHI (Wh/m2)",
    "DHI Uncertainty (Wh/m2)",
    "Standard Deviation DHI (Wh/m2)",
    "DNI (Wh/m2)",
    "DNI Uncertainty (Wh/m2)",
    "Standard Deviation DNI (Wh/m2)",
    "GHI Uncertainty (Wh/m2)",
    "Standard Deviation GHI (Wh/m2)",
    "Peak Wind Speed at 3m (m/s)",
    "Peak Wind Speed at 3m Uncertainty (m/s)",
    "Relative Humidity (%)",
    "Relative Humidity Uncertainty (%)",
    "Barometric Pressure (mB (hPa equiv))",
    "Barometric Pressure Uncertainty (mB (hPa equiv))"
]

TARGET_NAME = "GHI (Wh/m2)"


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize any non-standard degree symbols or double-encoded characters in headers."""
    col_map = {}
    for col in df.columns:
        clean = col.replace(chr(194) + chr(176), chr(176)).replace("Â°", chr(176))
        col_map[col] = clean
    return df.rename(columns=col_map)


def load_dataset(csv_path: str = "dataset.csv") -> pd.DataFrame:
    """
    Load and clean dataset from CSV.
    Imputes missing values with feature median as specified in Section 3.2 of the paper.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset file not found at: {csv_path}")

    try:
        df = pd.read_csv(csv_path, encoding="utf-8")
    except Exception:
        df = pd.read_csv(csv_path, encoding="latin-1")

    df = clean_column_names(df)

    # Impute missing values for numerical features using column median
    for col in FEATURE_NAMES:
        if col in df.columns and df[col].isnull().sum() > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)

    # Drop rows missing the target variable if any
    if TARGET_NAME in df.columns:
        df = df.dropna(subset=[TARGET_NAME])

    return df


class SolarDataset(Dataset):
    """PyTorch Dataset for solar radiation feature rows."""

    def __init__(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        self.X = torch.tensor(X, dtype=torch.float32)
        if y is not None:
            self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)
        else:
            self.y = None

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]


def get_10_fold_cv_splits(
    df: pd.DataFrame,
    seed: int = 42
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Generate 10-fold cross-validation index splits."""
    kf = KFold(n_splits=10, shuffle=True, random_state=seed)
    return list(kf.split(df))


def get_loso_cv_splits(
    df: pd.DataFrame
) -> List[Tuple[np.ndarray, np.ndarray, str]]:
    """
    Generate Leave-One-Station-Out (LOSO) 43-fold cross-validation splits.
    Yields (train_indices, test_indices, excluded_station_name).
    """
    logo = LeaveOneGroupOut()
    groups = df["Station_Name"].values
    splits = []
    for train_idx, test_idx in logo.split(df, groups=groups):
        station_name = df.iloc[test_idx[0]]["Station_Name"]
        splits.append((train_idx, test_idx, station_name))
    return splits


def prepare_fold_data(
    df: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    scaler_type: str = "standard"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, object]:
    """
    Extract, scale, and return train/test arrays and the fitted scaler.
    Ensures scaler is fit strictly on training data to eliminate leakage.
    """
    X_train_raw = df.iloc[train_idx][FEATURE_NAMES].values
    y_train = df.iloc[train_idx][TARGET_NAME].values
    X_test_raw = df.iloc[test_idx][FEATURE_NAMES].values
    y_test = df.iloc[test_idx][TARGET_NAME].values

    if scaler_type == "minmax":
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    return X_train, y_train, X_test, y_test, scaler
