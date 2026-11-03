"""
RandomizedSearchCV Automated Hyperparameter Optimization Pipeline with MLflow Integration.
Supports 10-Fold CV (or custom folds) across Deep Learning and Classical ML architectures.
"""

import os
import sys
import json
import argparse
from typing import Dict, Any, Tuple, Optional

# Ensure workspace root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
from scipy.stats import uniform, randint, loguniform

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVR
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

import mlflow

from src.dataset import (
    load_dataset,
    get_10_fold_cv_splits,
    SolarDataset,
    FEATURE_NAMES,
    TARGET_NAME
)
from src.models import build_model


def get_default_device(requested_device: Optional[str] = None) -> str:
    """Return requested device or automatically detect CUDA GPU availability."""
    if requested_device:
        return requested_device
    return "cuda" if torch.cuda.is_available() else "cpu"


def prepare_fold_data_scaled(
    df: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    scaler_type: str = "standard"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, object, object]:
    """Prepare scaled train/test splits with zero data leakage."""
    X_train_raw = df.iloc[train_idx][FEATURE_NAMES].values
    y_train_raw = df.iloc[train_idx][TARGET_NAME].values
    X_test_raw = df.iloc[test_idx][FEATURE_NAMES].values
    y_test_raw = df.iloc[test_idx][TARGET_NAME].values

    if scaler_type == "minmax":
        x_scaler = MinMaxScaler()
        y_scaler = MinMaxScaler()
    else:
        x_scaler = StandardScaler()
        y_scaler = StandardScaler()

    X_train = x_scaler.fit_transform(X_train_raw)
    X_test = x_scaler.transform(X_test_raw)
    y_train = y_scaler.fit_transform(y_train_raw.reshape(-1, 1))
    y_test = y_scaler.transform(y_test_raw.reshape(-1, 1))

    return X_train, y_train, X_test, y_test, x_scaler, y_scaler


def get_param_distributions(model_name: str) -> Dict[str, Any]:
    """Return the hyperparameter search distribution space for a specified model."""
    model_key = model_name.lower()

    if model_key in ["transformer", "ft_transformer"]:
        return {
            "d_model": [32, 48, 64],
            "nhead": [2, 4, 8],
            "num_layers": [1, 2, 3],
            "dim_feedforward": [48, 64, 80, 96, 112, 128, 144, 160, 176, 192],
            "dropout": [0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
            "lr": loguniform(1e-4, 5e-3),
            "weight_decay": loguniform(1e-5, 1e-3),
            "batch_size": [16, 32, 64]
        }

    elif model_key == "lstm":
        return {
            "embed_dim": [16, 32, 64],
            "hidden_dim": [32, 64, 128],
            "num_layers": [1, 2, 3],
            "dropout": [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40],
            "bidirectional": [True, False],
            "lr": loguniform(1e-4, 5e-3),
            "weight_decay": loguniform(1e-5, 1e-3),
            "batch_size": [16, 32, 64]
        }

    elif model_key == "cnn1d":
        return {
            "base_channels": [16, 32, 64],
            "dropout": [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40],
            "lr": loguniform(1e-4, 5e-3),
            "weight_decay": loguniform(1e-5, 1e-3),
            "batch_size": [16, 32, 64]
        }

    elif model_key == "svr":
        return {
            "C": loguniform(1.0, 1000.0),
            "epsilon": loguniform(0.01, 1.0),
            "gamma": ["scale", "auto"]
        }

    elif model_key == "hgb":
        return {
            "max_depth": randint(3, 11),
            "learning_rate": loguniform(0.01, 0.3),
            "max_iter": [50, 100, 150, 200, 250, 300],
            "min_samples_leaf": randint(5, 51),
            "l2_regularization": uniform(0.0, 10.0)
        }

    elif model_key == "xgb":
        dist = {
            "max_depth": randint(3, 11),
            "learning_rate": loguniform(0.01, 0.3),
            "n_estimators": [50, 100, 150, 200, 250, 300],
            "subsample": uniform(0.6, 0.4)
        }
        if HAS_XGB:
            dist["colsample_bytree"] = uniform(0.6, 0.4)
            dist["reg_alpha"] = loguniform(1e-3, 10.0)
        return dist

    elif model_key == "rf":
        return {
            "n_estimators": [50, 100, 150, 200],
            "max_depth": randint(5, 26),
            "min_samples_split": randint(2, 11),
            "min_samples_leaf": randint(1, 11)
        }

    elif model_key == "ann":
        return {
            "hidden_layer_sizes": [
                (16,), (32,), (64,), (128,),
                (16, 8), (16, 16),
                (32, 8), (32, 16), (32, 32),
                (64, 16), (64, 32), (64, 64),
                (128, 32), (128, 64)
            ],
            "alpha": loguniform(1e-4, 10.0),
            "learning_rate_init": loguniform(1e-4, 1e-2)
        }

    else:
        raise ValueError(f"Tuning not supported for model: '{model_name}'")


def sample_hyperparameters(model_name: str, seed: Optional[int] = None) -> Dict[str, Any]:
    """Sample a random parameter configuration for the specified model from its distributions."""
    rng = np.random.RandomState(seed)
    distributions = get_param_distributions(model_name)
    params = {}

    for param_name, dist in distributions.items():
        if isinstance(dist, list):
            idx = rng.randint(0, len(dist))
            params[param_name] = dist[idx]
        elif hasattr(dist, "rvs"):
            val = dist.rvs(random_state=rng)
            if hasattr(val, "item"):
                val = val.item()
            params[param_name] = val
        else:
            params[param_name] = dist

    # Ensure validity constraints for Transformer (nhead must divide d_model)
    if model_name.lower() in ["transformer", "ft_transformer"]:
        d_model = params["d_model"]
        valid_heads = [h for h in [2, 4, 8] if d_model % h == 0]
        params["nhead"] = int(rng.choice(valid_heads))

    return params


def train_and_eval_dl_fold(
    model_name: str,
    params: Dict[str, Any],
    train_loader: DataLoader,
    test_loader: DataLoader,
    y_test_raw: np.ndarray,
    y_scaler: object,
    epochs: int = 20,
    device: str = "cpu"
) -> float:
    """Train DL model on one fold with given params and return fold RMSE."""
    arch_kwargs = {
        k: v for k, v in params.items()
        if k not in ["lr", "weight_decay", "batch_size"]
    }
    model = build_model(model_name, **arch_kwargs).to(device)

    lr = params.get("lr", 1e-3)
    weight_decay = params.get("weight_decay", 1e-4)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            preds = model(batch_X)
            loss = criterion(preds, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        scheduler.step()

    # Inference on test set
    model.eval()
    all_preds_scaled = []
    with torch.no_grad():
        for batch_X, _ in test_loader:
            batch_X = batch_X.to(device)
            preds = model(batch_X)
            all_preds_scaled.extend(preds.cpu().numpy().flatten())

    y_pred_scaled = np.array(all_preds_scaled).reshape(-1, 1)
    y_pred_raw = y_scaler.inverse_transform(y_pred_scaled).flatten()
    y_pred_raw = np.clip(y_pred_raw, 0.0, None)

    return float(np.sqrt(mean_squared_error(y_test_raw, y_pred_raw)))


def clipped_rmse_scorer(estimator, X: np.ndarray, y: np.ndarray) -> float:
    """Scikit-learn compatible scoring function computing negative RMSE on non-negative predictions."""
    preds = estimator.predict(X)
    preds = np.clip(preds, 0.0, None)
    return -float(np.sqrt(mean_squared_error(y, preds)))


def tune_classical_model(
    model_name: str,
    df: pd.DataFrame,
    cv_splits: list,
    n_iter: int = 30,
    seed: int = 42
) -> Tuple[Dict[str, Any], float]:
    """Tune classical ML models using scikit-learn's RandomizedSearchCV."""
    model_key = model_name.lower()

    if model_key == "svr":
        base_estimator = Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", TransformedTargetRegressor(
                regressor=SVR(),
                transformer=StandardScaler()
            ))
        ])
        param_distributions = {
            "regressor__regressor__C": loguniform(1.0, 1000.0),
            "regressor__regressor__epsilon": loguniform(0.01, 1.0),
            "regressor__regressor__gamma": ["scale", "auto"]
        }

    elif model_key == "hgb":
        base_estimator = Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", HistGradientBoostingRegressor(random_state=seed))
        ])
        param_distributions = {
            "regressor__max_depth": randint(3, 11),
            "regressor__learning_rate": loguniform(0.01, 0.3),
            "regressor__max_iter": [50, 100, 150, 200, 250, 300],
            "regressor__min_samples_leaf": randint(5, 51),
            "regressor__l2_regularization": uniform(0.0, 10.0)
        }

    elif model_key == "xgb":
        inner = XGBRegressor(random_state=seed) if HAS_XGB else GradientBoostingRegressor(random_state=seed)
        base_estimator = Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", inner)
        ])
        param_distributions = {
            "regressor__max_depth": randint(3, 11),
            "regressor__learning_rate": loguniform(0.01, 0.3),
            "regressor__n_estimators": [50, 100, 150, 200, 250, 300],
            "regressor__subsample": uniform(0.6, 0.4)
        }
        if HAS_XGB:
            param_distributions["regressor__colsample_bytree"] = uniform(0.6, 0.4)
            param_distributions["regressor__reg_alpha"] = loguniform(1e-3, 10.0)

    elif model_key == "rf":
        base_estimator = Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", RandomForestRegressor(random_state=seed, n_jobs=-1))
        ])
        param_distributions = {
            "regressor__n_estimators": [50, 100, 150, 200],
            "regressor__max_depth": randint(5, 26),
            "regressor__min_samples_split": randint(2, 11),
            "regressor__min_samples_leaf": randint(1, 11)
        }

    elif model_key == "ann":
        base_estimator = Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", TransformedTargetRegressor(
                regressor=MLPRegressor(max_iter=800, early_stopping=True, random_state=seed),
                transformer=StandardScaler()
            ))
        ])
        param_distributions = {
            "regressor__regressor__hidden_layer_sizes": [
                (16,), (32,), (64,), (128,),
                (16, 8), (16, 16),
                (32, 8), (32, 16), (32, 32),
                (64, 16), (64, 32), (64, 64),
                (128, 32), (128, 64)
            ],
            "regressor__regressor__alpha": loguniform(1e-4, 10.0),
            "regressor__regressor__learning_rate_init": loguniform(1e-4, 1e-2)
        }
    else:
        raise ValueError(f"Classical tuning not supported for model: '{model_name}'")

    X = df[FEATURE_NAMES].values
    y = df[TARGET_NAME].values

    search = RandomizedSearchCV(
        estimator=base_estimator,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=cv_splits,
        scoring=clipped_rmse_scorer,
        random_state=seed,
        refit=False
    )
    search.fit(X, y)

    best_rmse = float(-search.best_score_)
    best_params = {}
    for k, v in search.best_params_.items():
        clean_k = k.split("__")[-1]
        if hasattr(v, "item"):
            v = v.item()
        best_params[clean_k] = v

    return best_params, best_rmse


def tune_dl_model(
    model_name: str,
    df: pd.DataFrame,
    cv_splits: list,
    n_iter: int = 30,
    epochs: int = 20,
    device: str = "cpu",
    seed: int = 42
) -> Tuple[Dict[str, Any], float]:
    """Execute randomized search across CV folds for PyTorch deep learning architectures."""
    best_rmse = float("inf")
    best_params = {}

    for trial_idx in range(n_iter):
        trial_seed = seed + trial_idx
        params = sample_hyperparameters(model_name, seed=trial_seed)
        batch_size = params.get("batch_size", 32)
        fold_rmses = []

        for fold_idx, (train_idx, test_idx) in enumerate(cv_splits):
            X_train, y_train, X_test, y_test, x_scaler, y_scaler = prepare_fold_data_scaled(
                df, train_idx, test_idx, scaler_type="standard"
            )
            y_test_raw = df.iloc[test_idx][TARGET_NAME].values

            train_dataset = SolarDataset(X_train, y_train)
            test_dataset = SolarDataset(X_test, y_test)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

            fold_rmse = train_and_eval_dl_fold(
                model_name=model_name,
                params=params,
                train_loader=train_loader,
                test_loader=test_loader,
                y_test_raw=y_test_raw,
                y_scaler=y_scaler,
                epochs=epochs,
                device=device
            )
            fold_rmses.append(fold_rmse)

        mean_rmse = float(np.mean(fold_rmses))
        print(f"  [Iter {trial_idx + 1}/{n_iter}] Mean CV RMSE: {mean_rmse:.2f} Wh/m²")

        if mean_rmse < best_rmse:
            best_rmse = mean_rmse
            best_params = params

    return best_params, best_rmse


def run_tuning(
    model_name: str,
    n_trials: int = 30,
    cv_folds: int = 10,
    epochs: int = 20,
    device: Optional[str] = None,
    dataset_path: str = "dataset.csv",
    output_dir: str = "configs",
    seed: int = 42
) -> Tuple[Dict[str, Any], float]:
    """
    Execute hyperparameter search using RandomizedSearchCV / randomized search loop
    and log results to MLflow.
    """
    target_device = get_default_device(device)
    is_dl = model_name.lower() in ["transformer", "ft_transformer", "lstm", "cnn1d"]

    print("\n==================================================")
    print(f" Randomized Hyperparameter Tuning: {model_name.upper()}")
    print(f" Iterations: {n_trials} | CV Strategy: {cv_folds}-Fold CV | Device: {target_device.upper()}")
    print("==================================================")

    df = load_dataset(dataset_path)
    cv_splits = get_10_fold_cv_splits(df, seed=seed)
    if cv_folds < len(cv_splits):
        cv_splits = cv_splits[:cv_folds]

    # Setup MLflow
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("Solar_Radiation_Prediction")

    with mlflow.start_run(run_name=f"random_search_{model_name}"):
        mlflow.log_params({
            "tuning_method": "RandomizedSearchCV",
            "tuning_model": model_name,
            "n_iter": n_trials,
            "cv_folds": cv_folds,
            "epochs_per_trial": epochs if is_dl else 0,
            "device": target_device,
            "seed": seed
        })

        if is_dl:
            best_params, best_rmse = tune_dl_model(
                model_name=model_name,
                df=df,
                cv_splits=cv_splits,
                n_iter=n_trials,
                epochs=epochs,
                device=target_device,
                seed=seed
            )
        else:
            best_params, best_rmse = tune_classical_model(
                model_name=model_name,
                df=df,
                cv_splits=cv_splits,
                n_iter=n_trials,
                seed=seed
            )

        print("\n==================== TUNING COMPLETED ====================")
        print(f" Best {cv_folds}-Fold Mean RMSE: {best_rmse:.2f} Wh/m²")
        print(" Best Hyperparameters:")
        for k, v in best_params.items():
            print(f"   -> {k}: {v}")
        print("==========================================================\n")

        # Log best results to MLflow
        mlflow.log_metric("best_cv_rmse", best_rmse)
        for k, v in best_params.items():
            if isinstance(v, (int, float, str, bool)):
                mlflow.log_param(f"best_{k}", v)

        # Export best config to JSON
        os.makedirs(output_dir, exist_ok=True)
        config_path = os.path.join(output_dir, f"{model_name.lower()}_best.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(best_params, f, indent=4)

        mlflow.log_artifact(config_path, artifact_path="tuned_configs")
        print(f" Exported best configuration to: {config_path}")

    return best_params, best_rmse


def main():
    parser = argparse.ArgumentParser(description="Automated Hyperparameter Optimization via RandomizedSearchCV")
    parser.add_argument(
        "--model",
        type=str,
        default="transformer",
        choices=["transformer", "ft_transformer", "lstm", "cnn1d", "svr", "hgb", "xgb", "rf", "ann"],
        help="Architecture to tune"
    )
    parser.add_argument("--trials", type=int, default=30, help="Number of randomized search iterations")
    parser.add_argument("--cv-folds", type=int, default=10, help="Number of CV folds to evaluate per trial")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs per fold for DL models")
    parser.add_argument("--device", type=str, default=None, help="Device to use ('cpu' or 'cuda')")
    parser.add_argument("--dataset", type=str, default="dataset.csv", help="Path to dataset.csv")
    parser.add_argument("--output-dir", type=str, default="configs", help="Directory to save best configs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    run_tuning(
        model_name=args.model,
        n_trials=args.trials,
        cv_folds=args.cv_folds,
        epochs=args.epochs,
        device=args.device,
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
