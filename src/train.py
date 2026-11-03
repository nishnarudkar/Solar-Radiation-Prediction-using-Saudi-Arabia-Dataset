"""
MLflow-integrated Training, Cross-Validation, and Parameter Logging Pipeline.
Supports 10-Fold CV and 43-Fold Leave-One-Station-Out (LOSO) Cross-Validation.
"""

import os
import sys
import json
from typing import Tuple, Optional

# Ensure workspace root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
import argparse
import tempfile
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

import mlflow
import mlflow.pytorch

from src.dataset import (
    load_dataset,
    get_10_fold_cv_splits,
    get_loso_cv_splits,
    SolarDataset,
    FEATURE_NAMES,
    TARGET_NAME
)
from src.models import build_model, MODEL_REGISTRY
from src.export_results import compute_comprehensive_metrics, export_cross_validation_csvs


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute standard regression metrics matching paper Table 2 and Table 3."""
    return compute_comprehensive_metrics(y_true, y_pred)


def prepare_fold_data(
    df: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    scaler_type: str = "standard"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, object, object]:
    """
    Extract, scale, and return train/test feature & target arrays along with scalers.
    Ensures scalers are fit strictly on training data to eliminate data leakage.
    """
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


def train_single_fold(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    y_test_raw: np.ndarray,
    y_scaler: object,
    epochs: int = 30,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: str = "cpu"
) -> Tuple[dict, np.ndarray, float]:
    """Train a model for one fold and return validation metrics, predictions, and elapsed time."""
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()

    start_time = time.time()

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

    training_time = time.time() - start_time

    # Evaluation
    model.eval()
    all_preds_scaled = []
    t_inf_start = time.perf_counter()
    with torch.no_grad():
        for batch_X, _ in test_loader:
            batch_X = batch_X.to(device)
            preds = model(batch_X)
            all_preds_scaled.extend(preds.cpu().numpy().flatten())
    inf_time_ms = ((time.perf_counter() - t_inf_start) / max(len(all_preds_scaled), 1)) * 1000.0

    y_pred_scaled = np.array(all_preds_scaled).reshape(-1, 1)
    y_pred_raw = y_scaler.inverse_transform(y_pred_scaled).flatten()
    y_pred_raw = np.clip(y_pred_raw, 0.0, None)

    metrics = compute_metrics(y_test_raw, y_pred_raw)
    metrics["training_time"] = training_time
    metrics["inference_time_ms"] = inf_time_ms
    return metrics, y_pred_raw, training_time


try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


from sklearn.compose import TransformedTargetRegressor

ALL_CLASSICAL_MODELS = {
    "lr": lambda seed: LinearRegression(),
    "hgb": lambda seed: HistGradientBoostingRegressor(random_state=seed),
    "xgb": lambda seed: XGBRegressor(random_state=seed, n_estimators=100) if HAS_XGB else GradientBoostingRegressor(random_state=seed),
    "rf": lambda seed: RandomForestRegressor(n_estimators=100, random_state=seed, n_jobs=-1),
    "ann": lambda seed: TransformedTargetRegressor(
        regressor=MLPRegressor(hidden_layer_sizes=(16, 8), max_iter=1000, alpha=1.0, batch_size=32, early_stopping=True, random_state=seed),
        transformer=StandardScaler()
    ),
    "svr": lambda seed: TransformedTargetRegressor(
        regressor=SVR(C=100.0, epsilon=0.1, gamma='scale'),
        transformer=StandardScaler()
    ),
    "dt": lambda seed: DecisionTreeRegressor(random_state=seed),
    "knn": lambda seed: KNeighborsRegressor(n_neighbors=5)
}


def train_single_fold_classical(
    model_fn,
    X_train: np.ndarray,
    y_train_raw: np.ndarray,
    X_test: np.ndarray,
    y_test_raw: np.ndarray,
    seed: int = 42
) -> Tuple[dict, np.ndarray, float, object]:
    """Train a scikit-learn / classical model on a single fold and return the fitted model."""
    model = model_fn(seed)
    start_time = time.time()
    model.fit(X_train, y_train_raw)
    training_time = time.time() - start_time

    t_inf_start = time.perf_counter()
    y_pred = model.predict(X_test)
    inf_time_ms = ((time.perf_counter() - t_inf_start) / max(len(y_pred), 1)) * 1000.0
    y_pred = np.clip(y_pred, 0.0, None)

    metrics = compute_metrics(y_test_raw, y_pred)
    metrics["training_time"] = training_time
    metrics["inference_time_ms"] = inf_time_ms
    return metrics, y_pred, training_time, model


def generate_evaluation_plots(
    y_true_all: np.ndarray,
    y_pred_all: np.ndarray,
    model_name: str,
    output_dir: str
) -> dict:
    """Generate and save publication-quality evaluation figures."""
    os.makedirs(output_dir, exist_ok=True)
    plot_paths = {}

    # 1. Parity Plot (Actual vs Predicted GHI)
    plt.figure(figsize=(7, 6))
    plt.scatter(y_true_all, y_pred_all, alpha=0.5, color="#1f77b4", edgecolors="none", s=25)
    min_val = min(y_true_all.min(), y_pred_all.min())
    max_val = max(y_true_all.max(), y_pred_all.max())
    plt.plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="1:1 Perfect Fit")
    plt.title(f"{model_name.upper()} - Actual vs. Predicted GHI", fontsize=13, fontweight="bold")
    plt.xlabel("Measured GHI (Wh/m²)", fontsize=11)
    plt.ylabel("Predicted GHI (Wh/m²)", fontsize=11)
    plt.legend(loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    parity_path = os.path.join(output_dir, f"{model_name}_parity_plot.png")
    plt.savefig(parity_path, dpi=300)
    plt.close()
    plot_paths["parity_plot"] = parity_path

    # 2. Residual Distribution Plot
    residuals = y_true_all - y_pred_all
    plt.figure(figsize=(7, 5))
    sns.histplot(residuals, kde=True, color="#2ca02c", bins=30)
    plt.axvline(0, color="red", linestyle="--", lw=1.5)
    plt.title(f"{model_name.upper()} - Residuals Distribution", fontsize=13, fontweight="bold")
    plt.xlabel("Residual Error (Measured - Predicted GHI)", fontsize=11)
    plt.ylabel("Frequency", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    residuals_path = os.path.join(output_dir, f"{model_name}_residuals_distribution.png")
    plt.savefig(residuals_path, dpi=300)
    plt.close()
    plot_paths["residuals_plot"] = residuals_path

    return plot_paths


def run_cross_validation(
    model_name: str,
    cv_strategy: str = "kfold",
    epochs: int = 30,
    batch_size: int = 32,
    lr: float = 1e-3,
    csv_path: str = "dataset.csv",
    save_model_dir: str = "model",
    results_dir: str = "results",
    seed: int = 42,
    device: Optional[str] = None,
    config_path: Optional[str] = None
) -> dict:
    """
    Run complete cross-validation and log all parameters, metrics, and artifacts to MLflow.
    """
    set_seed(seed)
    df = load_dataset(csv_path)

    # Determine execution device (GPU vs CPU)
    if device is None:
        target_device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        target_device = device

    # Load custom tuned configuration if provided
    custom_config = {}
    if config_path and os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            custom_config = json.load(f)
        print(f" Loaded tuned hyperparameters from: {config_path}")
        if "epochs" in custom_config:
            epochs = custom_config["epochs"]
        if "batch_size" in custom_config:
            batch_size = custom_config["batch_size"]
        if "lr" in custom_config:
            lr = custom_config["lr"]

    if cv_strategy == "loso":
        splits = get_loso_cv_splits(df)
        cv_name = "43-Fold Leave-One-Station-Out (LOSO) CV"
        num_folds = len(splits)
    else:
        splits = get_10_fold_cv_splits(df, seed=seed)
        cv_name = "10-Fold CV"
        num_folds = 10

    print(f"\n==================================================")
    print(f" Training {model_name.upper()} under {cv_name}")
    print(f" Dataset: {len(df)} records across {df['Station_Name'].nunique()} stations")
    print(f" Epochs: {epochs} | Batch Size: {batch_size} | LR: {lr} | Device: {target_device.upper()}")
    print(f"==================================================")

    # Set up MLflow tracking
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("Solar_Radiation_Prediction")

    fold_metrics_list = []
    fold_predictions_list = []
    all_y_true = []
    all_y_pred = []
    total_training_time = 0.0

    is_classical = model_name.lower() in ALL_CLASSICAL_MODELS

    # Count total trainable parameters
    if not is_classical:
        dummy_model = build_model(model_name)
        total_params = sum(p.numel() for p in dummy_model.parameters() if p.requires_grad)
    else:
        dummy_model = ALL_CLASSICAL_MODELS[model_name.lower()](seed)
        total_params = 0

    with mlflow.start_run(run_name=f"{model_name}_{cv_strategy}"):
        # 1. Log All Parameters
        mlflow.log_params({
            "model_architecture": model_name,
            "cv_strategy": cv_strategy,
            "num_folds": num_folds,
            "epochs": epochs if not is_classical else 0,
            "batch_size": batch_size if not is_classical else 0,
            "learning_rate": lr if not is_classical else 0.0,
            "weight_decay": 1e-4 if not is_classical else 0.0,
            "optimizer": "AdamW" if not is_classical else "N/A",
            "lr_scheduler": "CosineAnnealingLR" if not is_classical else "N/A",
            "lr_scheduler_T_max": epochs if not is_classical else 0,
            "lr_scheduler_eta_min": 1e-6 if not is_classical else 0.0,
            "loss_function": "MSELoss" if not is_classical else "SquaredError",
            "num_features": len(FEATURE_NAMES),
            "total_dataset_records": len(df),
            "total_stations": df["Station_Name"].nunique(),
            "total_trainable_parameters": total_params,
            "random_seed": seed,
            "is_deep_learning": not is_classical
        })

        # Set explicit run tags for unambiguous gating queries
        run_type = "smoke_test" if (epochs < 10 and not is_classical) else "final_benchmark"
        mlflow.set_tags({
            "run_type": run_type,
            "model_architecture": model_name,
            "cv_strategy": cv_strategy,
            "is_deep_learning": str(not is_classical)
        })

        best_overall_model = None
        best_overall_scaler = None
        best_fold_rmse = float("inf")

        for fold_idx, split_info in enumerate(splits):
            if cv_strategy == "loso":
                train_idx, test_idx, station_name = split_info
                fold_desc = f"Fold {fold_idx+1}/{num_folds} [Station: {station_name}]"
            else:
                train_idx, test_idx = split_info
                fold_desc = f"Fold {fold_idx+1}/{num_folds}"

            # Prepare data with feature and target scalers
            X_train, y_train, X_test, y_test, x_scaler, y_scaler = prepare_fold_data(
                df, train_idx, test_idx, scaler_type="standard"
            )

            y_test_raw = df.iloc[test_idx][TARGET_NAME].values
            y_train_raw = df.iloc[train_idx][TARGET_NAME].values

            if is_classical:
                metrics, y_pred, fold_time, fitted_model = train_single_fold_classical(
                    model_fn=ALL_CLASSICAL_MODELS[model_name.lower()],
                    X_train=X_train,
                    y_train_raw=y_train_raw,
                    X_test=X_test,
                    y_test_raw=y_test_raw,
                    seed=seed
                )
            else:
                train_dataset = SolarDataset(X_train, y_train)
                test_dataset = SolarDataset(X_test, y_test)
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

                # Extract architecture parameters from custom config if present
                arch_kwargs = {
                    k: v for k, v in custom_config.items()
                    if k not in ["lr", "weight_decay", "batch_size", "epochs"]
                }
                model = build_model(model_name, **arch_kwargs)
                weight_decay = custom_config.get("weight_decay", 1e-4)

                metrics, y_pred, fold_time = train_single_fold(
                    model=model,
                    train_loader=train_loader,
                    test_loader=test_loader,
                    y_test_raw=y_test_raw,
                    y_scaler=y_scaler,
                    epochs=epochs,
                    lr=lr,
                    weight_decay=weight_decay,
                    device=target_device
                )
                fitted_model = model

            fold_metrics_list.append(metrics)
            fold_predictions_list.append(y_pred)
            all_y_true.extend(y_test_raw)
            all_y_pred.extend(y_pred)
            total_training_time += fold_time

            # Check if this fold produced the best checkpoint to persist
            if metrics["rmse"] < best_fold_rmse:
                best_fold_rmse = metrics["rmse"]
                best_overall_model = fitted_model
                best_overall_scaler = {"x_scaler": x_scaler, "y_scaler": y_scaler} if not is_classical else x_scaler

            # Log per-fold metrics
            mlflow.log_metric(f"fold_{fold_idx+1}_mae", metrics["mae"])
            mlflow.log_metric(f"fold_{fold_idx+1}_rmse", metrics["rmse"])
            mlflow.log_metric(f"fold_{fold_idx+1}_r2", metrics["r2"])
            mlflow.log_metric(f"fold_{fold_idx+1}_mbe", metrics.get("mbe", 0.0))

            if (fold_idx + 1) % 5 == 0 or (fold_idx + 1) == num_folds:
                print(f"  --> {fold_desc} | MAE: {metrics['mae']:.2f} | RMSE: {metrics['rmse']:.2f} | R²: {metrics['r2']:.4f}")

        # Compute aggregate cross-validation statistics (Mean ± Std)
        maes = [m["mae"] for m in fold_metrics_list]
        mses = [m["mse"] for m in fold_metrics_list]
        rmses = [m["rmse"] for m in fold_metrics_list]
        r2s = [m["r2"] for m in fold_metrics_list]
        mbes = [m.get("mbe", 0.0) for m in fold_metrics_list]
        nmaes = [m.get("nmae_pct", 0.0) for m in fold_metrics_list]
        nrmses = [m.get("nrmse_pct", 0.0) for m in fold_metrics_list]

        summary_metrics = {
            "mean_mae": float(np.mean(maes)),
            "std_mae": float(np.std(maes)),
            "mean_mse": float(np.mean(mses)),
            "std_mse": float(np.std(mses)),
            "mean_rmse": float(np.mean(rmses)),
            "std_rmse": float(np.std(rmses)),
            "mean_r2": float(np.mean(r2s)),
            "std_r2": float(np.std(r2s)),
            "mean_mbe": float(np.mean(mbes)),
            "std_mbe": float(np.std(mbes)),
            "mean_nmae_pct": float(np.mean(nmaes)),
            "std_nmae_pct": float(np.std(nmaes)),
            "mean_nrmse_pct": float(np.mean(nrmses)),
            "std_nrmse_pct": float(np.std(nrmses)),
            "avg_inference_time_ms": float(np.mean([m.get("inference_time_ms", 0.0) for m in fold_metrics_list])),
            "avg_training_time_s": float(total_training_time / num_folds),
            "total_training_time_s": float(total_training_time)
        }

        # 2. Log Final Summary Metrics to MLflow
        for k, v in summary_metrics.items():
            mlflow.log_metric(k, v)

        # 3. Export Comprehensive CSV Files (Samples, Folds, Stations, Benchmark)
        csv_paths = export_cross_validation_csvs(
            df=df,
            fold_splits=splits,
            fold_predictions_list=fold_predictions_list,
            fold_metrics_list=fold_metrics_list,
            model_name=model_name,
            cv_strategy=cv_strategy,
            results_dir=results_dir,
            target_col=TARGET_NAME
        )

        for csv_key, csv_path in csv_paths.items():
            if os.path.exists(csv_path):
                mlflow.log_artifact(csv_path, artifact_path="results_csv")

        # 4. Generate & Log Figures/Plots as Artifacts
        with tempfile.TemporaryDirectory() as tmp_dir:
            plot_dict = generate_evaluation_plots(
                np.array(all_y_true), np.array(all_y_pred), model_name, tmp_dir
            )
            for plot_name, plot_path in plot_dict.items():
                mlflow.log_artifact(plot_path, artifact_path="evaluation_plots")

            # Save checkpoint & scaler in model directory
            os.makedirs(save_model_dir, exist_ok=True)
            if not is_classical:
                model_save_path = os.path.join(save_model_dir, f"{model_name}_model.pt")
                scaler_save_path = os.path.join(save_model_dir, f"{model_name}_scaler.pkl")
                torch.save(best_overall_model.state_dict(), model_save_path)
                joblib.dump(best_overall_scaler, scaler_save_path)
                mlflow.log_artifact(model_save_path, artifact_path="checkpoints")
                mlflow.log_artifact(scaler_save_path, artifact_path="checkpoints")
            else:
                model_save_path = os.path.join(save_model_dir, f"{model_name}_classical_model.pkl")
                scaler_save_path = os.path.join(save_model_dir, f"{model_name}_classical_scaler.pkl")
                joblib.dump(best_overall_model, model_save_path)
                joblib.dump(best_overall_scaler, scaler_save_path)
                mlflow.log_artifact(model_save_path, artifact_path="checkpoints")
                mlflow.log_artifact(scaler_save_path, artifact_path="checkpoints")

        # Print Benchmark Table matching Paper Format
        print(f"\n==================== RESULTS: {model_name.upper()} ({cv_name}) ====================")
        print(f" MAE:   {summary_metrics['mean_mae']:.2f} ± {summary_metrics['std_mae']:.2f} Wh/m²")
        print(f" MSE:   {summary_metrics['mean_mse']:,.2f} ± {summary_metrics['std_mse']:,.2f}")
        print(f" RMSE:  {summary_metrics['mean_rmse']:.2f} ± {summary_metrics['std_rmse']:.2f} Wh/m²")
        print(f" R²:    {summary_metrics['mean_r2']:.4f} ± {summary_metrics['std_r2']:.4f}")
        print(f" MBE:   {summary_metrics['mean_mbe']:.2f} ± {summary_metrics['std_mbe']:.2f} Wh/m²")
        print(f" nMAE:  {summary_metrics['mean_nmae_pct']:.2f}% | nRMSE: {summary_metrics['mean_nrmse_pct']:.2f}%")
        print(f" Inference: {summary_metrics['avg_inference_time_ms']:.3f} ms / sample (Train: {summary_metrics['avg_training_time_s']:.2f} s / fold)")
        print(f" Saved Model: {model_save_path}")
        print(f" Saved CSVs:  {results_dir}/ ({model_name}_{cv_strategy}_*.csv)")
        print(f"========================================================================\n")

    return summary_metrics


def main():
    parser = argparse.ArgumentParser(description="Solar Radiation ML & Deep Learning Training with MLflow")
    parser.add_argument(
        "--model",
        type=str,
        default="all_dl",
        choices=["lstm", "cnn1d", "transformer", "ft_transformer", "all_dl", "all_classical", "all"] + list(ALL_CLASSICAL_MODELS.keys()),
        help="Architecture to train"
    )
    parser.add_argument(
        "--cv",
        type=str,
        default="kfold",
        choices=["kfold", "loso"],
        help="Cross-validation strategy: 'kfold' (10-fold) or 'loso' (43-fold Leave-One-Station-Out)"
    )
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs for DL")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for DL")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default=None, help="Device to use ('cpu' or 'cuda')")
    parser.add_argument("--config", type=str, default=None, help="Path to tuned JSON configuration file")
    parser.add_argument("--dataset", type=str, default="dataset.csv", help="Path to dataset.csv")
    parser.add_argument("--results-dir", type=str, default="results", help="Directory to save CSV result files")

    args = parser.parse_args()

    dl_models = ["lstm", "cnn1d", "transformer"]
    classical_models = list(ALL_CLASSICAL_MODELS.keys())

    if args.model == "all":
        models_to_train = classical_models + dl_models
    elif args.model == "all_dl":
        models_to_train = dl_models
    elif args.model == "all_classical":
        models_to_train = classical_models
    else:
        models_to_train = [args.model]

    results_table = []
    for m in models_to_train:
        model_config = args.config
        if model_config is None:
            auto_cfg = os.path.join("configs", f"{m}_best.json")
            if os.path.exists(auto_cfg):
                model_config = auto_cfg

        res = run_cross_validation(
            model_name=m,
            cv_strategy=args.cv,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            csv_path=args.dataset,
            results_dir=args.results_dir,
            seed=args.seed,
            device=args.device,
            config_path=model_config
        )
        results_table.append({
            "Model": m.upper(),
            "MAE": f"{res['mean_mae']:.2f} ± {res['std_mae']:.2f}",
            "MSE": f"{res['mean_mse']:,.2f} ± {res['std_mse']:,.2f}",
            "RMSE": f"{res['mean_rmse']:.2f} ± {res['std_rmse']:.2f}",
            "R2": f"{res['mean_r2']:.4f} ± {res['std_r2']:.4f}",
            "Inference (ms)": f"{res.get('avg_inference_time_ms', 0.0):.3f}"
        })

    print("\n\n==================== CONTROLLED 11-MODEL BENCHMARK TABLE ====================")
    res_df = pd.DataFrame(results_table)
    print(res_df.to_string(index=False))
    print("============================================================================\n")


if __name__ == "__main__":
    main()
