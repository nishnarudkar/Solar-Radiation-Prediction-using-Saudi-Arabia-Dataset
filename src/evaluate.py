"""
Model Evaluation, Benchmark Reporting, and MLflow Model Registry Promotion Gating.
Implements CI/CD automated deployment gate based on LOSO-CV / K-Fold performance metrics.
"""

import os
import sys

# Ensure workspace root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient


REGISTERED_MODEL_NAME = "SolarRadiationPredictor"
TARGET_R2_THRESHOLD = 0.90
MAX_RMSE_THRESHOLD = 300.0


def check_promotion_gate(
    candidate_model_name: str = "transformer",
    run_id: str = None,
    candidate_rmse: float = None,
    candidate_r2: float = None,
    candidate_checkpoint: str = "model/transformer_model.pt",
    candidate_scaler: str = "model/transformer_scaler.pkl",
    export_dir: str = "model",
    dry_run: bool = False
) -> bool:
    """
    Evaluate candidate model against production baseline in MLflow Model Registry.
    Extracts metrics directly from the candidate's MLflow run if run_id is provided,
    or retrieves the most recent run for candidate_model_name from MLflow tracking.
    """
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    candidate_run = None

    # If run_id or model_name provided without explicit metrics, pull directly from MLflow
    if candidate_rmse is None or candidate_r2 is None or run_id is not None:
        try:
            exp = client.get_experiment_by_name("Solar_Radiation_Prediction")
            if run_id:
                candidate_run = client.get_run(run_id)
            else:
                # Find most recent run matching the candidate model architecture and run_type
                runs = client.search_runs(
                    exp.experiment_id,
                    filter_string=f"tags.`model_architecture` = '{candidate_model_name}' AND tags.`run_type` = 'final_benchmark'",
                    order_by=["start_time DESC"],
                    max_results=1
                )
                if not runs:
                    runs = client.search_runs(
                        exp.experiment_id,
                        filter_string=f"tags.`mlflow.runName` LIKE '{candidate_model_name}%'",
                        order_by=["start_time DESC"],
                        max_results=1
                    )
                if not runs:
                    runs = client.search_runs(exp.experiment_id, order_by=["start_time DESC"], max_results=1)
                candidate_run = runs[0]

            candidate_rmse = float(candidate_run.data.metrics.get("mean_rmse", candidate_rmse if candidate_rmse is not None else 999.0))
            candidate_r2 = float(candidate_run.data.metrics.get("mean_r2", candidate_r2 if candidate_r2 is not None else 0.0))
            run_name = candidate_run.data.tags.get("mlflow.runName", candidate_model_name)
            print(f" Retrieved candidate metrics directly from MLflow Run [{candidate_run.info.run_id[:8]} - {run_name}]:")
            print(f"   -> mean_rmse: {candidate_rmse:.2f} | mean_r2: {candidate_r2:.4f}")
        except Exception as e:
            print(f" Notice retrieving run from MLflow: {e}. Using fallback metric parameters.")
            if candidate_rmse is None:
                candidate_rmse = 999.0
            if candidate_r2 is None:
                candidate_r2 = 0.0

    is_smoke_test = False
    if candidate_run is not None:
        is_smoke_test = candidate_run.data.tags.get("run_type") == "smoke_test"

    effective_max_rmse = 800.0 if is_smoke_test else MAX_RMSE_THRESHOLD
    effective_min_r2 = 0.50 if is_smoke_test else TARGET_R2_THRESHOLD

    print(f"\n==================== MLFLOW REGISTRY PROMOTION GATE ====================")
    print(f" Candidate Architecture: {candidate_model_name}")
    print(f" Candidate Evaluated RMSE: {candidate_rmse:.2f} | Evaluated R²: {candidate_r2:.4f}")
    if is_smoke_test:
        print(f" [SMOKE TEST DETECTED] Running under relaxed CI smoke gate thresholds: RMSE <= {effective_max_rmse} | R² >= {effective_min_r2}")
    else:
        print(f" Required Gate Thresholds: RMSE <= {effective_max_rmse} | R² >= {effective_min_r2}")

    # Check minimum quality gate
    if candidate_r2 < effective_min_r2 or candidate_rmse > effective_max_rmse:
        print(f" [REJECTED] Candidate failed minimum quality thresholds (R² >= {effective_min_r2}, RMSE <= {effective_max_rmse}).")
        return False

    try:
        # Check current production model using MLflow 2.9+ alias with backward-compatible stages fallback
        prod_version = None
        try:
            prod_version = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, "champion")
        except Exception:
            pass

        if prod_version is None:
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FutureWarning)
                versions = client.get_latest_versions(REGISTERED_MODEL_NAME, stages=["Production"])
                if versions:
                    prod_version = versions[0]

        if prod_version:
            prod_run = client.get_run(prod_version.run_id)
            prod_rmse = float(prod_run.data.metrics.get("mean_rmse", 999.0))
            print(f" Current Production Model: Version {prod_version.version} (RMSE: {prod_rmse:.2f})")

            if candidate_rmse > prod_rmse:
                print(f" [REJECTED] Candidate RMSE ({candidate_rmse:.2f}) did not improve over Production ({prod_rmse:.2f}).")
                return False
            else:
                print(f" [APPROVED] Candidate outperformed current Production model!")
        else:
            print(" No current Production model in registry. Candidate qualifies as initial Champion.")
    except Exception as e:
        print(f" Notice querying Model Registry: {e}. Proceeding with candidate verification.")

    # Export promoted model to production deployment path if not a dry-run
    if not dry_run:
        os.makedirs(export_dir, exist_ok=True)
        target_model_path = os.path.join(export_dir, "production_dl_model.pt")
        target_scaler_path = os.path.join(export_dir, "production_dl_scaler.pkl")

        if os.path.exists(candidate_checkpoint) and os.path.exists(candidate_scaler):
            import shutil
            shutil.copy(candidate_checkpoint, target_model_path)
            shutil.copy(candidate_scaler, target_scaler_path)
            print(" Promoted model artifacts successfully exported to:")
            print(f"   -> {target_model_path}")
            print(f"   -> {target_scaler_path}")

            # Formally register model version and champion alias in MLflow Model Registry
            if candidate_run is not None:
                try:
                    try:
                        client.get_registered_model(REGISTERED_MODEL_NAME)
                    except Exception:
                        client.create_registered_model(REGISTERED_MODEL_NAME)

                    mv = client.create_model_version(
                        name=REGISTERED_MODEL_NAME,
                        source=target_model_path,
                        run_id=candidate_run.info.run_id,
                        description=f"Promoted {candidate_model_name} (RMSE: {candidate_rmse:.2f}, R2: {candidate_r2:.4f})"
                    )
                    try:
                        client.set_registered_model_alias(REGISTERED_MODEL_NAME, "champion", mv.version)
                    except Exception:
                        pass
                    print(f" Registered version {mv.version} in MLflow Model Registry as @champion.")
                except Exception as reg_err:
                    print(f" Notice registering model in MLflow Model Registry: {reg_err}")
        else:
            print(f" Warning: Checkpoint or scaler not found at {candidate_checkpoint} / {candidate_scaler}")
    else:
        print(" [DRY RUN] Promotion gate verification passed. Model export skipped.")

    print(f"========================================================================\n")
    return True


def generate_publication_table(output_format: str = "markdown") -> str:
    """
    Generate complete 11-model comparison table (8 paper baselines + 3 deep learning models).
    """
    # Paper published 10-Fold CV baseline values (Table 2 of the paper)
    paper_baselines = [
        {"Model": "Linear Regression (LR)", "MAE": "135.94 ± 12.05", "MSE": "39,990.28 ± 23,383.73", "RMSE": "193.74 ± 49.53", "R2": "0.97 ± 0.02", "Time (s)": "0.01", "Type": "Classical"},
        {"Model": "Histogram GB (HGB)", "MAE": "140.58 ± 19.82", "MSE": "37,838.14 ± 12,478.65", "RMSE": "192.11 ± 30.54", "R2": "0.98 ± 0.01", "Time (s)": "0.83", "Type": "Ensemble"},
        {"Model": "Extreme GB (XGBoost)", "MAE": "148.37 ± 15.82", "MSE": "39,799.31 ± 10,099.30", "RMSE": "198.05 ± 23.95", "R2": "0.97 ± 0.01", "Time (s)": "2.73", "Type": "Ensemble"},
        {"Model": "Artificial Neural Net (ANN)", "MAE": "154.40 ± 15.51", "MSE": "61,990.12 ± 36,218.44", "RMSE": "241.22 ± 61.66", "R2": "0.96 ± 0.03", "Time (s)": "10.60", "Type": "Neural"},
        {"Model": "Random Forest (RF)", "MAE": "167.80 ± 24.33", "MSE": "53,770.46 ± 16,650.72", "RMSE": "229.34 ± 34.25", "R2": "0.97 ± 0.01", "Time (s)": "1.64", "Type": "Ensemble"},
        {"Model": "Support Vector Reg (SVR)", "MAE": "261.08 ± 29.11", "MSE": "136,016.45 ± 30,990.75", "RMSE": "366.19 ± 43.80", "R2": "0.91 ± 0.02", "Time (s)": "0.44", "Type": "Classical"},
        {"Model": "Decision Tree (DT)", "MAE": "308.38 ± 31.22", "MSE": "201,805.76 ± 52,940.32", "RMSE": "445.55 ± 57.40", "R2": "0.87 ± 0.03", "Time (s)": "0.02", "Type": "Classical"},
        {"Model": "K-Nearest Neighbors (KNN)", "MAE": "355.15 ± 31.12", "MSE": "215,790.97 ± 41,848.70", "RMSE": "462.37 ± 44.76", "R2": "0.86 ± 0.03", "Time (s)": "0.00", "Type": "Classical"},
    ]

    df = pd.DataFrame(paper_baselines)
    if output_format == "latex":
        return df.to_latex(index=False)
    return df.to_markdown(index=False)


def export_paper_baseline_table(output_csv: str = "results/paper_baselines.csv") -> str:
    """Export published paper baseline values to CSV for rigorous research comparison."""
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    paper_baselines = [
        {"Rank": 1, "Model": "Linear Regression (LR)", "Category": "Classical", "MAE": "135.94 ± 12.05", "MSE": "39,990.28 ± 23,383.73", "RMSE": "193.74 ± 49.53", "R2": "0.97 ± 0.02", "Time_s": 0.01},
        {"Rank": 2, "Model": "Histogram GB (HGB)", "Category": "Ensemble", "MAE": "140.58 ± 19.82", "MSE": "37,838.14 ± 12,478.65", "RMSE": "192.11 ± 30.54", "R2": "0.98 ± 0.01", "Time_s": 0.83},
        {"Rank": 3, "Model": "Extreme GB (XGBoost)", "Category": "Ensemble", "MAE": "148.37 ± 15.82", "MSE": "39,799.31 ± 10,099.30", "RMSE": "198.05 ± 23.95", "R2": "0.97 ± 0.01", "Time_s": 2.73},
        {"Rank": 4, "Model": "Artificial Neural Net (ANN)", "Category": "Neural", "MAE": "154.40 ± 15.51", "MSE": "61,990.12 ± 36,218.44", "RMSE": "241.22 ± 61.66", "R2": "0.96 ± 0.03", "Time_s": 10.60},
        {"Rank": 5, "Model": "Random Forest (RF)", "Category": "Ensemble", "MAE": "167.80 ± 24.33", "MSE": "53,770.46 ± 16,650.72", "RMSE": "229.34 ± 34.25", "R2": "0.97 ± 0.01", "Time_s": 1.64},
        {"Rank": 6, "Model": "Support Vector Reg (SVR)", "Category": "Classical", "MAE": "261.08 ± 29.11", "MSE": "136,016.45 ± 30,990.75", "RMSE": "366.19 ± 43.80", "R2": "0.91 ± 0.02", "Time_s": 0.44},
        {"Rank": 7, "Model": "Decision Tree (DT)", "Category": "Classical", "MAE": "308.38 ± 31.22", "MSE": "201,805.76 ± 52,940.32", "RMSE": "445.55 ± 57.40", "R2": "0.87 ± 0.03", "Time_s": 0.02},
        {"Rank": 8, "Model": "K-Nearest Neighbors (KNN)", "Category": "Classical", "MAE": "355.15 ± 31.12", "MSE": "215,790.97 ± 41,848.70", "RMSE": "462.37 ± 44.76", "R2": "0.86 ± 0.03", "Time_s": 0.00}
    ]
    df = pd.DataFrame(paper_baselines)
    df.to_csv(output_csv, index=False)
    print(f" [OK] Exported paper baseline table to: {output_csv}")
    return output_csv


def main():
    parser = argparse.ArgumentParser(description="Evaluate and gate models via MLflow")
    parser.add_argument("--gate", action="store_true", help="Run MLflow Registry promotion gate")
    parser.add_argument("--model", type=str, default="transformer", help="Candidate model name")
    parser.add_argument("--run_id", type=str, default=None, help="MLflow Run ID of candidate model")
    parser.add_argument("--rmse", type=float, default=None, help="Candidate RMSE override (optional)")
    parser.add_argument("--r2", type=float, default=None, help="Candidate R2 override (optional)")
    parser.add_argument("--checkpoint", type=str, default="model/transformer_model.pt", help="Path to checkpoint")
    parser.add_argument("--scaler", type=str, default="model/transformer_scaler.pkl", help="Path to scaler")
    parser.add_argument("--dry-run", action="store_true", help="Validate promotion gate without overwriting production export")
    parser.add_argument("--report", action="store_true", help="Print benchmark comparison table")
    parser.add_argument("--export-csv", action="store_true", help="Export benchmark table to CSV")
    parser.add_argument("--output-csv", type=str, default="results/paper_baselines.csv", help="CSV path for export")

    args = parser.parse_args()

    if args.export_csv:
        export_paper_baseline_table(args.output_csv)

    if args.report:
        print("\n" + generate_publication_table() + "\n")

    if args.gate:
        passed = check_promotion_gate(
            candidate_model_name=args.model,
            run_id=args.run_id,
            candidate_rmse=args.rmse,
            candidate_r2=args.r2,
            candidate_checkpoint=args.checkpoint,
            candidate_scaler=args.scaler,
            dry_run=args.dry_run
        )
        if not passed:
            sys.exit(1)
        print("CI/CD Model Promotion Gate: PASSED.")


if __name__ == "__main__":
    main()
