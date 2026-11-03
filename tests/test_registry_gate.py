"""
Unit tests for MLflow Model Registry promotion gate logic.
"""

from src.evaluate import check_promotion_gate


def test_promotion_gate_thresholds():
    """Verify promotion gate rejects models with bad R2 or RMSE."""
    # Bad R2
    assert not check_promotion_gate(
        candidate_model_name="bad_model",
        candidate_rmse=180.0,
        candidate_r2=0.80,  # Below TARGET_R2_THRESHOLD
        candidate_checkpoint="dummy.pt",
        candidate_scaler="dummy.pkl"
    )

    # Bad RMSE
    assert not check_promotion_gate(
        candidate_model_name="bad_model",
        candidate_rmse=450.0,  # Above MAX_RMSE_THRESHOLD
        candidate_r2=0.96,
        candidate_checkpoint="dummy.pt",
        candidate_scaler="dummy.pkl"
    )


def test_promotion_gate_acceptance():
    """Verify promotion gate accepts high-performing models that outperform production."""
    assert check_promotion_gate(
        candidate_model_name="test_transformer",
        candidate_rmse=95.0,
        candidate_r2=0.995,
        candidate_checkpoint="dummy.pt",
        candidate_scaler="dummy.pkl",
        dry_run=True
    )


def test_promotion_gate_mlflow_retrieval():
    """Verify promotion gate can extract metrics directly from logged MLflow runs."""
    import os
    import mlflow

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("Solar_Radiation_Prediction")

    with mlflow.start_run(run_name="transformer_test_retrieval"):
        mlflow.set_tags({
            "model_architecture": "transformer",
            "run_type": "final_benchmark"
        })
        mlflow.log_metrics({
            "mean_rmse": 90.0,
            "mean_r2": 0.996
        })

    passed = check_promotion_gate(
        candidate_model_name="transformer",
        candidate_checkpoint="model/transformer_model.pt",
        candidate_scaler="model/transformer_scaler.pkl",
        dry_run=True
    )
    assert passed is True
