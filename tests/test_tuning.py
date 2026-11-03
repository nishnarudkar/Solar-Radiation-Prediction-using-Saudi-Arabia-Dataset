"""
Unit and integration tests for RandomizedSearchCV hyperparameter optimization module.
"""

import os
import pytest
import torch
import numpy as np
import pandas as pd

from src.tune import (
    get_param_distributions,
    sample_hyperparameters,
    run_tuning,
    get_default_device
)
from src.dataset import FEATURE_NAMES, TARGET_NAME
from src.models import build_model


@pytest.fixture
def dummy_dataset_path(tmp_path):
    """Create a temporary dummy dataset CSV for fast test runs."""
    csv_file = tmp_path / "dummy_dataset.csv"
    np.random.seed(42)
    data = {feat: np.random.randn(40) for feat in FEATURE_NAMES}
    data["Station_Name"] = ["Station_A"] * 20 + ["Station_B"] * 20
    data[TARGET_NAME] = np.random.uniform(3000, 8000, 40)
    df = pd.DataFrame(data)
    df.to_csv(csv_file, index=False)
    return str(csv_file)


@pytest.mark.parametrize("model_name", ["transformer", "lstm", "cnn1d", "svr", "hgb", "xgb", "rf", "ann"])
def test_get_param_distributions(model_name):
    """Verify search spaces are defined as distributions/lists without errors."""
    dist = get_param_distributions(model_name)
    assert isinstance(dist, dict)
    assert len(dist) > 0


@pytest.mark.parametrize("model_name", ["transformer", "lstm", "cnn1d", "svr", "hgb", "xgb", "rf", "ann"])
def test_sample_hyperparameters(model_name):
    """Verify search spaces can be sampled without errors."""
    params = sample_hyperparameters(model_name, seed=42)

    assert isinstance(params, dict)
    assert len(params) > 0

    if model_name == "transformer":
        assert "d_model" in params
        assert "nhead" in params
        assert params["d_model"] % params["nhead"] == 0
    elif model_name == "svr":
        assert "C" in params
        assert "epsilon" in params


def test_get_default_device():
    """Verify auto-detection returns valid torch device string."""
    device = get_default_device()
    assert device in ["cpu", "cuda"]
    assert get_default_device("cpu") == "cpu"


def test_tuning_classical_smoke(dummy_dataset_path, tmp_path):
    """Run a fast smoke tuning test on classical SVR using RandomizedSearchCV."""
    out_dir = str(tmp_path / "configs")
    best_params, best_rmse = run_tuning(
        model_name="svr",
        n_trials=2,
        cv_folds=2,
        dataset_path=dummy_dataset_path,
        output_dir=out_dir,
        seed=42
    )

    assert isinstance(best_params, dict)
    assert "C" in best_params
    assert best_rmse > 0

    config_file = os.path.join(out_dir, "svr_best.json")
    assert os.path.exists(config_file)


def test_tuning_dl_smoke(dummy_dataset_path, tmp_path):
    """Run a fast smoke tuning test on FT-Transformer."""
    out_dir = str(tmp_path / "configs")
    best_params, best_rmse = run_tuning(
        model_name="transformer",
        n_trials=1,
        cv_folds=2,
        epochs=2,
        device="cpu",
        dataset_path=dummy_dataset_path,
        output_dir=out_dir,
        seed=42
    )

    assert isinstance(best_params, dict)
    assert "d_model" in best_params
    assert best_rmse > 0

    # Ensure sampled params are compatible with build_model
    arch_kwargs = {k: v for k, v in best_params.items() if k not in ["lr", "weight_decay", "batch_size", "epochs"]}
    model = build_model("transformer", **arch_kwargs)
    assert isinstance(model, torch.nn.Module)
