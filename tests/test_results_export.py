"""
Unit tests for the research CSV results export module (src/export_results.py).
"""

import os
import tempfile
import numpy as np
import pandas as pd
import pytest

from src.export_results import (
    compute_comprehensive_metrics,
    export_cross_validation_csvs,
    get_model_category
)
from src.evaluate import export_paper_baseline_table


def test_compute_comprehensive_metrics():
    """Verify calculation of MAE, RMSE, R2, MBE, NMAE, and NRMSE."""
    y_true = np.array([100.0, 200.0, 300.0, 400.0])
    y_pred = np.array([110.0, 190.0, 310.0, 390.0])  # Errors: +10, -10, +10, -10

    metrics = compute_comprehensive_metrics(y_true, y_pred)

    assert metrics["mae"] == pytest.approx(10.0)
    assert metrics["rmse"] == pytest.approx(10.0)
    assert metrics["mbe"] == pytest.approx(0.0)
    assert metrics["r2"] > 0.99
    # Mean y_true is 250.0, so NMAE = 10 / 250 * 100 = 4.0%
    assert metrics["nmae_pct"] == pytest.approx(4.0)
    assert metrics["nrmse_pct"] == pytest.approx(4.0)


def test_get_model_category():
    """Verify categorization of models for publication tables."""
    assert get_model_category("transformer") == "Deep Learning"
    assert get_model_category("cnn1d") == "Deep Learning"
    assert get_model_category("lstm") == "Deep Learning"
    assert get_model_category("ann") == "Neural Baseline"
    assert get_model_category("hgb") == "Ensemble"
    assert get_model_category("xgb") == "Ensemble"
    assert get_model_category("rf") == "Ensemble"
    assert get_model_category("svr") == "Classical ML"
    assert get_model_category("lr") == "Classical ML"


def test_export_cross_validation_csvs():
    """Verify that export_cross_validation_csvs generates all 4 expected CSV files with correct schema."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create synthetic dataset with stations and meteorology
        data = {
            "Station_Name": ["Riyadh"] * 5 + ["Jeddah"] * 5,
            "Latitude": [24.71] * 5 + [21.54] * 5,
            "Longitude": [46.67] * 5 + [39.17] * 5,
            "Date": [f"2020-0{i+1}" for i in range(5)] * 2,
            "Air Temperature (C°)": [30.0 + i for i in range(10)],
            "DHI (Wh/m2)": [150.0 + i * 5 for i in range(10)],
            "DNI (Wh/m2)": [600.0 + i * 10 for i in range(10)],
            "Wind Speed at 3m (m/s)": [3.5] * 10,
            "Relative Humidity (%)": [40.0] * 10,
            "Barometric Pressure (mB (hPa equiv))": [1013.0] * 10,
            "GHI (Wh/m2)": [500.0 + i * 20 for i in range(10)]
        }
        df = pd.DataFrame(data)

        # 2 folds: 5 train, 5 test each
        splits = [
            (np.arange(0, 5), np.arange(5, 10)),
            (np.arange(5, 10), np.arange(0, 5))
        ]
        fold_predictions = [
            df["GHI (Wh/m2)"].iloc[5:10].values + 5.0,
            df["GHI (Wh/m2)"].iloc[0:5].values - 5.0
        ]
        fold_metrics = [
            compute_comprehensive_metrics(df["GHI (Wh/m2)"].iloc[5:10].values, fold_predictions[0]),
            compute_comprehensive_metrics(df["GHI (Wh/m2)"].iloc[0:5].values, fold_predictions[1])
        ]

        csv_paths = export_cross_validation_csvs(
            df=df,
            fold_splits=splits,
            fold_predictions_list=fold_predictions,
            fold_metrics_list=fold_metrics,
            model_name="test_model",
            cv_strategy="kfold",
            results_dir=tmp_dir,
            target_col="GHI (Wh/m2)"
        )

        # Verify all 4 files exist
        for key in ["samples_csv", "folds_csv", "stations_csv", "summary_csv"]:
            assert key in csv_paths
            assert os.path.exists(csv_paths[key])

        # Verify samples CSV schema
        samples_df = pd.read_csv(csv_paths["samples_csv"])
        assert len(samples_df) == 10
        for col in ["record_id", "station_name", "latitude", "longitude", "actual_ghi", "predicted_ghi", "residual_error", "absolute_error", "relative_error_pct"]:
            assert col in samples_df.columns

        # Verify folds CSV schema
        folds_df = pd.read_csv(csv_paths["folds_csv"])
        assert len(folds_df) == 2
        for col in ["fold", "model_name", "category", "cv_strategy", "mae", "rmse", "r2", "mbe", "nmae_pct", "nrmse_pct"]:
            assert col in folds_df.columns

        # Verify stations CSV schema
        stations_df = pd.read_csv(csv_paths["stations_csv"])
        assert len(stations_df) == 2  # Riyadh and Jeddah
        for col in ["station_name", "num_samples", "actual_mean_ghi", "pred_mean_ghi", "mae", "rmse", "r2", "mbe"]:
            assert col in stations_df.columns

        # Verify benchmark summary CSV schema
        summary_df = pd.read_csv(csv_paths["summary_csv"])
        assert len(summary_df) == 1
        assert summary_df["model_name"].iloc[0] == "test_model"
        assert "mean_mae" in summary_df.columns
        assert "mean_rmse" in summary_df.columns
        assert "mean_mbe" in summary_df.columns


def test_export_paper_baseline_table():
    """Verify that paper baselines are exported cleanly to CSV."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        out_path = os.path.join(tmp_dir, "test_baselines.csv")
        export_paper_baseline_table(out_path)

        assert os.path.exists(out_path)
        df = pd.read_csv(out_path)
        assert len(df) == 8
        assert "Model" in df.columns
        assert "MAE" in df.columns
        assert "RMSE" in df.columns
        assert "R2" in df.columns


def test_export_statistical_significance_csv():
    """Verify export of statistical significance and 95% confidence intervals."""
    from src.export_results import export_statistical_significance_csv

    with tempfile.TemporaryDirectory() as tmp_dir:
        out_path = export_statistical_significance_csv(results_dir=tmp_dir)
        assert os.path.exists(out_path)

        df = pd.read_csv(out_path)
        assert len(df) == 11
        for col in ["Model", "Category", "MAE", "95% CI (MAE)", "RMSE", "95% CI (RMSE)", "R2", "95% CI (R2)", "p_value_vs_champion", "Significance", "Inference_ms"]:
            assert col in df.columns


def test_export_loso_statistical_significance_csv():
    """Verify export of 43-Fold LOSO statistical significance and 95% confidence intervals."""
    from src.export_results import export_loso_statistical_significance_csv

    with tempfile.TemporaryDirectory() as tmp_dir:
        out_path = export_loso_statistical_significance_csv(results_dir=tmp_dir)
        assert os.path.exists(out_path)

        df = pd.read_csv(out_path)
        assert len(df) == 11
        for col in ["Model", "Category", "MAE", "95% CI (MAE)", "RMSE", "95% CI (RMSE)", "R2", "95% CI (R2)", "p_value_vs_champion", "Significance", "Inference_ms"]:
            assert col in df.columns

