"""
Unified Inference Engine for Solar Radiation (GHI) Prediction.
Supports Linear Regression baseline, scikit-learn models, and PyTorch Deep Learning models.
"""

import os
import joblib
import numpy as np
import pandas as pd
import torch

from src.dataset import FEATURE_NAMES
from src.models import build_model


class SolarPredictor:
    """Unified predictor supporting classical ML and deep learning architectures."""

    def __init__(self, model_dir: str = "model", other_models_dir: str = "Other Models"):
        self.model_dir = model_dir
        self.other_models_dir = other_models_dir
        self.cached_models = {}
        self.cached_scalers = {}

    def get_available_models(self) -> list:
        """List all models currently ready for inference."""
        models = [
            {"id": "linear_regression", "name": "Linear Regression (Baseline Deployed)", "type": "Classical ML"},
            {"id": "transformer", "name": "Transformer (FT-Transformer)", "type": "Deep Learning"},
            {"id": "lstm", "name": "Solar LSTM (Recurrent Deep Net)", "type": "Deep Learning"},
            {"id": "cnn1d", "name": "1D CNN (Convolutional Deep Net)", "type": "Deep Learning"},
            {"id": "ann", "name": "Artificial Neural Network (ANN)", "type": "Neural Network"}
        ]
        return models

    def _load_sklearn_model(self, model_path: str, scaler_path: str):
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler

    def _load_pytorch_model(self, model_name: str, checkpoint_path: str, scaler_path: str):
        config_path = os.path.join("configs", f"{model_name.lower()}_best.json")
        arch_kwargs = {}
        if os.path.exists(config_path):
            try:
                import json
                with open(config_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                arch_kwargs = {
                    k: v for k, v in cfg.items()
                    if k not in ["lr", "weight_decay", "batch_size", "epochs"]
                }
            except Exception:
                arch_kwargs = {}

        model = build_model(model_name, **arch_kwargs)
        if os.path.exists(checkpoint_path):
            state_dict = torch.load(checkpoint_path, map_location=torch.device("cpu"), weights_only=True)
            model.load_state_dict(state_dict)
        model.eval()
        scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
        return model, scaler

    def predict(self, input_data: dict, model_type: str = "linear_regression") -> float:
        """
        Run inference for a single input record.

        Args:
            input_data (dict): Dictionary mapping feature names to numerical values.
            model_type (str): Key identifying the model architecture.

        Returns:
            float: Predicted Global Horizontal Irradiance (GHI) in Wh/m².
        """
        # Validate and assemble feature vector
        vector = []
        for feature in FEATURE_NAMES:
            if feature not in input_data or input_data[feature] is None or pd.isna(input_data[feature]):
                raise ValueError(f"Missing required feature: '{feature}'")
            vector.append(float(input_data[feature]))

        feature_array = np.array(vector, dtype=np.float64).reshape(1, -1)

        model_key = model_type.lower()

        # 1. Classical Baseline: Linear Regression
        if model_key == "linear_regression":
            model_path = os.path.join(self.model_dir, "linear_regression_model.pkl")
            scaler_path = os.path.join(self.model_dir, "linear_regression_standard_scaler.pkl")
            if model_key not in self.cached_models:
                self.cached_models[model_key], self.cached_scalers[model_key] = self._load_sklearn_model(model_path, scaler_path)
            model = self.cached_models[model_key]
            scaler = self.cached_scalers[model_key]
            scaled = scaler.transform(feature_array)
            pred = float(model.predict(scaled)[0])
            return max(0.0, pred)

        # 2. Artificial Neural Network (Scikit-learn ANN)
        elif model_key == "ann":
            model_path = os.path.join(self.other_models_dir, "ANN_model_scikit.pkl")
            scaler_path = os.path.join(self.other_models_dir, "standard_scaler_ann_scikit.pkl")
            if model_key not in self.cached_models:
                self.cached_models[model_key], self.cached_scalers[model_key] = self._load_sklearn_model(model_path, scaler_path)
            model = self.cached_models[model_key]
            scaler = self.cached_scalers[model_key]
            scaled = scaler.transform(feature_array)
            pred = float(model.predict(scaled)[0])
            return max(0.0, pred)

        # 3. PyTorch Deep Learning Models (LSTM, 1D CNN, Transformer)
        elif model_key in ["lstm", "cnn1d", "transformer", "ft_transformer", "production"]:
            arch_name = "transformer" if model_key in ["transformer", "ft_transformer", "production"] else model_key
            ckpt_path = os.path.join(self.model_dir, f"{arch_name}_model.pt")
            scaler_path = os.path.join(self.model_dir, f"{arch_name}_scaler.pkl")

            # Fallback to linear regression standard scaler if specific DL scaler not yet saved
            if not os.path.exists(scaler_path):
                scaler_path = os.path.join(self.model_dir, "linear_regression_standard_scaler.pkl")

            if model_key not in self.cached_models:
                self.cached_models[model_key], self.cached_scalers[model_key] = self._load_pytorch_model(
                    arch_name, ckpt_path, scaler_path
                )

            model = self.cached_models[model_key]
            scaler_obj = self.cached_scalers[model_key]

            # Check if scaler is a bundle dictionary or single scaler
            if isinstance(scaler_obj, dict):
                x_scaler = scaler_obj.get("x_scaler")
                y_scaler = scaler_obj.get("y_scaler")
            else:
                x_scaler = scaler_obj
                y_scaler = None

            scaled_x = x_scaler.transform(feature_array) if x_scaler is not None else feature_array
            x_tensor = torch.tensor(scaled_x, dtype=torch.float32)

            with torch.no_grad():
                raw_pred = model(x_tensor).item()

            if y_scaler is not None:
                final_pred = float(y_scaler.inverse_transform([[raw_pred]])[0][0])
            else:
                final_pred = float(raw_pred)

            return max(0.0, final_pred)

        else:
            raise ValueError(f"Unsupported model type '{model_type}'. Choose from: {self.get_available_models()}")
