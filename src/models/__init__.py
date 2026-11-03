"""
Model definitions and factory for Deep Learning Solar Radiation Prediction.
"""

from typing import Dict, Any
import torch.nn as nn
from .lstm import SolarLSTM
from .cnn1d import SolarCNN1D
from .transformer import SolarTransformer

MODEL_REGISTRY = {
    "lstm": SolarLSTM,
    "cnn1d": SolarCNN1D,
    "transformer": SolarTransformer,
    "ft_transformer": SolarTransformer,
}


def build_model(model_name: str, **kwargs) -> nn.Module:
    """
    Instantiate a deep learning model by name.

    Args:
        model_name (str): One of ['lstm', 'cnn1d', 'transformer', 'ft_transformer'].
        **kwargs: Optional architectural hyperparameters.

    Returns:
        nn.Module: PyTorch neural network instance.
    """
    key = model_name.lower()
    if key not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model name '{model_name}'. Choose from: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[key](**kwargs)


__all__ = ["SolarLSTM", "SolarCNN1D", "SolarTransformer", "build_model", "MODEL_REGISTRY"]
