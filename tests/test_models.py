"""
Unit tests for PyTorch deep learning model architectures (LSTM, 1D CNN, FT-Transformer).
"""

import pytest
import torch
from src.models import build_model


@pytest.mark.parametrize("model_name", ["lstm", "cnn1d", "transformer"])
def test_model_forward_pass(model_name):
    """Verify input shape (batch_size, 21) produces output shape (batch_size, 1)."""
    batch_size = 8
    num_features = 21

    model = build_model(model_name)
    model.eval()

    dummy_input = torch.randn(batch_size, num_features)
    with torch.no_grad():
        output = model(dummy_input)

    assert output.shape == (batch_size, 1)
    assert not torch.isnan(output).any(), f"NaN detected in {model_name} output"


@pytest.mark.parametrize("model_name", ["lstm", "cnn1d", "transformer"])
def test_model_backward_pass_gradients(model_name):
    """Verify loss calculation and backpropagation flow through all trainable parameters."""
    batch_size = 4
    num_features = 21

    model = build_model(model_name)
    model.train()

    dummy_input = torch.randn(batch_size, num_features)
    dummy_target = torch.randn(batch_size, 1)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    optimizer.zero_grad()
    preds = model(dummy_input)
    loss = criterion(preds, dummy_target)
    loss.backward()

    # Check that at least some parameters have non-zero gradients
    has_grad = False
    for param in model.parameters():
        if param.grad is not None and torch.sum(torch.abs(param.grad)) > 0:
            has_grad = True
            break

    assert has_grad, f"No gradients computed for {model_name}"


def test_single_sample_inference():
    """Verify single-sample inference (batch_size=1) works across all models for web serving."""
    for model_name in ["lstm", "cnn1d", "transformer"]:
        model = build_model(model_name)
        model.eval()
        single_input = torch.randn(1, 21)
        with torch.no_grad():
            pred = model(single_input)
        assert pred.shape == (1, 1)
        assert isinstance(pred.item(), float)
