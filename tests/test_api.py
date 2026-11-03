"""
Integration tests for Flask application routes and multi-model prediction API.
"""

import json
import pytest
from app import app
from src.dataset import FEATURE_NAMES


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_index_route(client):
    """Verify index route returns HTTP 200."""
    response = client.get("/")
    assert response.status_code == 200


def test_models_route(client):
    """Verify /models route returns list of available models including deep learning."""
    response = client.get("/models")
    assert response.status_code == 200
    data = response.get_json()
    assert "models" in data
    model_ids = [m["id"] for m in data["models"]]
    assert "linear_regression" in model_ids
    assert "transformer" in model_ids
    assert "lstm" in model_ids
    assert "cnn1d" in model_ids


def test_map_data_route(client):
    """Verify /map-data route returns station coordinates and mean GHI."""
    response = client.get("/map-data")
    assert response.status_code == 200
    data = response.get_json()
    assert isinstance(data, list)
    assert len(data) > 0
    assert "station_name" in data[0]
    assert "latitude" in data[0]
    assert "avg_ghi" in data[0]


def test_predict_linear_regression(client):
    """Verify /predict route works for default Linear Regression."""
    payload = {feature: 10.0 for feature in FEATURE_NAMES}
    payload["model_type"] = "linear_regression"

    response = client.post(
        "/predict",
        data=json.dumps(payload),
        content_type="application/json"
    )
    assert response.status_code == 200
    data = response.get_json()
    assert "prediction" in data
    assert isinstance(data["prediction"], float)
    assert data["model_used"] == "linear_regression"


@pytest.mark.parametrize("model_type", ["transformer", "lstm", "cnn1d"])
def test_predict_deep_learning_models(client, model_type):
    """Verify /predict route works for PyTorch deep learning models."""
    payload = {feature: 15.0 for feature in FEATURE_NAMES}
    payload["model_type"] = model_type

    response = client.post(
        "/predict",
        data=json.dumps(payload),
        content_type="application/json"
    )
    assert response.status_code == 200
    data = response.get_json()
    assert "prediction" in data
    assert isinstance(data["prediction"], float)
    assert data["model_used"] == model_type
