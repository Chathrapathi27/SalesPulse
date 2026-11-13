
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api import main as api_main


@pytest.fixture
def client():
    return TestClient(api_main.app)


@pytest.fixture
def prediction_file(tmp_path, monkeypatch):
    """Create a temporary valid predictions CSV."""

    file_path = tmp_path / "latest_predictions.csv"

    data = pd.DataFrame({
        "id": ["PRODUCT_001", "PRODUCT_002"],
        "date": ["2025-02-01", "2025-02-02"],
        "sales": [10.0, 15.0],
        "predicted_sales": [12.0, 16.0],
        "model_name": ["random_forest", "random_forest"],
    })

    data.to_csv(file_path, index=False)

    monkeypatch.setattr(
        api_main,
        "PREDICTIONS_FILE",
        file_path
    )

    return file_path


@pytest.fixture
def metrics_file(tmp_path, monkeypatch):
    """Create a temporary valid model metrics CSV."""

    file_path = tmp_path / "model_comparison.csv"

    data = pd.DataFrame({
        "model": ["Random Forest", "Linear Regression"],
        "MAE": [1.20, 1.25],
        "RMSE": [2.58, 2.72],
        "MAPE": [50.5, 54.8],
    })

    data.to_csv(file_path, index=False)

    monkeypatch.setattr(
        api_main,
        "METRICS_FILE",
        file_path
    )

    return file_path


# ============================================================
# HOME ENDPOINT TESTS
# ============================================================

def test_home_endpoint(client):
    response = client.get("/")

    assert response.status_code == 200

    result = response.json()

    assert result["project"] == "SalesPulse"
    assert result["status"] == "Running"
    assert result["version"] == "1.1.0"
    assert "predictions" in result["endpoints"]
    assert "metrics" in result["endpoints"]


# ============================================================
# HEALTH ENDPOINT TESTS
# ============================================================

def test_health_endpoint(client, prediction_file, metrics_file):
    response = client.get("/health")

    assert response.status_code == 200

    result = response.json()

    assert result["status"] == "healthy"
    assert result["api"] == "running"
    assert result["predictions_file_exists"] is True
    assert result["metrics_file_exists"] is True
    assert result["predictions_available"] is True
    assert result["metrics_available"] is True


def test_health_when_files_are_missing(client, tmp_path, monkeypatch):
    monkeypatch.setattr(
        api_main,
        "PREDICTIONS_FILE",
        tmp_path / "missing_predictions.csv"
    )

    monkeypatch.setattr(
        api_main,
        "METRICS_FILE",
        tmp_path / "missing_metrics.csv"
    )

    response = client.get("/health")

    assert response.status_code == 200

    result = response.json()

    assert result["predictions_file_exists"] is False
    assert result["metrics_file_exists"] is False


# ============================================================
# PREDICTION ENDPOINT TESTS
# ============================================================

def test_predict_endpoint(client, prediction_file):
    response = client.get("/predict")

    assert response.status_code == 200

    result = response.json()

    assert result["total_returned"] == 2
    assert len(result["predictions"]) == 2

    first_prediction = result["predictions"][0]

    assert first_prediction["id"] == "PRODUCT_001"
    assert first_prediction["sales"] == 10.0
    assert first_prediction["predicted_sales"] == 12.0
    assert first_prediction["model_name"] == "random_forest"


def test_predict_with_limit(client, prediction_file):
    response = client.get("/predict?limit=1")

    assert response.status_code == 200

    result = response.json()

    assert result["total_returned"] == 1
    assert len(result["predictions"]) == 1


def test_predict_rejects_invalid_limit(client, prediction_file):
    response = client.get("/predict?limit=0")

    assert response.status_code == 422


def test_predict_rejects_limit_above_maximum(client, prediction_file):
    response = client.get("/predict?limit=50001")

    assert response.status_code == 422


def test_predict_when_file_is_missing(client, tmp_path, monkeypatch):
    monkeypatch.setattr(
        api_main,
        "PREDICTIONS_FILE",
        tmp_path / "missing_predictions.csv"
    )

    response = client.get("/predict")

    assert response.status_code == 404
    assert "Prediction file not found" in response.json()["detail"]


def test_predict_with_invalid_schema(client, prediction_file):
    pd.DataFrame({
        "id": ["PRODUCT_001"],
        "date": ["2025-02-01"],
        "sales": [10.0],
    }).to_csv(prediction_file, index=False)

    response = client.get("/predict")

    assert response.status_code == 500
    assert "invalid schema" in response.json()["detail"]


def test_predict_with_invalid_dates(client, prediction_file):
    pd.DataFrame({
        "id": ["PRODUCT_001"],
        "date": ["invalid-date"],
        "sales": [10.0],
        "predicted_sales": [12.0],
        "model_name": ["random_forest"],
    }).to_csv(prediction_file, index=False)

    response = client.get("/predict")

    assert response.status_code == 500
    assert "invalid dates" in response.json()["detail"]


def test_predict_with_invalid_sales(client, prediction_file):
    pd.DataFrame({
        "id": ["PRODUCT_001"],
        "date": ["2025-02-01"],
        "sales": ["unknown"],
        "predicted_sales": [12.0],
        "model_name": ["random_forest"],
    }).to_csv(prediction_file, index=False)

    response = client.get("/predict")

    assert response.status_code == 500
    assert "invalid sales values" in response.json()["detail"]


def test_predict_with_empty_file(client, prediction_file):
    pd.DataFrame(columns=[
        "id",
        "date",
        "sales",
        "predicted_sales",
        "model_name",
    ]).to_csv(prediction_file, index=False)

    response = client.get("/predict")

    assert response.status_code == 200
    assert response.json() == {
        "total_returned": 0,
        "predictions": [],
    }


# ============================================================
# MODEL METRICS ENDPOINT TESTS
# ============================================================

def test_metrics_endpoint(client, metrics_file):
    response = client.get("/metrics")

    assert response.status_code == 200

    result = response.json()

    assert result["total_models"] == 2
    assert len(result["model_metrics"]) == 2

    first_model = result["model_metrics"][0]

    assert first_model["model"] == "Random Forest"
    assert first_model["MAE"] == pytest.approx(1.20)
    assert first_model["RMSE"] == pytest.approx(2.58)


def test_metrics_when_file_is_missing(client, tmp_path, monkeypatch):
    monkeypatch.setattr(
        api_main,
        "METRICS_FILE",
        tmp_path / "missing_metrics.csv"
    )

    response = client.get("/metrics")

    assert response.status_code == 404
    assert "Model comparison file not found" in response.json()["detail"]


def test_metrics_with_empty_file(client, metrics_file):
    pd.DataFrame(columns=[
        "model",
        "MAE",
        "RMSE",
        "MAPE",
    ]).to_csv(metrics_file, index=False)

    response = client.get("/metrics")

    assert response.status_code == 200

    assert response.json() == {
        "total_models": 0,
        "model_metrics": [],
    }
