"""Tests for the FastAPI inference API."""

from unittest.mock import MagicMock, patch

import pytest
import torch
from fastapi.testclient import TestClient

from customer_support.api import PRIORITY_NAMES, app


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)


@pytest.fixture
def mock_model():
    """Create a mock model that returns fake logits."""
    mock = MagicMock()
    # Return logits that predict class 1 (medium) with high confidence
    mock_output = MagicMock()
    mock_output.logits = torch.tensor([[0.1, 2.5, 0.3]])  # Class 1 has highest logit
    mock.return_value = mock_output
    return mock


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer."""
    mock = MagicMock()
    mock.return_value = {
        "input_ids": torch.tensor([[101, 2054, 2003, 102] + [0] * 508]),
        "attention_mask": torch.tensor([[1, 1, 1, 1] + [0] * 508]),
    }
    return mock


class TestRootEndpoint:
    """Tests for the root endpoint."""

    def test_root_returns_200(self, client):
        """Test that root endpoint returns 200 status."""
        response = client.get("/")
        assert response.status_code == 200

    def test_root_returns_service_info(self, client):
        """Test that root endpoint returns service information."""
        response = client.get("/")
        data = response.json()
        assert "service" in data
        assert "version" in data
        assert "endpoints" in data


class TestHealthEndpoint:
    """Tests for the health check endpoint."""

    def test_health_returns_200(self, client, mock_model):
        """Test that health endpoint returns 200 when model loads."""
        with patch("customer_support.api.get_model", return_value=mock_model):
            response = client.get("/health")
            assert response.status_code == 200

    def test_health_returns_status(self, client, mock_model):
        """Test that health endpoint returns status field."""
        with patch("customer_support.api.get_model", return_value=mock_model):
            response = client.get("/health")
            data = response.json()
            assert "status" in data
            assert "model_loaded" in data

    def test_health_reports_healthy_when_model_loaded(self, client, mock_model):
        """Test that health reports healthy when model is loaded."""
        with patch("customer_support.api.get_model", return_value=mock_model):
            response = client.get("/health")
            data = response.json()
            assert data["status"] == "healthy"
            assert data["model_loaded"] is True

    def test_health_reports_unhealthy_when_model_fails(self, client):
        """Test that health reports unhealthy when model loading fails."""
        with patch("customer_support.api.get_model", side_effect=FileNotFoundError("Model not found")):
            response = client.get("/health")
            data = response.json()
            assert data["status"] == "unhealthy"
            assert data["model_loaded"] is False


class TestPredictEndpoint:
    """Tests for the prediction endpoint."""

    def test_predict_returns_200_with_valid_request(self, client, mock_model, mock_tokenizer):
        """Test that predict endpoint returns 200 with valid input."""
        with (
            patch("customer_support.api.get_model", return_value=mock_model),
            patch("customer_support.api.get_tokenizer", return_value=mock_tokenizer),
        ):
            response = client.post("/predict", json={"text": "My computer is broken"})
            assert response.status_code == 200

    def test_predict_returns_expected_fields(self, client, mock_model, mock_tokenizer):
        """Test that predict endpoint returns all expected fields."""
        with (
            patch("customer_support.api.get_model", return_value=mock_model),
            patch("customer_support.api.get_tokenizer", return_value=mock_tokenizer),
        ):
            response = client.post("/predict", json={"text": "My computer is broken"})
            data = response.json()
            assert "priority" in data
            assert "priority_id" in data
            assert "confidence" in data

    def test_predict_priority_id_in_valid_range(self, client, mock_model, mock_tokenizer):
        """Test that priority_id is in valid range [0, 2]."""
        with (
            patch("customer_support.api.get_model", return_value=mock_model),
            patch("customer_support.api.get_tokenizer", return_value=mock_tokenizer),
        ):
            response = client.post("/predict", json={"text": "My computer is broken"})
            data = response.json()
            assert data["priority_id"] in [0, 1, 2]

    def test_predict_confidence_in_valid_range(self, client, mock_model, mock_tokenizer):
        """Test that confidence is in valid range [0, 1]."""
        with (
            patch("customer_support.api.get_model", return_value=mock_model),
            patch("customer_support.api.get_tokenizer", return_value=mock_tokenizer),
        ):
            response = client.post("/predict", json={"text": "My computer is broken"})
            data = response.json()
            assert 0.0 <= data["confidence"] <= 1.0

    def test_predict_priority_matches_priority_id(self, client, mock_model, mock_tokenizer):
        """Test that priority string matches priority_id."""
        with (
            patch("customer_support.api.get_model", return_value=mock_model),
            patch("customer_support.api.get_tokenizer", return_value=mock_tokenizer),
        ):
            response = client.post("/predict", json={"text": "My computer is broken"})
            data = response.json()
            assert data["priority"] == PRIORITY_NAMES[data["priority_id"]]

    def test_predict_priority_is_valid_value(self, client, mock_model, mock_tokenizer):
        """Test that priority is one of: low, medium, high."""
        with (
            patch("customer_support.api.get_model", return_value=mock_model),
            patch("customer_support.api.get_tokenizer", return_value=mock_tokenizer),
        ):
            response = client.post("/predict", json={"text": "My computer is broken"})
            data = response.json()
            assert data["priority"] in ["low", "medium", "high"]

    def test_predict_empty_text_returns_422(self, client):
        """Test that empty text returns 422 validation error."""
        response = client.post("/predict", json={"text": ""})
        assert response.status_code == 422

    def test_predict_missing_text_field_returns_422(self, client):
        """Test that missing text field returns 422 validation error."""
        response = client.post("/predict", json={})
        assert response.status_code == 422

    def test_predict_returns_503_when_model_not_found(self, client, mock_tokenizer):
        """Test that predict returns 503 when model checkpoint is missing."""
        with (
            patch("customer_support.api.get_model", side_effect=FileNotFoundError("Model not found")),
            patch("customer_support.api.get_tokenizer", return_value=mock_tokenizer),
        ):
            response = client.post("/predict", json={"text": "My computer is broken"})
            assert response.status_code == 503
