"""Tests for the simplified backend proxy endpoints."""

import pytest
from unittest.mock import patch, AsyncMock
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_check():
    """GET / returns status ok."""
    resp = client.get("/")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"


@patch("app.main.start_pipeline", new_callable=AsyncMock)
def test_analyze_returns_analysis_id(mock_start):
    """POST /analyze returns analysis_id and status."""
    mock_start.return_value = {"analysis_id": "a_test123", "status": "done"}
    resp = client.post("/analyze", json={"prompt": "Is the sky blue?"})
    assert resp.status_code == 200
    data = resp.json()
    assert "analysis_id" in data
    assert data["analysis_id"].startswith("a_")
    assert data["schema_version"] == "1.0"


@patch("app.main.start_pipeline", new_callable=AsyncMock)
def test_analyze_requires_prompt(mock_start):
    """POST /analyze without prompt returns 422."""
    resp = client.post("/analyze", json={})
    assert resp.status_code == 422


@patch("app.main.get_analysis_status", new_callable=AsyncMock)
def test_get_analysis_returns_status(mock_status):
    """GET /analysis/{id} returns status from Modal."""
    mock_status.return_value = {
        "status": "running",
        "stage": "extract_claims",
        "stages_completed": ["llm_calls"],
    }
    resp = client.get("/analysis/a_test123")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "running"
