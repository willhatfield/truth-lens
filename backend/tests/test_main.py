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


@patch("app.main.run_pipeline")
def test_analyze_returns_analysis_id(mock_run):
    """POST /analyze returns analysis_id and stream URLs."""
    resp = client.post("/analyze", json={"prompt": "Is the sky blue?"})
    assert resp.status_code == 200
    data = resp.json()
    assert "analysis_id" in data
    assert data["analysis_id"].startswith("a_")
    assert "ws_url" in data
    assert "sse_url" in data


@patch("app.main.run_pipeline")
def test_analyze_requires_prompt(mock_run):
    """POST /analyze without prompt returns 422."""
    resp = client.post("/analyze", json={})
    assert resp.status_code == 422

# Ghost test
# @patch("app.main.get_analysis_status", new_callable=AsyncMock)
# def test_get_analysis_returns_status(mock_status):
#     """GET /analysis/{id} returns status from Modal."""
#     mock_status.return_value = {
#         "status": "running",
#         "stage": "extract_claims",
#         "stages_completed": ["llm_calls"],
#     }
#     resp = client.get("/analysis/a_test123")
#     assert resp.status_code == 200
#     data = resp.json()
#     assert data["status"] == "running"
