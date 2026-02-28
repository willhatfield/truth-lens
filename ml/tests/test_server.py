"""Tests for ml/server.py FastAPI endpoints."""

import json
import os
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi.testclient import TestClient


ENV = {
    "ML_SERVICE_API_KEY": "test-service-key",
    "ML_MODAL_ENDPOINT_PREFIX": "https://test--truthlens-ml",
    "ML_MODAL_API_KEY": "test-modal-key",
    "RESULTS_DIR": "/tmp/test-results",
}

HEADERS = {"Authorization": "Bearer test-service-key"}
BAD_HEADERS = {"Authorization": "Bearer wrong-key"}


def make_client():
    with patch.dict("os.environ", ENV):
        import importlib
        import server
        importlib.reload(server)
        return TestClient(server.app), server


def test_health_check():
    """GET / returns ok status and ml service name."""
    with patch.dict("os.environ", ENV):
        import server
        client = TestClient(server.app)
        resp = client.get("/")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["service"] == "ml"


def test_missing_token_returns_401():
    """POST /pipeline/run with no token → 401."""
    with patch.dict("os.environ", ENV):
        import server
        client = TestClient(server.app)
        resp = client.post("/pipeline/run", json={
            "analysis_id": "a_1", "prompt": "", "model_outputs": []
        })
    assert resp.status_code == 403  # HTTPBearer returns 403 when no token


def test_wrong_token_returns_401():
    """POST /pipeline/run with wrong token → 401."""
    with patch.dict("os.environ", ENV):
        import server
        client = TestClient(server.app)
        resp = client.post(
            "/pipeline/run",
            json={"analysis_id": "a_1", "prompt": "", "model_outputs": []},
            headers=BAD_HEADERS,
        )
    assert resp.status_code == 401


def test_pipeline_run_returns_running():
    """Valid POST /pipeline/run returns running status."""
    with patch.dict("os.environ", ENV):
        import server
        server._PIPELINE_STATE.clear()
        with patch.object(server, "asyncio") as mock_asyncio:
            mock_asyncio.create_task = MagicMock()
            mock_asyncio.to_thread = MagicMock(return_value=MagicMock())
            client = TestClient(server.app)
            resp = client.post(
                "/pipeline/run",
                json={"analysis_id": "a_new", "prompt": "", "model_outputs": [
                    {"model_id": "gpt4", "response_text": "hello"}
                ]},
                headers=HEADERS,
            )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "running"


def test_idempotency_running():
    """Second POST with same analysis_id=running returns existing state."""
    with patch.dict("os.environ", ENV):
        import server
        server._PIPELINE_STATE["a_idem"] = {
            "status": "running",
            "stage": "embed_claims",
            "stages_completed": ["llm_calls"],
            "result": None,
            "error": None,
        }
        client = TestClient(server.app)
        resp = client.post(
            "/pipeline/run",
            json={"analysis_id": "a_idem", "prompt": "", "model_outputs": []},
            headers=HEADERS,
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "running"


def test_idempotency_done():
    """Second POST with same analysis_id=done returns existing state, no new task."""
    with patch.dict("os.environ", ENV):
        import server
        server._PIPELINE_STATE["a_done"] = {
            "status": "done",
            "stage": "complete",
            "stages_completed": ["llm_calls", "extract_claims"],
            "result": {"ok": True},
            "error": None,
        }
        client = TestClient(server.app)
        resp = client.post(
            "/pipeline/run",
            json={"analysis_id": "a_done", "prompt": "", "model_outputs": []},
            headers=HEADERS,
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "done"
    assert data["result"] is not None


def test_done_state_has_non_null_result():
    """GET /pipeline/{id} with done status always returns non-null result."""
    with patch.dict("os.environ", ENV):
        import server
        server._PIPELINE_STATE["a_result"] = {
            "status": "done",
            "stage": "complete",
            "stages_completed": ["llm_calls"],
            "result": {"analysis_id": "a_result", "claims": []},
            "error": None,
        }
        client = TestClient(server.app)
        resp = client.get("/pipeline/a_result", headers=HEADERS)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "done"
    assert data["result"] is not None
    assert data["error"] is None


def test_error_state_has_non_null_error():
    """GET /pipeline/{id} with error status always returns non-null error."""
    with patch.dict("os.environ", ENV):
        import server
        server._PIPELINE_STATE["a_err"] = {
            "status": "error",
            "stage": "failed",
            "stages_completed": [],
            "result": None,
            "error": "extract_claims failed: timeout",
        }
        client = TestClient(server.app)
        resp = client.get("/pipeline/a_err", headers=HEADERS)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "error"
    assert data["error"] is not None
    assert data["result"] is None


def test_unknown_analysis_id_returns_404():
    """GET /pipeline/{unknown_id} → 404."""
    with patch.dict("os.environ", ENV):
        import server
        server._PIPELINE_STATE.clear()
        with patch("os.path.exists", return_value=False):
            client = TestClient(server.app)
            resp = client.get("/pipeline/nonexistent_id_xyz", headers=HEADERS)
    assert resp.status_code == 404


def test_disk_fallback_for_missing_state():
    """If analysis_id absent from _PIPELINE_STATE, read progress from disk."""
    progress_data = {
        "status": "running",
        "stage": "embed_claims",
        "stages_completed": ["llm_calls", "extract_claims"],
        "warnings": [],
    }
    with patch.dict("os.environ", ENV):
        import server
        server._PIPELINE_STATE.clear()

        progress_path = os.path.join(ENV["RESULTS_DIR"], "a_disk_progress.json")
        os.makedirs(ENV["RESULTS_DIR"], exist_ok=True)
        with open(progress_path, "w") as f:
            json.dump(progress_data, f)

        client = TestClient(server.app)
        resp = client.get("/pipeline/a_disk_progress", headers=HEADERS)

        # Cleanup
        os.remove(progress_path)

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "running"
    assert "extract_claims" in data["stages_completed"]
