"""Tests for the Railway ML service HTTP client."""

import pytest
from unittest.mock import patch, AsyncMock, MagicMock


ENV = {
    "ML_SERVICE_URL": "http://ml.railway.internal:8001",
    "ML_SERVICE_API_KEY": "test-secret",
}


@pytest.mark.asyncio
@patch("app.ml_client.httpx.AsyncClient")
async def test_successful_poll_cycle(mock_client_cls):
    """POST running → poll done → returns result."""
    start_resp = MagicMock()
    start_resp.raise_for_status = MagicMock()
    start_resp.json.return_value = {
        "status": "running",
        "stage": "starting",
        "stages_completed": [],
    }

    poll_resp = MagicMock()
    poll_resp.raise_for_status = MagicMock()
    poll_resp.json.return_value = {
        "status": "done",
        "stage": "complete",
        "stages_completed": ["llm_calls", "extract_claims"],
        "result": {"analysis_id": "a_1", "claims": []},
    }

    mock_client = AsyncMock()
    mock_client.post.return_value = start_resp
    mock_client.get.return_value = poll_resp
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client_cls.return_value = mock_client

    with patch.dict("os.environ", ENV):
        with patch("app.ml_client.asyncio.sleep", new_callable=AsyncMock):
            from app.ml_client import run_ml_pipeline
            result = await run_ml_pipeline(
                "a_1",
                [{"model_id": "gpt4", "response_text": "some response"}],
            )

    assert result == {"analysis_id": "a_1", "claims": []}


@pytest.mark.asyncio
@patch("app.ml_client.httpx.AsyncClient")
async def test_error_propagation(mock_client_cls):
    """status=error from poll → returns warnings dict."""
    start_resp = MagicMock()
    start_resp.raise_for_status = MagicMock()
    start_resp.json.return_value = {
        "status": "running",
        "stages_completed": [],
    }

    poll_resp = MagicMock()
    poll_resp.raise_for_status = MagicMock()
    poll_resp.json.return_value = {
        "status": "error",
        "error": "extract_claims failed",
        "stages_completed": [],
    }

    mock_client = AsyncMock()
    mock_client.post.return_value = start_resp
    mock_client.get.return_value = poll_resp
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client_cls.return_value = mock_client

    with patch.dict("os.environ", ENV):
        with patch("app.ml_client.asyncio.sleep", new_callable=AsyncMock):
            from app.ml_client import run_ml_pipeline
            result = await run_ml_pipeline(
                "a_1",
                [{"model_id": "gpt4", "response_text": "text"}],
            )

    assert "warnings" in result
    assert any("ml pipeline error" in w for w in result["warnings"])


@pytest.mark.asyncio
@patch("app.ml_client.httpx.AsyncClient")
async def test_timeout_returns_warning(mock_client_cls):
    """After 240 polls with status=running, returns timeout warning."""
    start_resp = MagicMock()
    start_resp.raise_for_status = MagicMock()
    start_resp.json.return_value = {"status": "running", "stages_completed": []}

    poll_resp = MagicMock()
    poll_resp.raise_for_status = MagicMock()
    poll_resp.json.return_value = {"status": "running", "stages_completed": []}

    mock_client = AsyncMock()
    mock_client.post.return_value = start_resp
    mock_client.get.return_value = poll_resp
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client_cls.return_value = mock_client

    with patch.dict("os.environ", ENV):
        with patch("app.ml_client.asyncio.sleep", new_callable=AsyncMock):
            from app.ml_client import run_ml_pipeline
            result = await run_ml_pipeline(
                "a_timeout",
                [{"model_id": "gpt4", "response_text": "text"}],
            )

    assert "warnings" in result
    assert any("timed out" in w for w in result["warnings"])


@pytest.mark.asyncio
@patch("app.ml_client.httpx.AsyncClient")
async def test_stage_progress_publish(mock_client_cls):
    """publish is called when stages_completed grows between polls."""
    start_resp = MagicMock()
    start_resp.raise_for_status = MagicMock()
    start_resp.json.return_value = {"status": "running", "stages_completed": []}

    poll_running = MagicMock()
    poll_running.raise_for_status = MagicMock()
    poll_running.json.return_value = {
        "status": "running",
        "stages_completed": ["llm_calls"],
    }

    poll_done = MagicMock()
    poll_done.raise_for_status = MagicMock()
    poll_done.json.return_value = {
        "status": "done",
        "stages_completed": ["llm_calls", "extract_claims"],
        "result": {"ok": True},
    }

    call_count = {"n": 0}

    mock_client = AsyncMock()
    mock_client.post.return_value = start_resp

    async def mock_get(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return poll_running
        return poll_done

    mock_client.get = mock_get
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client_cls.return_value = mock_client

    published = []

    async def mock_publish(analysis_id, envelope):
        published.append(envelope)

    with patch.dict("os.environ", ENV):
        with patch("app.ml_client.asyncio.sleep", new_callable=AsyncMock):
            from app.ml_client import run_ml_pipeline
            await run_ml_pipeline(
                "a_stages",
                [{"model_id": "gpt4", "response_text": "text"}],
                publish=mock_publish,
            )

    assert len(published) >= 1
    types = [e.get("type") for e in published]
    assert "STAGE_PROGRESS" in types


@pytest.mark.asyncio
async def test_empty_model_outputs_skips_http():
    """Empty model_outputs returns skip warning without HTTP calls."""
    with patch.dict("os.environ", ENV):
        from app.ml_client import run_ml_pipeline
        result = await run_ml_pipeline("a_empty", [])

    assert "warnings" in result
    assert any("skipped" in w for w in result["warnings"])


@pytest.mark.asyncio
@patch("app.ml_client.httpx.AsyncClient")
async def test_immediate_done_on_start(mock_client_cls):
    """If POST returns status=done immediately, return result without polling."""
    start_resp = MagicMock()
    start_resp.raise_for_status = MagicMock()
    start_resp.json.return_value = {
        "status": "done",
        "result": {"analysis_id": "a_1", "claims": [1, 2, 3]},
    }

    mock_client = AsyncMock()
    mock_client.post.return_value = start_resp
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client_cls.return_value = mock_client

    with patch.dict("os.environ", ENV):
        from app.ml_client import run_ml_pipeline
        result = await run_ml_pipeline(
            "a_1",
            [{"model_id": "gpt4", "response_text": "text"}],
        )

    mock_client.get.assert_not_called()
    assert result == {"analysis_id": "a_1", "claims": [1, 2, 3]}
