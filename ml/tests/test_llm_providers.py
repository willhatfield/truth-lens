"""Tests for llm_providers.py — LLM API calls and concurrent orchestration."""

import pytest
from unittest.mock import patch, MagicMock

from llm_providers import call_openai, call_claude, call_gemini, call_all_llms


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------

def _mock_httpx_response(json_data, status_code=200):
    """Create a mock httpx response."""
    mock_resp = MagicMock()
    mock_resp.json.return_value = json_data
    mock_resp.status_code = status_code
    mock_resp.raise_for_status.return_value = None
    return mock_resp


def _mock_httpx_client(mock_resp):
    """Create a mock httpx.Client context manager."""
    mock_client = MagicMock()
    mock_client.__enter__ = MagicMock(return_value=mock_client)
    mock_client.__exit__ = MagicMock(return_value=False)
    mock_client.post.return_value = mock_resp
    return mock_client


# ---------------------------------------------------------------------------
# OpenAI response fixture data
# ---------------------------------------------------------------------------

_OPENAI_RESPONSE = {
    "choices": [
        {
            "message": {
                "content": "The Earth revolves around the Sun."
            }
        }
    ]
}

_CLAUDE_RESPONSE = {
    "content": [
        {
            "text": "The Earth orbits the Sun in an elliptical path."
        }
    ]
}

_GEMINI_RESPONSE = {
    "candidates": [
        {
            "content": {
                "parts": [
                    {
                        "text": "Earth follows an elliptical orbit around the Sun."
                    }
                ]
            }
        }
    ]
}


# ---------------------------------------------------------------------------
# Individual provider tests
# ---------------------------------------------------------------------------

class TestCallOpenai:
    def test_call_openai_success(self):
        mock_resp = _mock_httpx_response(_OPENAI_RESPONSE)
        mock_client = _mock_httpx_client(mock_resp)

        with patch("httpx.Client", return_value=mock_client):
            result = call_openai("Is Earth round?", "sk-test-key")

        assert result["model_id"] == "openai_gpt4"
        assert result["response_text"] == "The Earth revolves around the Sun."
        assert "error" not in result

    def test_call_openai_error(self):
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.side_effect = ConnectionError("network down")

        with patch("httpx.Client", return_value=mock_client):
            result = call_openai("Is Earth round?", "sk-test-key")

        assert result["model_id"] == "openai_gpt4"
        assert result["response_text"] == ""
        assert "error" in result
        assert "network down" in result["error"]


class TestCallClaude:
    def test_call_claude_success(self):
        mock_resp = _mock_httpx_response(_CLAUDE_RESPONSE)
        mock_client = _mock_httpx_client(mock_resp)

        with patch("httpx.Client", return_value=mock_client):
            result = call_claude("Is Earth round?", "sk-ant-test-key")

        assert result["model_id"] == "claude_sonnet_4"
        assert result["response_text"] == "The Earth orbits the Sun in an elliptical path."
        assert "error" not in result


class TestCallGemini:
    def test_call_gemini_success(self):
        mock_resp = _mock_httpx_response(_GEMINI_RESPONSE)
        mock_client = _mock_httpx_client(mock_resp)

        with patch("httpx.Client", return_value=mock_client):
            result = call_gemini("Is Earth round?", "gemini-test-key")

        assert result["model_id"] == "gemini_2_0"
        assert result["response_text"] == "Earth follows an elliptical orbit around the Sun."
        assert "error" not in result


# ---------------------------------------------------------------------------
# Concurrent orchestration tests
# ---------------------------------------------------------------------------

class TestCallAllLlms:
    def test_call_all_llms_concurrent(self):
        """All three providers succeed; verify all responses collected."""
        openai_resp = _mock_httpx_response(_OPENAI_RESPONSE)
        claude_resp = _mock_httpx_response(_CLAUDE_RESPONSE)
        gemini_resp = _mock_httpx_response(_GEMINI_RESPONSE)

        openai_client = _mock_httpx_client(openai_resp)
        claude_client = _mock_httpx_client(claude_resp)
        gemini_client = _mock_httpx_client(gemini_resp)

        call_count = 0

        def _client_factory(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count % 3 == 1:
                return openai_client
            if call_count % 3 == 2:
                return claude_client
            return gemini_client

        api_keys = {
            "openai": "sk-test",
            "anthropic": "sk-ant-test",
            "gemini": "gemini-test",
        }

        with patch("httpx.Client", side_effect=_client_factory):
            responses, warnings = call_all_llms("test prompt", api_keys)

        assert len(responses) == 3
        model_ids = {r["model_id"] for r in responses}
        assert "openai_gpt4" in model_ids
        assert "claude_sonnet_4" in model_ids
        assert "gemini_2_0" in model_ids
        assert len(warnings) == 0

    def test_call_all_llms_partial_failure(self):
        """One provider fails; verify warning generated and others collected."""
        openai_resp = _mock_httpx_response(_OPENAI_RESPONSE)
        gemini_resp = _mock_httpx_response(_GEMINI_RESPONSE)

        openai_client = _mock_httpx_client(openai_resp)
        gemini_client = _mock_httpx_client(gemini_resp)

        # Claude client raises an exception
        failing_client = MagicMock()
        failing_client.__enter__ = MagicMock(return_value=failing_client)
        failing_client.__exit__ = MagicMock(return_value=False)
        failing_client.post.side_effect = ConnectionError("API unreachable")

        call_count = 0

        def _client_factory(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count % 3 == 1:
                return openai_client
            if call_count % 3 == 2:
                return failing_client
            return gemini_client

        api_keys = {
            "openai": "sk-test",
            "anthropic": "sk-ant-test",
            "gemini": "gemini-test",
        }

        with patch("httpx.Client", side_effect=_client_factory):
            responses, warnings = call_all_llms("test prompt", api_keys)

        # Two providers succeed, one fails
        assert len(responses) == 2
        model_ids = {r["model_id"] for r in responses}
        assert "openai_gpt4" in model_ids
        assert "gemini_2_0" in model_ids

        # Warning generated for the failed provider
        assert len(warnings) == 1
        assert "anthropic" in warnings[0].lower() or "API unreachable" in warnings[0]
