"""Tests for auth_utils.validate_bearer_token."""

import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from auth_utils import validate_bearer_token


class TestValidateToken:
    """Tests for Bearer token validation."""

    def test_valid_token_returns_credentials(self):
        """Valid token matching MODAL_API_KEY returns the token string."""
        token = SimpleNamespace(credentials="test-key-123")
        with patch.dict(os.environ, {"MODAL_API_KEY": "test-key-123"}):
            result = validate_bearer_token(token)
        assert result == "test-key-123"

    def test_invalid_token_raises_401(self):
        """Mismatched token raises HTTPException with 401."""
        from fastapi import HTTPException

        token = SimpleNamespace(credentials="wrong-key")
        with patch.dict(os.environ, {"MODAL_API_KEY": "correct-key"}):
            with pytest.raises(HTTPException) as exc_info:
                validate_bearer_token(token)
        assert exc_info.value.status_code == 401

    def test_missing_env_var_raises_500(self):
        """Missing MODAL_API_KEY env var raises HTTPException with 500."""
        from fastapi import HTTPException

        token = SimpleNamespace(credentials="any-key")
        env = os.environ.copy()
        env.pop("MODAL_API_KEY", None)
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(HTTPException) as exc_info:
                validate_bearer_token(token)
        assert exc_info.value.status_code == 500

    def test_empty_token_raises_401(self):
        """Empty credentials string raises 401."""
        from fastapi import HTTPException

        token = SimpleNamespace(credentials="")
        with patch.dict(os.environ, {"MODAL_API_KEY": "real-key"}):
            with pytest.raises(HTTPException) as exc_info:
                validate_bearer_token(token)
        assert exc_info.value.status_code == 401

    def test_valid_token_detail_message(self):
        """Invalid token error includes descriptive detail message."""
        from fastapi import HTTPException

        token = SimpleNamespace(credentials="bad")
        with patch.dict(os.environ, {"MODAL_API_KEY": "good"}):
            with pytest.raises(HTTPException) as exc_info:
                validate_bearer_token(token)
        assert "Invalid" in exc_info.value.detail or "missing" in exc_info.value.detail
