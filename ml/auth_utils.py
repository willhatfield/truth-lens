"""Reusable Bearer token validation for TruthLens HTTP endpoints."""

import os

from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials


def validate_bearer_token(
    token: HTTPAuthorizationCredentials,
) -> str:
    """Check Bearer token against MODAL_API_KEY environment variable.

    Returns the validated token string on success.
    Raises HTTPException 401 on mismatch, 500 if env var missing.
    """
    expected = os.environ.get("MODAL_API_KEY")
    if expected is None:
        raise HTTPException(
            status_code=500,
            detail="MODAL_API_KEY not configured on server",
        )
    if token.credentials != expected:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key",
        )
    return token.credentials
