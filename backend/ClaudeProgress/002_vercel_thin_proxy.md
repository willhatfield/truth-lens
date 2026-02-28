# 002 — Vercel Thin Proxy Simplification

**Date:** 2026-02-28
**Branch:** `feat/backend-vercel-modal-arch`

## Summary

Simplified the backend from a full pipeline orchestrator with WebSocket/SSE streaming to a thin proxy that calls Modal's 3 pipeline HTTP endpoints. This enables deployment on Vercel Hobby (10s function timeout) since all heavy lifting is done on Modal.

## Changes

### Simplified
- `app/main.py` — 3 routes: `GET /`, `POST /analyze`, `GET /analysis/{id}` + CORS
- `app/ml_client.py` — 3 async functions: `start_pipeline`, `get_analysis_status`, `_endpoint_url`
- `app/schemas.py` — 2 models: `AnalyzeRequest`, `AnalyzeResponse`

### Removed
- `app/orchestrator.py` — moved to `ml/orchestrator.py`
- `app/providers/` — moved to `ml/llm_providers.py`
- `app/modal_calls/` — already on Modal
- `app/ws.py` — no WebSocket on Vercel Hobby
- `app/streaming.py` — no streaming needed
- `app/store.py` — results on Modal Volume

### Kept
- `app/safe_answer.py` — reference copy

### Tests Added
- `tests/test_main.py` — 4 tests
- `tests/test_ml_client.py` — 4 tests (async, using pytest-asyncio)

## Test Results

8 passed, 0 failed

## Next Steps

1. Deploy to Vercel with `ML_MODAL_ENDPOINT_PREFIX` and `ML_MODAL_API_KEY` env vars
2. Frontend: switch from WebSocket/SSE to polling pattern
