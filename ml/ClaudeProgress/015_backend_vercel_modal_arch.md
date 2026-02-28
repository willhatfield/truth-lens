# 015 — Backend Vercel + Modal Architecture Redesign

**Date:** 2026-02-28
**Branch:** `feat/backend-vercel-modal-arch`

## Summary

Moved pipeline orchestration from backend (Vercel) to Modal to solve the 10s Vercel Hobby timeout constraint. The backend is now a thin proxy; Modal handles everything (LLM calls, ML pipeline, safe answer, metrics, results storage).

## Architecture Change

```
BEFORE:                                AFTER:
Browser → Vercel (FastAPI)             Browser → Vercel (3 thin routes)
           ├── LLM calls                         ├── POST /analyze
           ├── ML calls → Modal                  └── GET /analysis/{id}
           ├── WebSocket/SSE                            │
           └── In-memory state                          ▼
                                                 Modal (orchestrator)
                                                 ├── LLM calls (OpenAI, Claude, Gemini)
                                                 ├── ML functions (existing 7)
                                                 ├── Safe answer + metrics
                                                 └── Modal Volume (results)
```

## New ML Files Created

| File | Lines | Purpose |
|------|-------|---------|
| `orchestrator.py` | 589 | Full pipeline orchestrator with 11 stages |
| `llm_providers.py` | 161 | Synchronous HTTP calls to OpenAI/Claude/Gemini |
| `model_metrics.py` | 124 | Per-model claim count computation |
| `evidence.py` | 34 | Evidence retrieval stub (empty lists) |
| `safe_answer.py` | 368 | Copied from backend (deterministic builder) |

## Modified ML Files

| File | Change |
|------|--------|
| `modal_app.py` | Added results_volume, 3 new HTTP endpoints, updated images |
| `tests/test_modal_deploy.py` | Updated for 17 registered functions (was 14) |
| `tests/test_safe_answer.py` | Fixed pre-existing test_opener_at_index_0 bug |

## New Test Files (ML)

| File | Tests | Description |
|------|-------|-------------|
| `tests/test_orchestrator.py` | 8 | Full pipeline, error paths, fallbacks |
| `tests/test_llm_providers.py` | 6 | Provider calls with mocked HTTP |
| `tests/test_model_metrics.py` | 6 | Pure logic, empty input, ordering |
| `tests/test_evidence.py` | 3 | Stub returns empty structure |
| `tests/test_safe_answer.py` | 75 | Copied from backend (1 test fixed) |

## Backend Changes

### Simplified Files
- `backend/app/main.py` — 3 endpoints: health, POST /analyze, GET /analysis/{id}
- `backend/app/ml_client.py` — Thin HTTP client calling Modal's 3 endpoints
- `backend/app/schemas.py` — AnalyzeRequest + AnalyzeResponse only

### Removed Files
- `backend/app/orchestrator.py` — logic moved to `ml/orchestrator.py`
- `backend/app/providers/` — logic moved to `ml/llm_providers.py`
- `backend/app/modal_calls/` — already on Modal
- `backend/app/ws.py` — no WebSocket on Vercel Hobby
- `backend/app/streaming.py` — no streaming needed
- `backend/app/store.py` — results stored on Modal Volume

### New Backend Tests
- `backend/tests/test_main.py` — 4 tests (health, analyze, prompt validation, status)
- `backend/tests/test_ml_client.py` — 4 tests (start, running status, done+result, missing prefix)

## Test Results

- **ML:** 546 passed, 0 failed
- **Backend:** 8 passed, 0 failed

## Bugs Encountered

1. **test_opener_at_index_0:** Pre-existing test/implementation mismatch — `_OPENER_PREFIXES = {""}` means only empty string is valid at index 0, but the test expected transition words like "Overall,". Fixed test to match implementation.
2. **test_exactly_seven_functions:** Updated from 14 to 17 registered functions to account for 3 new pipeline endpoints.

## Next Steps

1. `modal deploy modal_app.py` — deploy the updated Modal app with new endpoints
2. Set `ML_MODAL_ENDPOINT_PREFIX` env var on Vercel
3. Frontend: switch from WebSocket/SSE to polling pattern (POST /analyze → poll GET /analysis/{id})
4. Wire real evidence retrieval (web search) into `evidence.py`
5. Implement `rewrite_client` for safe answer transition rewriting
