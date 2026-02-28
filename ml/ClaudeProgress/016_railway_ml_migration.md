# 016 — Railway ML Service Migration

**Date:** 2026-02-28
**Branch:** `feat/railway-ml-migration`

## Summary

Added Railway deployment infrastructure for the ML service and extended
`orchestrator.py` to accept pre-computed model outputs so the Railway
server can bypass the LLM stage when outputs are already available from
the backend.

## Files Created

| File | Purpose |
|------|---------|
| `ml/requirements-server.txt` | Lean pip requirements for Railway container (no torch/transformers/umap/scikit-learn) |
| `ml/server.py` | FastAPI server wrapping `orchestrator.run_full_pipeline` for Railway deployment |
| `ml/Dockerfile` | Minimal python:3.11-slim image; shell-form CMD so `$PORT` expands at runtime |
| `ml/railway.toml` | Railway build/deploy config: Dockerfile builder, health check, restart policy, 1 replica |

## Files Modified

| File | Change |
|------|--------|
| `ml/orchestrator.py` | Added `model_outputs=None` param to `run_full_pipeline`; branch skips LLM stage when param is provided |

## Architecture

```
Browser → Vercel backend → POST /pipeline/run (Railway ML service)
                         ← 200 {status: "running"}
         ← poll GET /pipeline/{id} until status == "done" or "error"
```

The Railway ML service (`server.py`):
- Authenticates via `ML_SERVICE_API_KEY` Bearer token
- Accepts pre-built `model_outputs` from the backend (skips LLM calls)
- Spawns `_run_pipeline_sync` in a thread via `asyncio.to_thread`
- Stores state in `_PIPELINE_STATE` dict (safe; `numReplicas = 1`)
- Falls back to disk (`/tmp/results`) for crash recovery on `GET /pipeline/{id}`
- Calls Modal HTTP endpoints for all heavy ML stages

## Key Design Decisions

- **Shell-form CMD** in Dockerfile: `CMD uvicorn server:app ...` (not JSON array) so Railway's injected `$PORT` env var is expanded by the shell at container start.
- **No torch/transformers in Railway image**: heavy ML lives exclusively inside Modal containers; Railway only needs `fastapi`, `uvicorn`, `httpx`, `pydantic`, `python-dotenv`, `requests`.
- **`model_outputs` bypass**: when the backend has already collected LLM responses, it passes them directly; the orchestrator skips `_run_llm_stage`, populates `models_completed`/`stages_completed`, and writes a progress file before continuing to the extract stage.

## Environment Variables Required on Railway

| Variable | Description |
|----------|-------------|
| `ML_SERVICE_API_KEY` | Bearer token the backend sends to authenticate requests |
| `ML_MODAL_ENDPOINT_PREFIX` | Modal app URL prefix (e.g. `https://org--app`) |
| `ML_MODAL_API_KEY` | Modal API key for calling HTTP endpoints |
| `RESULTS_DIR` | Optional override; defaults to `/tmp/results` |
| `PORT` | Injected automatically by Railway |

## Bugs Encountered

None.

## Next Steps

1. `railway up` from `ml/` to deploy the service
2. Set the 4 env vars above in the Railway dashboard
3. Update Vercel backend `ml_client.py` to call Railway ML service instead of Modal orchestrator endpoint
4. Add integration tests for `server.py` endpoints (health, pipeline/run, pipeline/{id})
5. Consider adding `stages_completed` to the result returned by `run_full_pipeline` so `_PIPELINE_STATE` can surface it accurately after pipeline completion
