# 016 Railway Migration

## Branch
`feat/railway-migration`

## Changes Made

### New Files
- `backend/Dockerfile` — Python 3.11-slim image, shell-form CMD for Railway $PORT expansion
- `backend/railway.toml` — dockerfile builder, health check at `/`, on_failure restart
- `ml/requirements-server.txt` — lean deps for Railway container (no torch/transformers/umap)
- `ml/server.py` — FastAPI server wrapping orchestrator; endpoints: GET /, POST /pipeline/run, GET /pipeline/{id}
- `ml/Dockerfile` — Python 3.11-slim image, copies *.py, shell-form CMD
- `ml/railway.toml` — dockerfile builder, numReplicas=1, health check at `/`

### Modified Files
- `backend/app/main.py` — Added CORSMiddleware with FRONTEND_ORIGIN env var support
- `backend/app/ml_client.py` — Full rewrite: polls Railway ML service instead of calling Modal directly; added `publish` param for STAGE_PROGRESS events
- `backend/app/orchestrator.py` — Pass `publish=publish` to `run_ml_pipeline` (1-line change)
- `ml/orchestrator.py` — Added `model_outputs=None` param to `run_full_pipeline`; when provided, skips LLM stage and uses pre-computed outputs from backend

### New Tests
- `backend/tests/test_ml_client.py` — Replaced with tests for new polling client
- `ml/tests/test_server.py` — Health check, auth, idempotency, state contracts, disk fallback
- `ml/tests/test_orchestrator_model_outputs.py` — Tests for model_outputs skip behavior

## Architecture

```
Frontend (Vercel)
    ↕ SSE / WebSocket
Backend (Railway) — LLM streaming + pipeline orchestration
    ↕ Railway Private Network (http://ml.railway.internal)
ML Orchestrator (Railway, 1 replica) — pipeline coordination
    ↕ HTTP
Modal GPU/CPU Functions — extract, embed, cluster, rerank, nli, umap, score
```

## Environment Variables Required

### Backend Railway Service
- `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`
- `ML_SERVICE_URL` — private URL e.g. `http://truthlens-ml.railway.internal:8001`
- `ML_SERVICE_API_KEY` — shared secret
- `FRONTEND_ORIGIN` — Vercel URL for CORS

### ML Railway Service
- `ML_MODAL_ENDPOINT_PREFIX` — e.g. `https://vicxiya24--truthlens-ml`
- `ML_MODAL_API_KEY` — Modal bearer token
- `ML_SERVICE_API_KEY` — must match backend value
- `RESULTS_DIR` — `/tmp/results` (crash-recovery fallback)

## Railway Dashboard Setup
1. Create project with two services from same GitHub repo
2. `truthlens-backend`: Root Dir = `backend`, Config = `/backend/railway.toml`
3. `truthlens-ml`: Root Dir = `ml`, Config = `/ml/railway.toml`, replicas = 1
4. Enable private networking
5. Deploy ML first → get private URL → set `ML_SERVICE_URL` on backend
6. Deploy backend → set public URL as `VITE_BACKEND_URL` on Vercel

## Key Design Decisions
- `_PIPELINE_STATE` is module-level dict; safe because ML pinned to 1 replica
- Idempotency: POST /pipeline/run checks existing state before spawning task
- Disk fallback: GET /pipeline/{id} reads progress JSON if not in memory (crash recovery)
- Poll interval: 5s, max 240 polls = 1200s timeout
- STAGE_PROGRESS events emitted to SSE stream as stages_completed grows

## Next Steps
- Set environment variables in Railway dashboard
- `modal deploy ml/modal_app.py` to confirm Modal functions are current
- Verify: `GET https://{backend}.railway.app/` → `{"status": "ok"}`
- Verify: full analysis with SSE → observe STAGE_PROGRESS events
