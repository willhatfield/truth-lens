"""FastAPI server wrapping the TruthLens ML orchestrator for Railway deployment."""

import asyncio
import json
import os
from typing import Any

import httpx
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

load_dotenv()

app = FastAPI()
_bearer = HTTPBearer()

# Module-level state — safe because ML service is pinned to 1 replica.
_PIPELINE_STATE: dict[str, dict[str, Any]] = {}

_RESULTS_DIR = os.getenv("RESULTS_DIR", "/tmp/results")

_MODAL_ENDPOINT_PREFIX = os.getenv("ML_MODAL_ENDPOINT_PREFIX", "").strip().rstrip("/")
_MODAL_API_KEY = os.getenv("ML_MODAL_API_KEY", "").strip()

# Timeouts match Modal function specs exactly.
_TIMEOUTS = {
    "extract_claims": 610,
    "embed_claims": 610,
    "rerank_evidence_batch": 610,
    "compute_umap": 610,
    "nli_verify_batch": 910,
    "cluster_claims": 310,
    "score_clusters": 310,
}

_ENDPOINT_SUFFIXES = {
    "extract_claims": "http-extract-claims",
    "embed_claims": "http-embed-claims",
    "cluster_claims": "http-cluster-claims",
    "rerank_evidence_batch": "http-rerank-evidence-batch",
    "nli_verify_batch": "http-nli-verify-batch",
    "compute_umap": "http-compute-umap",
    "score_clusters": "http-score-clusters",
}


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------


def _validate_service_token(
    credentials: HTTPAuthorizationCredentials = Depends(_bearer),
) -> str:
    expected = os.environ.get("ML_SERVICE_API_KEY")
    if expected is None:
        raise HTTPException(status_code=500, detail="ML_SERVICE_API_KEY not configured")
    if credentials.credentials != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return credentials.credentials


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class PipelineRunRequest(BaseModel):
    analysis_id: str
    prompt: str
    model_outputs: list[dict[str, Any]]


class PipelineStatus(BaseModel):
    status: str
    stage: str | None = None
    stages_completed: list[str] | None = None
    result: dict[str, Any] | None = None
    error: str | None = None


# ---------------------------------------------------------------------------
# Modal function builders
# ---------------------------------------------------------------------------


def _build_ml_functions() -> dict[str, Any]:
    """Build synchronous httpx callers for Modal endpoints."""
    prefix = _MODAL_ENDPOINT_PREFIX
    api_key = _MODAL_API_KEY

    def _make_caller(suffix: str, timeout: int):
        url = f"{prefix}-{suffix}.modal.run"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        def caller(payload: dict) -> dict:
            with httpx.Client(timeout=timeout) as client:
                resp = client.post(url, headers=headers, json=payload)
                resp.raise_for_status()
                return resp.json()

        return caller

    return {
        name: _make_caller(suffix, _TIMEOUTS[name])
        for name, suffix in _ENDPOINT_SUFFIXES.items()
    }


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------


def _run_pipeline_sync(
    analysis_id: str,
    prompt: str,
    model_outputs: list[dict[str, Any]],
) -> None:
    """Synchronous pipeline runner executed in a thread."""
    import sys

    sys.path.insert(0, "/app")
    from orchestrator import run_full_pipeline

    os.makedirs(_RESULTS_DIR, exist_ok=True)
    ml_functions = _build_ml_functions()
    api_keys: dict[str, str] = {}

    try:
        result = run_full_pipeline(
            analysis_id=analysis_id,
            prompt=prompt,
            api_keys=api_keys,
            ml_functions=ml_functions,
            results_dir=_RESULTS_DIR,
            model_outputs=model_outputs if model_outputs else None,
        )
        _PIPELINE_STATE[analysis_id] = {
            "status": "done",
            "result": result,
            "stage": "complete",
            "stages_completed": result.get("stages_completed", []),
            "error": None,
        }
    except Exception as exc:
        _PIPELINE_STATE[analysis_id] = {
            "status": "error",
            "result": None,
            "stage": "failed",
            "stages_completed": [],
            "error": str(exc),
        }


def _read_disk_state(analysis_id: str) -> dict[str, Any] | None:
    """Read progress/result from disk as a crash-recovery fallback."""
    result_path = os.path.join(_RESULTS_DIR, f"{analysis_id}.json")
    progress_path = os.path.join(_RESULTS_DIR, f"{analysis_id}_progress.json")

    if os.path.exists(result_path):
        with open(result_path) as f:
            data = json.load(f)
        if data.get("status") == "done":
            return {
                "status": "done",
                "result": data,
                "stage": "complete",
                "stages_completed": data.get("stages_completed", []),
                "error": None,
            }
        if data.get("status") == "error":
            return {
                "status": "error",
                "result": None,
                "stage": "failed",
                "stages_completed": [],
                "error": data.get("error", "unknown error"),
            }

    if os.path.exists(progress_path):
        with open(progress_path) as f:
            data = json.load(f)
        return {
            "status": data.get("status", "running"),
            "result": None,
            "stage": data.get("stage", ""),
            "stages_completed": data.get("stages_completed", []),
            "error": data.get("error") if data.get("status") == "error" else None,
        }

    return None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "ml"}


@app.post("/pipeline/run", response_model=PipelineStatus)
async def pipeline_run(
    req: PipelineRunRequest,
    _token: str = Depends(_validate_service_token),
) -> PipelineStatus:
    existing = _PIPELINE_STATE.get(req.analysis_id)
    if existing is not None:
        status = existing["status"]
        if status in ("running", "done"):
            return PipelineStatus(
                status=status,
                stage=existing.get("stage"),
                stages_completed=existing.get("stages_completed", []),
                result=existing.get("result"),
                error=existing.get("error"),
            )

    _PIPELINE_STATE[req.analysis_id] = {
        "status": "running",
        "result": None,
        "stage": "starting",
        "stages_completed": [],
        "error": None,
    }

    asyncio.create_task(
        asyncio.to_thread(
            _run_pipeline_sync,
            req.analysis_id,
            req.prompt,
            req.model_outputs,
        )
    )

    return PipelineStatus(
        status="running",
        stage="starting",
        stages_completed=[],
        result=None,
        error=None,
    )


@app.get("/pipeline/{analysis_id}", response_model=PipelineStatus)
async def pipeline_get(
    analysis_id: str,
    _token: str = Depends(_validate_service_token),
) -> PipelineStatus:
    state = _PIPELINE_STATE.get(analysis_id)

    if state is None:
        disk_state = _read_disk_state(analysis_id)
        if disk_state is None:
            raise HTTPException(status_code=404, detail="analysis_id not found")
        state = disk_state

    status = state["status"]
    if status == "done":
        result = state.get("result")
        return PipelineStatus(
            status="done",
            stage=state.get("stage"),
            stages_completed=state.get("stages_completed", []),
            result=result if result is not None else {},
            error=None,
        )
    if status == "error":
        error = state.get("error") or "unknown error"
        return PipelineStatus(
            status="error",
            stage=state.get("stage"),
            stages_completed=state.get("stages_completed", []),
            result=None,
            error=error,
        )
    return PipelineStatus(
        status="running",
        stage=state.get("stage"),
        stages_completed=state.get("stages_completed", []),
        result=None,
        error=None,
    )
