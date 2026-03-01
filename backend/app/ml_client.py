import asyncio
import os
from typing import Any, Awaitable, Callable

import httpx

from .schemas import EventEnvelope

PublishFn = Callable[[str, dict], Awaitable[None]]


def _ml_service_url() -> str:
    url = os.getenv("ML_SERVICE_URL", "").strip().rstrip("/")
    if not url:
        raise RuntimeError("ML_SERVICE_URL is not set")
    return url


def _ml_auth_headers() -> dict[str, str]:
    key = os.getenv("ML_SERVICE_API_KEY", "").strip()
    if not key:
        raise RuntimeError("ML_SERVICE_API_KEY is not set")
    return {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }


async def run_ml_pipeline(
    analysis_id: str,
    model_outputs: list[dict[str, str]],
    prompt: str = "",
    passages_by_claim: dict | None = None,
    publish: PublishFn | None = None,
) -> dict[str, Any]:
    filtered_responses = []
    for item in model_outputs:
        text = str(item.get("response_text", "")).strip()
        if not text:
            continue
        filtered_responses.append(
            {
                "model_id": str(item.get("model_id", "")),
                "response_text": text,
            }
        )

    if not filtered_responses:
        return {"warnings": ["ml pipeline skipped: no non-empty model responses"]}

    base_url = _ml_service_url()
    headers = _ml_auth_headers()

    async with httpx.AsyncClient(timeout=30.0) as client:
        start_resp = await client.post(
            f"{base_url}/pipeline/run",
            headers=headers,
            json={
                "analysis_id": analysis_id,
                "prompt": prompt,
                "model_outputs": filtered_responses,
            },
        )
        start_resp.raise_for_status()
        start_data = start_resp.json()

    status = start_data.get("status", "")
    if status == "done":
        result = start_data.get("result")
        return result if result is not None else {}
    if status == "error":
        error = start_data.get("error", "unknown ml error")
        return {"warnings": [f"ml pipeline error: {error}"]}

    seen_stages: list[str] = list(start_data.get("stages_completed", []) or [])
    max_polls = 240
    poll_count = 0

    while poll_count < max_polls:
        await asyncio.sleep(5)
        poll_count += 1

        async with httpx.AsyncClient(timeout=30.0) as client:
            poll_resp = await client.get(
                f"{base_url}/pipeline/{analysis_id}",
                headers=headers,
            )
            poll_resp.raise_for_status()
            poll_data = poll_resp.json()

        current_stages = list(poll_data.get("stages_completed", []) or [])
        if publish is not None and len(current_stages) > len(seen_stages):
            for stage in current_stages[len(seen_stages):]:
                envelope = EventEnvelope(
                    analysis_id=analysis_id,
                    type="STAGE_PROGRESS",
                    payload={"stage": stage},
                )
                await publish(analysis_id, envelope.model_dump())
        seen_stages = current_stages

        poll_status = poll_data.get("status", "")
        if poll_status == "done":
            result = poll_data.get("result")
            return result if result is not None else {}
        if poll_status == "error":
            error = poll_data.get("error", "unknown ml error")
            return {"warnings": [f"ml pipeline error: {error}"]}

    return {"warnings": ["ml pipeline timed out after 1200s"]}
