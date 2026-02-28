from typing import AsyncIterator, Awaitable, Callable
import asyncio
import os

from .schemas import EventEnvelope, SCHEMA_VERSION
from .providers.openai import stream_openai
from .providers.gemini import stream_gemini
from .providers.claude import stream_claude
from .modal_calls.kimi import stream_kimi
from .modal_calls.llama import stream_llama
from .ml_client import run_ml_pipeline

StreamerFn = Callable[[str], AsyncIterator[str]]
PublishFn = Callable[[str, dict], Awaitable[None]]
MODEL_TIMEOUT_SECONDS = 90

def _is_enabled(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _get_model_streamers() -> list[tuple[str, StreamerFn]]:
    streamers: list[tuple[str, StreamerFn]] = [
        ("openai_gpt4", stream_openai),
        ("gemini_2_0", stream_gemini),
        ("claude_sonnet_4", stream_claude),
    ]
    if _is_enabled("ENABLE_KIMI"):
        streamers.append(("kimi", stream_kimi))
    if _is_enabled("ENABLE_LLAMA"):
        streamers.append(("llama_3_8b", stream_llama))
    return streamers


async def _run_model(
    analysis_id: str,
    prompt: str,
    model_id: str,
    streamer: StreamerFn,
    publish: PublishFn,
) -> dict:
    await publish(
        analysis_id,
        EventEnvelope(
            analysis_id=analysis_id,
            type="MODEL_STARTED",
            payload={"model_id": model_id},
        ).model_dump(),
    )

    full_text = ""
    warning = None
    try:
        async with asyncio.timeout(MODEL_TIMEOUT_SECONDS):
            async for delta in streamer(prompt):
                full_text += delta
                await publish(
                    analysis_id,
                    EventEnvelope(
                        analysis_id=analysis_id,
                        type="MODEL_TOKEN",
                        payload={"model_id": model_id, "delta": delta},
                    ).model_dump(),
                )
    except TimeoutError:
        warning = (
            f"{model_id} stream timed out after {MODEL_TIMEOUT_SECONDS}s"
        )
        await publish(
            analysis_id,
            EventEnvelope(
                analysis_id=analysis_id,
                type="STAGE_FAILED",
                payload={"stage": f"{model_id}_stream", "message": warning},
            ).model_dump(),
        )
    except Exception as exc:
        warning = f"{model_id} stream failed: {exc}"
        await publish(
            analysis_id,
            EventEnvelope(
                analysis_id=analysis_id,
                type="STAGE_FAILED",
                payload={"stage": f"{model_id}_stream", "message": warning},
            ).model_dump(),
        )

    await publish(
        analysis_id,
        EventEnvelope(
            analysis_id=analysis_id,
            type="MODEL_DONE",
            payload={"model_id": model_id, "response_text": full_text},
        ).model_dump(),
    )

    return {
        "model_id": model_id,
        "response_text": full_text,
        "warning": warning,
    }


async def run_pipeline(analysis_id: str, prompt: str, publish: PublishFn) -> None:
    try:
        tasks = []
        for model_id, streamer in _get_model_streamers():
            tasks.append(
                _run_model(
                    analysis_id=analysis_id,
                    prompt=prompt,
                    model_id=model_id,
                    streamer=streamer,
                    publish=publish,
                )
            )
        model_results = await asyncio.gather(*tasks)

        warnings: list[str] = []
        models: list[dict] = []
        for result in model_results:
            models.append(
                {
                    "model_id": result["model_id"],
                    "response_text": result["response_text"],
                }
            )
            if result["warning"]:
                warnings.append(result["warning"])

        ml_result: dict = {}
        try:
            ml_result = await run_ml_pipeline(
                analysis_id=analysis_id,
                model_outputs=models,
                publish=publish,
            )
            warnings.extend(ml_result.get("warnings", []) or [])
        except Exception as exc:
            ml_warning = f"ml pipeline failed: {exc}"
            warnings.append(ml_warning)
            await publish(
                analysis_id,
                EventEnvelope(
                    analysis_id=analysis_id,
                    type="STAGE_FAILED",
                    payload={"stage": "ml_pipeline", "message": ml_warning},
                ).model_dump(),
            )

        await publish(
            analysis_id,
            EventEnvelope(
                analysis_id=analysis_id,
                type="DONE",
                payload={
                    "result": {
                        "schema_version": SCHEMA_VERSION,
                        "analysis_id": analysis_id,
                        "prompt": prompt,
                        "models": models,
                        "ml": ml_result,
                        "warnings": warnings,
                    }
                },
            ).model_dump(),
        )

    except Exception as exc:
        # Always surface fatal errors instead of dying silently.
        await publish(
            analysis_id,
            EventEnvelope(
                analysis_id=analysis_id,
                type="FATAL_ERROR",
                payload={"message": repr(exc)},
            ).model_dump(),
        )
