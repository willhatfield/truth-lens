from typing import AsyncIterator, Awaitable, Callable
import asyncio

from .schemas import EventEnvelope, SCHEMA_VERSION
from .providers.openai import stream_openai
from .providers.gemini import stream_gemini
from .providers.claude import stream_claude
from .modal_calls.kimi import stream_kimi
from .modal_calls.llama import stream_llama

StreamerFn = Callable[[str], AsyncIterator[str]]
PublishFn = Callable[[str, dict], Awaitable[None]]

MODEL_STREAMERS: list[tuple[str, StreamerFn]] = [
    ("openai_gpt4", stream_openai),
    ("gemini_2_0", stream_gemini),
    ("claude_sonnet_4", stream_claude),
    ("kimi", stream_kimi),
    ("llama_3_8b", stream_llama),
]


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
        for model_id, streamer in MODEL_STREAMERS:
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
