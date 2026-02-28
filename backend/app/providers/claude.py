import os
from typing import AsyncIterator
import asyncio

from ..streaming import stream_static_text


def _generate_claude_text(prompt: str, model: str, api_key: str) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    parts: list[str] = []
    with client.messages.stream(
        model=model,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        for text in stream.text_stream:
            if text:
                parts.append(text)
    return "".join(parts)


def _candidate_models() -> list[str]:
    configured = os.getenv("CLAUDE_MODEL", "").strip()
    if configured:
        return [configured]
    return [
        "claude-sonnet-4-20250514",
        "claude-3-7-sonnet-latest",
        "claude-3-5-sonnet-latest",
    ]


async def stream_claude(prompt: str) -> AsyncIterator[str]:
    api_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("CLAUDE_API_KEY")
    if not api_key:
        msg = "[claude unavailable: set ANTHROPIC_API_KEY or CLAUDE_API_KEY]"
        async for chunk in stream_static_text(msg):
            yield chunk
        return

    last_exc: Exception | None = None
    for model_name in _candidate_models():
        try:
            text = await asyncio.to_thread(
                _generate_claude_text,
                prompt,
                model_name,
                api_key,
            )
            async for chunk in stream_static_text(text):
                yield chunk
            return
        except Exception as exc:
            last_exc = exc
            if "not_found_error" in str(exc):
                continue
            async for chunk in stream_static_text(f"[claude streaming failed] {exc}"):
                yield chunk
            return

    async for chunk in stream_static_text(
        f"[claude streaming failed] {last_exc or 'no usable Claude model configured'}"
    ):
        yield chunk
