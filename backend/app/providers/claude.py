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


async def stream_claude(prompt: str) -> AsyncIterator[str]:
    api_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("CLAUDE_API_KEY")
    if not api_key:
        msg = "[claude unavailable: set ANTHROPIC_API_KEY or CLAUDE_API_KEY]"
        async for chunk in stream_static_text(msg):
            yield chunk
        return

    try:
        text = await asyncio.to_thread(
            _generate_claude_text,
            prompt,
            "claude-3-5-sonnet-latest",
            api_key,
        )
        async for chunk in stream_static_text(text):
            yield chunk
    except Exception as exc:
        async for chunk in stream_static_text(f"[claude streaming failed] {exc}"):
            yield chunk
