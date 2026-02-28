import os
from typing import AsyncIterator

from ..streaming import stream_static_text


async def stream_claude(prompt: str) -> AsyncIterator[str]:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        msg = "[claude unavailable: set ANTHROPIC_API_KEY]"
        async for chunk in stream_static_text(msg):
            yield chunk
        return

    # Placeholder until the direct Anthropic streaming client is wired in.
    msg = f"[claude stub] prompt received ({len(prompt)} chars)"
    async for chunk in stream_static_text(msg):
        yield chunk
