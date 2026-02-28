import os
from typing import AsyncIterator

from ..streaming import stream_static_text


async def stream_gemini(prompt: str) -> AsyncIterator[str]:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        msg = "[gemini unavailable: set GEMINI_API_KEY]"
        async for chunk in stream_static_text(msg):
            yield chunk
        return

    # Placeholder until the direct Gemini streaming client is wired in.
    msg = f"[gemini stub] prompt received ({len(prompt)} chars)"
    async for chunk in stream_static_text(msg):
        yield chunk
