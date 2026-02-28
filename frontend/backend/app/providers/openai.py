import os
from typing import AsyncIterator

from ..streaming import stream_static_text


async def stream_openai(prompt: str) -> AsyncIterator[str]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        msg = "[openai unavailable: set OPENAI_API_KEY]"
        async for chunk in stream_static_text(msg):
            yield chunk
        return

    # Placeholder until the direct OpenAI streaming client is wired in.
    msg = f"[openai stub] prompt received ({len(prompt)} chars)"
    async for chunk in stream_static_text(msg):
        yield chunk
