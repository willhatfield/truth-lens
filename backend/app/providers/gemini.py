import os
from typing import AsyncIterator
import asyncio

from ..streaming import stream_static_text


def _generate_gemini_text(prompt: str, model: str, api_key: str) -> str:
    from google import genai

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model,
        contents=prompt,
    )
    text = response.text or ""
    if not text:
        raise RuntimeError("Gemini returned empty response")
    return text


async def stream_gemini(prompt: str) -> AsyncIterator[str]:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set")

    text = await asyncio.to_thread(
        _generate_gemini_text,
        prompt,
        "gemini-2.0-flash",
        api_key,
    )
    async for chunk in stream_static_text(text):
        yield chunk
