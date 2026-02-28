import os
from typing import AsyncIterator
import asyncio

from ..streaming import stream_static_text


def _generate_gemini_text(prompt: str, model: str, api_key: str) -> str:
    import google.generativeai as genai

    genai.configure(api_key=api_key)
    generator = genai.GenerativeModel(model)
    response = generator.generate_content(prompt, stream=True)

    parts: list[str] = []
    for chunk in response:
        text = getattr(chunk, "text", None)
        if text:
            parts.append(text)
    return "".join(parts)


async def stream_gemini(prompt: str) -> AsyncIterator[str]:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        msg = "[gemini unavailable: set GEMINI_API_KEY]"
        async for chunk in stream_static_text(msg):
            yield chunk
        return

    try:
        text = await asyncio.to_thread(
            _generate_gemini_text,
            prompt,
            "gemini-2.0-flash",
            api_key,
        )
        async for chunk in stream_static_text(text):
            yield chunk
    except Exception as exc:
        async for chunk in stream_static_text(f"[gemini streaming failed] {exc}"):
            yield chunk
