import os
from typing import AsyncIterator, Optional

from openai import AsyncOpenAI

from ..streaming import stream_static_text


_client: Optional[AsyncOpenAI] = None

def _get_openai_client() -> AsyncOpenAI:
    global _client
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    if _client is None:
        _client = AsyncOpenAI(api_key=api_key)
    return _client


async def stream_openai(
    prompt: str,
    *,
    model: str = None,
    instructions: Optional[str] = None,
) -> AsyncIterator[str]:
    if model is None:
        model = os.getenv("OPENAI_MODEL", "gpt-4o")
    client = _get_openai_client()
    kwargs = {"model": model, "input": prompt, "stream": True}
    if instructions is not None:
        kwargs["instructions"] = instructions
    stream = await client.responses.create(**kwargs)

    async for event in stream:
        etype = getattr(event, "type", None) or (event.get("type") if isinstance(event, dict) else None)

        if etype == "response.output_text.delta":
            delta = getattr(event, "delta", None) or (event.get("delta") if isinstance(event, dict) else "")
            if delta:
                yield delta

        elif etype == "error":
            err = getattr(event, "error", None) or (event.get("error") if isinstance(event, dict) else None)
            msg = getattr(err, "message", None) if err else None
            raise RuntimeError(f"OpenAI stream error: {msg or 'unknown'}")
