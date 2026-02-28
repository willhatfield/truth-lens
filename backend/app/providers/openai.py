import os
from typing import AsyncIterator, Optional

from openai import AsyncOpenAI

from ..streaming import stream_static_text


_client: Optional[AsyncOpenAI] = None

def _get_openai_client() -> Optional[AsyncOpenAI]:
    global _client
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    if _client is None:
        _client = AsyncOpenAI(api_key=api_key)
    return _client


async def stream_openai(
    prompt: str,
    *,
    model: str = "gpt-5.2",
    instructions: Optional[str] = None,
) -> AsyncIterator[str]:
    client = _get_openai_client()
    if client is None:
        async for chunk in stream_static_text("[openai unavailable: set OPENAI_API_KEY]"):
            yield chunk
        return

    try:
        stream = await client.responses.create(
            model=model,
            input=prompt,
            instructions=instructions,
            stream=True,
        )

        async for event in stream:
            etype = getattr(event, "type", None) or (event.get("type") if isinstance(event, dict) else None)

            if etype == "response.output_text.delta":
                delta = getattr(event, "delta", None) or (event.get("delta") if isinstance(event, dict) else "")
                if delta:
                    yield delta

            elif etype == "error":
                err = getattr(event, "error", None) or (event.get("error") if isinstance(event, dict) else None)
                msg = getattr(err, "message", None) if err else None
                async for chunk in stream_static_text(f"[openai error] {msg or 'unknown'}"):
                    yield chunk
                return

    except Exception as exc:
        async for chunk in stream_static_text(f"[openai streaming failed] {exc}"):
            yield chunk