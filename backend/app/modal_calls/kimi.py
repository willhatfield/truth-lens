from typing import AsyncIterator
import asyncio

from ..streaming import stream_static_text

MODAL_APP_NAME = "truthlens-ml"
KIMI_FUNCTION_NAME = "generate_kimi"


def _extract_text(result) -> str:
    if isinstance(result, dict):
        if "response_text" in result:
            return str(result["response_text"])
        if "text" in result:
            return str(result["text"])
    return str(result)


async def stream_kimi(prompt: str) -> AsyncIterator[str]:
    try:
        import modal

        fn = modal.Function.from_name(MODAL_APP_NAME, KIMI_FUNCTION_NAME)
        result = await asyncio.to_thread(fn.remote, {"prompt": prompt})
        text = _extract_text(result)
        async for chunk in stream_static_text(text):
            yield chunk
    except Exception:
        msg = "[kimi unavailable: deploy Modal function 'generate_kimi']"
        async for chunk in stream_static_text(msg):
            yield chunk
