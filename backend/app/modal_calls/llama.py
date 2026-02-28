from typing import AsyncIterator
import asyncio

from ..streaming import stream_static_text

MODAL_APP_NAME = "truthlens-ml"
LLAMA_FUNCTION_NAME = "generate_llama"
LLAMA_CLASS_NAME = "LlamaGenerator"


def _extract_text(result) -> str:
    if isinstance(result, dict):
        if "response_text" in result:
            return str(result["response_text"])
        if "text" in result:
            return str(result["text"])
    return str(result)


async def stream_llama(prompt: str) -> AsyncIterator[str]:
    try:
        import modal

        # Prefer class-based generator with warm model load.
        try:
            cls = modal.Cls.from_name(MODAL_APP_NAME, LLAMA_CLASS_NAME)
            runner = cls()
            result = await asyncio.to_thread(
                runner.generate.remote, {"prompt": prompt}
            )
        except Exception:
            fn = modal.Function.from_name(MODAL_APP_NAME, LLAMA_FUNCTION_NAME)
            result = await asyncio.to_thread(fn.remote, {"prompt": prompt})
        text = _extract_text(result)
        async for chunk in stream_static_text(text):
            yield chunk
    except Exception:
        msg = "[llama unavailable: deploy Modal function 'generate_llama']"
        async for chunk in stream_static_text(msg):
            yield chunk
