import asyncio
from typing import AsyncIterator


def chunk_text(text: str, chunk_size: int = 32) -> list[str]:
    if chunk_size <= 0:
        chunk_size = 32
    if not text:
        return [""]
    chunks: list[str] = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i : i + chunk_size])
    return chunks


async def stream_static_text(
    text: str, chunk_size: int = 32, delay_seconds: float = 0.0
) -> AsyncIterator[str]:
    for chunk in chunk_text(text, chunk_size=chunk_size):
        if delay_seconds > 0:
            await asyncio.sleep(delay_seconds)
        else:
            await asyncio.sleep(0)
        yield chunk

