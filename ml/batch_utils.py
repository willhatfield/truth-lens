"""Reusable chunking helpers for batched ML inference."""

from typing import List, TypeVar

T = TypeVar("T")

MAX_BATCHES_LIMIT = 10000


def chunk_list(
    items: List[T],
    batch_size: int,
    max_batches: int = 1000,
) -> List[List[T]]:
    """Split *items* into batches of *batch_size*.

    Uses a bounded loop capped at *max_batches* iterations.
    Returns at most max_batches chunks; remaining items are dropped.
    """
    if batch_size < 1:
        batch_size = 1
    if max_batches < 1:
        max_batches = 1
    if max_batches > MAX_BATCHES_LIMIT:
        max_batches = MAX_BATCHES_LIMIT

    chunks: List[List[T]] = []
    total = len(items)

    idx = 0
    for _ in range(max_batches):
        if idx >= total:
            break
        end = idx + batch_size
        if end > total:
            end = total
        chunks.append(items[idx:end])
        idx = end

    return chunks


def flatten_batch_results(
    batched: List[List[T]],
    max_batches: int = 1000,
) -> List[T]:
    """Flatten a list-of-lists into a single list.

    Uses a bounded loop capped at *max_batches* iterations.
    """
    if max_batches < 1:
        max_batches = 1
    if max_batches > MAX_BATCHES_LIMIT:
        max_batches = MAX_BATCHES_LIMIT

    flat: List[T] = []

    for i in range(max_batches):
        if i >= len(batched):
            break
        batch = batched[i]
        batch_len = len(batch)
        for j in range(batch_len):
            flat.append(batch[j])

    return flat
