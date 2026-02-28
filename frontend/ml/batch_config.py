"""Named batch-size profiles for ML inference pipelines.

Each BatchProfile bundles the batch sizes used by embed, NLI, and
rerank stages together with a chunk_max_batches upper bound.
"""

from pydantic import BaseModel, Field

from batch_utils import MAX_BATCHES_LIMIT


class BatchProfile(BaseModel):
    """Validated batch-size configuration for one inference run."""

    embed_batch_size: int = Field(..., gt=0, description="Sentence-transformer encoding batch size")
    nli_batch_size: int = Field(..., gt=0, description="NLI inference batch size")
    rerank_batch_size: int = Field(..., gt=0, description="Cross-encoder reranking batch size")
    chunk_max_batches: int = Field(
        ...,
        gt=0,
        le=MAX_BATCHES_LIMIT,
        description="Upper bound passed to chunk_list",
    )


# ── Named profiles ──────────────────────────────────────────────────────────

CONSERVATIVE = BatchProfile(
    embed_batch_size=32,
    nli_batch_size=8,
    rerank_batch_size=16,
    chunk_max_batches=500,
)

BALANCED = BatchProfile(
    embed_batch_size=64,
    nli_batch_size=16,
    rerank_batch_size=32,
    chunk_max_batches=1000,
)

AGGRESSIVE = BatchProfile(
    embed_batch_size=128,
    nli_batch_size=32,
    rerank_batch_size=64,
    chunk_max_batches=2000,
)

_PROFILES = {
    "conservative": CONSERVATIVE,
    "balanced": BALANCED,
    "aggressive": AGGRESSIVE,
}

MAX_PROFILES = 10  # bounded iteration limit for profile lookup


def get_profile(name: str) -> BatchProfile:
    """Return a named profile (case-insensitive, whitespace-stripped).

    Falls back to BALANCED for unrecognised names.
    """
    key = name.strip().lower()
    matched = _PROFILES.get(key)
    if matched is not None:
        return matched
    return BALANCED
