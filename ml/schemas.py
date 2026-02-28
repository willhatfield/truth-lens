"""Pydantic request/response models for the TruthLens ML pipeline."""

from typing import List, Optional

from pydantic import BaseModel, Field


# ── Embed Claims ──────────────────────────────────────────────────────────────

class EmbedClaimsRequest(BaseModel):
    """Input for the embed_claims Modal function."""

    claim_texts: List[str] = Field(
        ..., min_length=1, description="Non-empty list of claim strings to embed."
    )
    batch_size: int = Field(
        default=32, ge=1, le=512, description="Batch size for encoding."
    )


class EmbedClaimsResponse(BaseModel):
    """Output from embed_claims."""

    vectors: List[List[float]] = Field(default_factory=list)
    dimension: int = Field(default=0)
    model_name: str = Field(default="")
    error: Optional[str] = Field(default=None)


# ── Cluster Claims ────────────────────────────────────────────────────────────

class ClusterClaimsRequest(BaseModel):
    """Input for the cluster_claims Modal function."""

    vectors: List[List[float]] = Field(
        ..., min_length=1, description="Embedding vectors to cluster."
    )
    threshold: float = Field(
        default=0.5, gt=0.0, le=2.0, description="Distance threshold."
    )


class ClusterClaimsResponse(BaseModel):
    """Output from cluster_claims."""

    clusters: List[List[int]] = Field(default_factory=list)
    num_clusters: int = Field(default=0)
    error: Optional[str] = Field(default=None)


# ── Rerank Evidence ───────────────────────────────────────────────────────────

class RankedPassage(BaseModel):
    """A single passage with its reranking score and original index."""

    index: int = Field(..., ge=0)
    text: str
    score: float


class RerankEvidenceRequest(BaseModel):
    """Input for the rerank_evidence Modal function."""

    claim: str = Field(..., min_length=1, description="The claim to rerank against.")
    passages: List[str] = Field(
        ..., min_length=1, description="Passages to rerank."
    )
    top_k: int = Field(
        default=5, ge=1, le=100, description="Number of top passages to return."
    )


class RerankEvidenceResponse(BaseModel):
    """Output from rerank_evidence."""

    ranked_passages: List[RankedPassage] = Field(default_factory=list)
    error: Optional[str] = Field(default=None)


# ── NLI Verify ────────────────────────────────────────────────────────────────

class NliPair(BaseModel):
    """A premise-hypothesis pair for NLI inference."""

    premise: str = Field(..., min_length=1)
    hypothesis: str = Field(..., min_length=1)


class NliResult(BaseModel):
    """NLI prediction for a single pair."""

    label: str = Field(default="neutral")
    scores: dict = Field(default_factory=dict)


class NliVerifyRequest(BaseModel):
    """Input for the nli_verify Modal function."""

    pairs: List[NliPair] = Field(
        ..., min_length=1, description="Premise-hypothesis pairs."
    )
    batch_size: int = Field(
        default=16, ge=1, le=256, description="Batch size for NLI."
    )


class NliVerifyResponse(BaseModel):
    """Output from nli_verify."""

    results: List[NliResult] = Field(default_factory=list)
    error: Optional[str] = Field(default=None)


# ── Compute UMAP ─────────────────────────────────────────────────────────────

class UmapPoint(BaseModel):
    """3-D UMAP coordinate for a single vector."""

    x: float = Field(default=0.0)
    y: float = Field(default=0.0)
    z: float = Field(default=0.0)


class ComputeUmapRequest(BaseModel):
    """Input for the compute_umap Modal function."""

    vectors: List[List[float]] = Field(
        ..., min_length=1, description="High-dimensional vectors."
    )
    n_neighbors: int = Field(
        default=15, ge=2, le=200, description="UMAP n_neighbors parameter."
    )
    min_dist: float = Field(
        default=0.1, gt=0.0, le=1.0, description="UMAP min_dist parameter."
    )


class ComputeUmapResponse(BaseModel):
    """Output from compute_umap."""

    coords_3d: List[UmapPoint] = Field(default_factory=list)
    error: Optional[str] = Field(default=None)
