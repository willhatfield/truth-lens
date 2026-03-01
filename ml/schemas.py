"""Pydantic request/response models for the TruthLens ML pipeline.

32 models total, organized by pipeline function.
All requests inherit BaseRequest (schema_version, analysis_id).
All responses inherit BaseResponse (schema_version, analysis_id, warnings).
"""

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


# ── Base Classes ─────────────────────────────────────────────────────────────

class BaseRequest(BaseModel):
    """Common fields for all pipeline requests."""
    schema_version: str = Field(default="1.0")
    analysis_id: str = Field(..., min_length=1)


class BaseResponse(BaseModel):
    """Common fields for all pipeline responses."""
    schema_version: str = Field(default="1.0")
    analysis_id: str = Field(..., min_length=1)
    warnings: List[str] = Field(default_factory=list)


# ── extract_claims Models ────────────────────────────────────────────────────

class ModelResponse(BaseModel):
    """A single LLM response to decompose into claims."""
    model_id: str = Field(..., min_length=1)
    response_text: str = Field(..., min_length=1)


class ClaimSpan(BaseModel):
    """Character offsets into the original response_text."""
    start: int = Field(..., ge=0)
    end: int = Field(..., ge=0)


class Claim(BaseModel):
    """An atomic factual claim extracted from a model response."""
    claim_id: str = Field(..., min_length=1)
    model_id: str = Field(..., min_length=1)
    claim_text: str = Field(..., min_length=1)
    span: Optional[ClaimSpan] = Field(default=None)


class ExtractClaimsRequest(BaseRequest):
    """Input for the extract_claims Modal function."""
    responses: List[ModelResponse] = Field(..., min_length=1)


class ExtractClaimsResponse(BaseResponse):
    """Output from extract_claims."""
    claims: List[Claim] = Field(default_factory=list)


# ── embed_claims Models ──────────────────────────────────────────────────────

class ClaimInput(BaseModel):
    """Claim ID + text pair for embedding."""
    claim_id: str = Field(..., min_length=1)
    claim_text: str = Field(..., min_length=1)


class EmbedClaimsRequest(BaseRequest):
    """Input for the embed_claims Modal function."""
    claims: List[ClaimInput] = Field(..., min_length=1)
    model_name: str = Field(default="BAAI/bge-large-en-v1.5")


class EmbedClaimsResponse(BaseResponse):
    """Output from embed_claims."""
    vectors: Dict[str, List[float]] = Field(default_factory=dict)
    dim: int = Field(default=0)


# ── cluster_claims Models ────────────────────────────────────────────────────

class ClaimMetadata(BaseModel):
    """Metadata for a claim used during clustering."""
    model_id: str = Field(..., min_length=1)
    claim_text: str = Field(..., min_length=1)


class Cluster(BaseModel):
    """A group of semantically similar claims."""
    cluster_id: str = Field(..., min_length=1)
    claim_ids: List[str] = Field(..., min_length=1)
    representative_claim_id: str = Field(..., min_length=1)
    representative_text: str = Field(..., min_length=1)


class ClusterClaimsRequest(BaseRequest):
    """Input for the cluster_claims Modal function."""
    vectors: Dict[str, List[float]] = Field(..., min_length=1)
    claims: Dict[str, ClaimMetadata] = Field(..., min_length=1)
    sim_threshold: float = Field(default=0.85, gt=0.0, le=1.0)


class ClusterClaimsResponse(BaseResponse):
    """Output from cluster_claims."""
    clusters: List[Cluster] = Field(default_factory=list)


# ── rerank_evidence_batch Models ─────────────────────────────────────────────

class PassageInput(BaseModel):
    """A candidate evidence passage."""
    passage_id: str = Field(..., min_length=1)
    text: str = Field(..., min_length=1)


class RerankItem(BaseModel):
    """A single claim with its candidate passages for reranking."""
    claim_id: str = Field(..., min_length=1)
    claim_text: str = Field(..., min_length=1)
    passages: List[PassageInput] = Field(..., min_length=1)


class ClaimRanking(BaseModel):
    """Reranked result for a single claim."""
    claim_id: str = Field(..., min_length=1)
    ordered_passage_ids: List[str] = Field(default_factory=list)
    scores: Dict[str, float] = Field(default_factory=dict)


class RerankEvidenceBatchRequest(BaseRequest):
    """Input for the rerank_evidence_batch Modal function."""
    items: List[RerankItem] = Field(..., min_length=1)
    reranker_model: str = Field(default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    top_k: int = Field(default=10, ge=1, le=100)


class RerankEvidenceBatchResponse(BaseResponse):
    """Output from rerank_evidence_batch."""
    rankings: List[ClaimRanking] = Field(default_factory=list)


# ── nli_verify_batch Models ──────────────────────────────────────────────────

class NliPairInput(BaseModel):
    """A (claim, passage) pair for NLI classification."""
    pair_id: str = Field(..., min_length=1)
    claim_id: str = Field(..., min_length=1)
    passage_id: str = Field(..., min_length=1)
    claim_text: str = Field(..., min_length=1)
    passage_text: str = Field(..., min_length=1)


class NliResultOutput(BaseModel):
    """NLI prediction for a single pair."""
    pair_id: str = Field(..., min_length=1)
    claim_id: str = Field(..., min_length=1)
    passage_id: str = Field(..., min_length=1)
    label: str = Field(default="neutral")
    probs: Dict[str, float] = Field(default_factory=dict)


class NliVerifyBatchRequest(BaseRequest):
    """Input for the nli_verify_batch Modal function."""
    pairs: List[NliPairInput] = Field(..., min_length=1)
    nli_model: str = Field(
        default="cross-encoder/nli-deberta-v3-large"
    )
    batch_size: int = Field(default=16, ge=1, le=256)


class NliVerifyBatchResponse(BaseResponse):
    """Output from nli_verify_batch."""
    results: List[NliResultOutput] = Field(default_factory=list)


# ── compute_umap Models ─────────────────────────────────────────────────────

class ComputeUmapRequest(BaseRequest):
    """Input for the compute_umap Modal function."""
    vectors: Dict[str, List[float]] = Field(..., min_length=1)
    random_state: int = Field(default=42)
    n_neighbors: int = Field(default=15, ge=2, le=200)
    min_dist: float = Field(default=0.1, gt=0.0, le=1.0)


class ComputeUmapResponse(BaseResponse):
    """Output from compute_umap."""
    coords3d: Dict[str, List[float]] = Field(default_factory=dict)


# ── score_clusters Models ────────────────────────────────────────────────────

class ScoringWeights(BaseModel):
    """Weights for agreement vs verification in trust score."""
    agreement_weight: float = Field(default=0.4, ge=0.0, le=1.0)
    verification_weight: float = Field(default=0.6, ge=0.0, le=1.0)


class VerdictThresholds(BaseModel):
    """Thresholds for verdict classification."""
    safe_min: int = Field(default=75, ge=0, le=100)
    caution_min: int = Field(default=45, ge=0, le=100)


class AgreementDetail(BaseModel):
    """Agreement info for a cluster score."""
    models_supporting: List[str] = Field(default_factory=list)
    count: int = Field(default=0, ge=0)


class VerificationDetail(BaseModel):
    """NLI verification info for a cluster score."""
    best_entailment_prob: float = Field(default=0.0, ge=0.0, le=1.0)
    best_contradiction_prob: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence_passage_id: str = Field(default="")


class ClusterScore(BaseModel):
    """Trust score + verdict for a single cluster."""
    cluster_id: str = Field(..., min_length=1)
    trust_score: int = Field(default=0, ge=0, le=100)
    verdict: str = Field(default="REJECT")
    agreement: AgreementDetail = Field(default_factory=AgreementDetail)
    verification: VerificationDetail = Field(default_factory=VerificationDetail)


class ScoreClustersRequest(BaseRequest):
    """Input for the score_clusters Modal function."""
    clusters: List[Cluster] = Field(..., min_length=1)
    claims: Dict[str, ClaimMetadata] = Field(..., min_length=1)
    nli_results: List[NliResultOutput] = Field(default_factory=list)
    weights: ScoringWeights = Field(default_factory=ScoringWeights)
    verdict_thresholds: VerdictThresholds = Field(default_factory=VerdictThresholds)


class ScoreClustersResponse(BaseResponse):
    """Output from score_clusters."""
    scores: List[ClusterScore] = Field(default_factory=list)
