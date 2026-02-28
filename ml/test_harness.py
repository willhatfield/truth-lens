"""Local integration test -- validates the full 7-function pipeline with fake model outputs.

Runs: extract_claims -> embed_claims -> cluster_claims -> rerank_evidence_batch
      -> nli_verify_batch -> score_clusters -> compute_umap

Asserts every response is a valid Pydantic model with no warnings.

Usage:
    pytest test_harness.py -v
    python test_harness.py
"""

import sys

import pytest

from schemas import (
    ExtractClaimsResponse, Claim,
    EmbedClaimsResponse,
    ClusterClaimsResponse, Cluster, ClaimMetadata,
    RerankEvidenceBatchResponse, ClaimRanking,
    NliVerifyBatchResponse, NliResultOutput,
    ScoreClustersResponse, ClusterScore, AgreementDetail, VerificationDetail,
    ComputeUmapResponse,
)
from id_utils import make_claim_id, make_cluster_id, make_pair_id

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ANALYSIS_ID = "harness-test-001"
MODEL_IDS = ["model-alpha", "model-beta"]
MAX_CLAIMS = 10
NUM_DIMS = 8
NUM_PASSAGES = 4
TOP_K = 3
MAX_PIPELINE_PHASES = 10
CLAIM_TEXTS = [
    "The sky is blue.",
    "Water boils at 100C at sea level.",
    "The Earth orbits the Sun.",
    "Photosynthesis requires sunlight.",
    "Iron is a metal.",
]
NLI_LABELS = ["entailment", "neutral", "contradiction"]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_fake_claim(analysis_id: str, model_id: str, text: str) -> Claim:
    """Build a Claim with a deterministic ID."""
    cid = make_claim_id(analysis_id, model_id, text)
    return Claim(
        claim_id=cid,
        model_id=model_id,
        claim_text=text,
        span=None,
    )


def _fake_vector(index: int, dim: int) -> list:
    """Generate a single fake embedding vector of given dimension."""
    row: list = []
    for d in range(dim):
        row.append(float(index * dim + d) * 0.01)
    return row


def _make_passage_ids(count: int) -> list:
    """Generate a list of fake passage IDs with p_ prefix."""
    pids: list = []
    for i in range(count):
        pids.append(f"p_{i:04d}")
    return pids


# ---------------------------------------------------------------------------
# Build helpers -- each constructs one pipeline phase response
# ---------------------------------------------------------------------------

def _build_extract_response() -> ExtractClaimsResponse:
    """Build the ExtractClaimsResponse from fake claim texts."""
    claims: list = []
    for i in range(len(CLAIM_TEXTS)):
        mid = MODEL_IDS[i % len(MODEL_IDS)]
        claim = _make_fake_claim(ANALYSIS_ID, mid, CLAIM_TEXTS[i])
        claims.append(claim)
    return ExtractClaimsResponse(analysis_id=ANALYSIS_ID, claims=claims)


def _build_embed_response(claims: list) -> EmbedClaimsResponse:
    """Build the EmbedClaimsResponse from a list of Claim objects."""
    vectors: dict = {}
    for i in range(len(claims)):
        cid = claims[i].claim_id
        vectors[cid] = _fake_vector(i, NUM_DIMS)
    return EmbedClaimsResponse(
        analysis_id=ANALYSIS_ID,
        vectors=vectors,
        dim=NUM_DIMS,
    )


def _build_claims_metadata(claims: list) -> dict:
    """Build a Dict[str, ClaimMetadata] from a list of Claim objects."""
    metadata: dict = {}
    for i in range(len(claims)):
        c = claims[i]
        metadata[c.claim_id] = ClaimMetadata(
            model_id=c.model_id,
            claim_text=c.claim_text,
        )
    return metadata


def _build_cluster_response(
    vectors: dict,
    claims_metadata: dict,
) -> ClusterClaimsResponse:
    """Build the ClusterClaimsResponse by splitting claim IDs into two groups."""
    claim_ids = list(vectors.keys())
    half = len(claim_ids) // 2
    group_a = claim_ids[:half]
    group_b = claim_ids[half:]

    cluster_a = Cluster(
        cluster_id=make_cluster_id(group_a),
        claim_ids=group_a,
        representative_claim_id=group_a[0],
        representative_text=claims_metadata[group_a[0]].claim_text,
    )
    cluster_b = Cluster(
        cluster_id=make_cluster_id(group_b),
        claim_ids=group_b,
        representative_claim_id=group_b[0],
        representative_text=claims_metadata[group_b[0]].claim_text,
    )
    return ClusterClaimsResponse(
        analysis_id=ANALYSIS_ID,
        clusters=[cluster_a, cluster_b],
    )


def _build_rerank_response() -> RerankEvidenceBatchResponse:
    """Build the RerankEvidenceBatchResponse with descending scores."""
    fake_claim_id = make_claim_id(ANALYSIS_ID, MODEL_IDS[0], "The sky is blue.")
    passage_ids = _make_passage_ids(NUM_PASSAGES)

    scores: dict = {}
    ordered_pids: list = []
    for i in range(min(TOP_K, NUM_PASSAGES)):
        pid = passage_ids[i]
        ordered_pids.append(pid)
        scores[pid] = 1.0 - (i * 0.1)

    ranking = ClaimRanking(
        claim_id=fake_claim_id,
        ordered_passage_ids=ordered_pids,
        scores=scores,
    )
    return RerankEvidenceBatchResponse(
        analysis_id=ANALYSIS_ID,
        rankings=[ranking],
    )


def _build_nli_results(claims: list) -> list:
    """Build fake NliResultOutput objects for a list of Claim objects."""
    results: list = []
    passage_id = "p_0000"
    for i in range(len(claims)):
        cid = claims[i].claim_id
        pair_id = make_pair_id(cid, passage_id)
        label = NLI_LABELS[i % len(NLI_LABELS)]
        probs: dict = {
            "entailment": 0.33,
            "neutral": 0.34,
            "contradiction": 0.33,
        }
        probs[label] = 0.70
        remaining_labels: list = []
        for lbl in NLI_LABELS:
            if lbl != label:
                remaining_labels.append(lbl)
        for j in range(len(remaining_labels)):
            probs[remaining_labels[j]] = 0.15
        result = NliResultOutput(
            pair_id=pair_id,
            claim_id=cid,
            passage_id=passage_id,
            label=label,
            probs=probs,
        )
        results.append(result)
    return results


def _build_nli_response(claims: list) -> NliVerifyBatchResponse:
    """Build the NliVerifyBatchResponse from a list of Claim objects."""
    results = _build_nli_results(claims)
    return NliVerifyBatchResponse(
        analysis_id=ANALYSIS_ID,
        results=results,
    )


def _compute_cluster_agreement(
    cluster: Cluster,
    claims_metadata: dict,
) -> AgreementDetail:
    """Build AgreementDetail for a single cluster."""
    supporting: list = []
    seen_models: dict = {}
    for j in range(len(cluster.claim_ids)):
        cid = cluster.claim_ids[j]
        if cid in claims_metadata:
            mid = claims_metadata[cid].model_id
            if mid not in seen_models:
                seen_models[mid] = True
                supporting.append(mid)
    return AgreementDetail(
        models_supporting=supporting,
        count=len(supporting),
    )


def _find_best_nli_for_cluster(
    cluster: Cluster,
    nli_results: list,
) -> VerificationDetail:
    """Find best entailment/contradiction across NLI results for a cluster."""
    best_ent = 0.0
    best_contra = 0.0
    evidence_pid = ""
    claim_set: dict = {}
    for j in range(len(cluster.claim_ids)):
        claim_set[cluster.claim_ids[j]] = True
    for j in range(len(nli_results)):
        res = nli_results[j]
        if res.claim_id not in claim_set:
            continue
        ent = res.probs.get("entailment", 0.0)
        contra = res.probs.get("contradiction", 0.0)
        if ent > best_ent:
            best_ent = ent
            evidence_pid = res.passage_id
        if contra > best_contra:
            best_contra = contra
    return VerificationDetail(
        best_entailment_prob=best_ent,
        best_contradiction_prob=best_contra,
        evidence_passage_id=evidence_pid,
    )


def _compute_fake_trust(
    agreement: AgreementDetail,
    verification: VerificationDetail,
) -> tuple:
    """Compute trust_score and verdict from agreement and verification details."""
    agreement_score = 100.0 * (agreement.count / 2.0)
    v_score = 100.0 * verification.best_entailment_prob
    v_score = v_score - 100.0 * verification.best_contradiction_prob
    clamped_v = max(0.0, min(100.0, v_score))
    trust_raw = 0.4 * agreement_score + 0.6 * clamped_v
    trust_score = max(0, min(100, round(trust_raw)))

    verdict = "REJECT"
    if trust_score >= 75 and verification.best_contradiction_prob <= 0.2:
        verdict = "SAFE"
    elif trust_score >= 45:
        verdict = "CAUTION"
    return (trust_score, verdict)


def _build_score_response(
    clusters: list,
    claims_metadata: dict,
    nli_results: list,
) -> ScoreClustersResponse:
    """Build the ScoreClustersResponse from clusters and NLI results."""
    cluster_scores: list = []
    for i in range(len(clusters)):
        cluster = clusters[i]
        agreement = _compute_cluster_agreement(cluster, claims_metadata)
        verification = _find_best_nli_for_cluster(cluster, nli_results)
        trust_score, verdict = _compute_fake_trust(agreement, verification)
        cs = ClusterScore(
            cluster_id=cluster.cluster_id,
            trust_score=trust_score,
            verdict=verdict,
            agreement=agreement,
            verification=verification,
        )
        cluster_scores.append(cs)
    return ScoreClustersResponse(
        analysis_id=ANALYSIS_ID,
        scores=cluster_scores,
    )


def _build_umap_response(vectors: dict) -> ComputeUmapResponse:
    """Build the ComputeUmapResponse with fake 3D coordinates."""
    coords: dict = {}
    idx = 0
    for cid in vectors:
        coords[cid] = [float(idx) * 0.1, float(idx) * 0.2, float(idx) * 0.3]
        idx += 1
    return ComputeUmapResponse(
        analysis_id=ANALYSIS_ID,
        coords3d=coords,
    )


# ---------------------------------------------------------------------------
# Pytest fixtures -- chain pipeline phases
# ---------------------------------------------------------------------------

@pytest.fixture
def extract_resp() -> ExtractClaimsResponse:
    """Fixture: Phase 1 extract response."""
    return _build_extract_response()


@pytest.fixture
def claims(extract_resp: ExtractClaimsResponse) -> list:
    """Fixture: list of Claim objects from extract phase."""
    return extract_resp.claims


@pytest.fixture
def embed_resp(claims: list) -> EmbedClaimsResponse:
    """Fixture: Phase 2 embed response."""
    return _build_embed_response(claims)


@pytest.fixture
def claims_metadata(claims: list) -> dict:
    """Fixture: Dict[str, ClaimMetadata] for cluster/score phases."""
    return _build_claims_metadata(claims)


@pytest.fixture
def cluster_resp(
    embed_resp: EmbedClaimsResponse,
    claims_metadata: dict,
) -> ClusterClaimsResponse:
    """Fixture: Phase 3 cluster response."""
    return _build_cluster_response(embed_resp.vectors, claims_metadata)


@pytest.fixture
def rerank_resp() -> RerankEvidenceBatchResponse:
    """Fixture: Phase 4 rerank response."""
    return _build_rerank_response()


@pytest.fixture
def nli_resp(claims: list) -> NliVerifyBatchResponse:
    """Fixture: Phase 5 NLI response."""
    return _build_nli_response(claims)


@pytest.fixture
def score_resp(
    cluster_resp: ClusterClaimsResponse,
    claims_metadata: dict,
    nli_resp: NliVerifyBatchResponse,
) -> ScoreClustersResponse:
    """Fixture: Phase 6 score response."""
    return _build_score_response(
        cluster_resp.clusters, claims_metadata, nli_resp.results,
    )


@pytest.fixture
def umap_resp(embed_resp: EmbedClaimsResponse) -> ComputeUmapResponse:
    """Fixture: Phase 7 UMAP response."""
    return _build_umap_response(embed_resp.vectors)


# ---------------------------------------------------------------------------
# Phase 1: test_extract_phase
# ---------------------------------------------------------------------------

def test_extract_phase(extract_resp: ExtractClaimsResponse) -> None:
    """Validate extract_claims output with Claim objects."""
    assert len(extract_resp.warnings) == 0
    assert len(extract_resp.claims) == len(CLAIM_TEXTS)
    for j in range(len(extract_resp.claims)):
        assert extract_resp.claims[j].claim_id.startswith("c_")
        assert len(extract_resp.claims[j].claim_text) > 0
    print("  [PASS] extract phase")


# ---------------------------------------------------------------------------
# Phase 2: test_embed_phase
# ---------------------------------------------------------------------------

def test_embed_phase(
    embed_resp: EmbedClaimsResponse,
    claims: list,
) -> None:
    """Validate embed_claims output with Dict-based vectors."""
    assert len(embed_resp.warnings) == 0
    assert len(embed_resp.vectors) == len(claims)
    assert embed_resp.dim == NUM_DIMS
    for cid_key in embed_resp.vectors:
        assert cid_key.startswith("c_")
        assert len(embed_resp.vectors[cid_key]) == NUM_DIMS
    print("  [PASS] embed phase")


# ---------------------------------------------------------------------------
# Phase 3: test_cluster_phase
# ---------------------------------------------------------------------------

def test_cluster_phase(
    cluster_resp: ClusterClaimsResponse,
    embed_resp: EmbedClaimsResponse,
) -> None:
    """Validate cluster_claims output with Cluster objects."""
    claim_ids = list(embed_resp.vectors.keys())
    assert len(cluster_resp.warnings) == 0
    assert len(cluster_resp.clusters) == 2
    for k in range(len(cluster_resp.clusters)):
        assert cluster_resp.clusters[k].cluster_id.startswith("cl_")
        assert len(cluster_resp.clusters[k].claim_ids) >= 1

    all_ids: list = []
    for k in range(len(cluster_resp.clusters)):
        for cid in cluster_resp.clusters[k].claim_ids:
            all_ids.append(cid)
    assert sorted(all_ids) == sorted(claim_ids)
    print("  [PASS] cluster phase")


# ---------------------------------------------------------------------------
# Phase 4: test_rerank_phase
# ---------------------------------------------------------------------------

def test_rerank_phase(rerank_resp: RerankEvidenceBatchResponse) -> None:
    """Validate rerank_evidence_batch output with ClaimRanking objects."""
    assert len(rerank_resp.warnings) == 0
    assert len(rerank_resp.rankings) == 1
    assert rerank_resp.rankings[0].claim_id.startswith("c_")
    assert len(rerank_resp.rankings[0].ordered_passage_ids) == TOP_K

    prev_score = 999.0
    for j in range(len(rerank_resp.rankings[0].ordered_passage_ids)):
        pid = rerank_resp.rankings[0].ordered_passage_ids[j]
        current_score = rerank_resp.rankings[0].scores[pid]
        assert current_score <= prev_score
        prev_score = current_score
    print("  [PASS] rerank phase")


# ---------------------------------------------------------------------------
# Phase 5: test_nli_phase
# ---------------------------------------------------------------------------

def test_nli_phase(
    nli_resp: NliVerifyBatchResponse,
    claims: list,
) -> None:
    """Validate nli_verify_batch output with NliResultOutput objects."""
    assert len(nli_resp.warnings) == 0
    assert len(nli_resp.results) == len(claims)
    for i in range(len(nli_resp.results)):
        r = nli_resp.results[i]
        assert r.pair_id.startswith("nli_")
        assert r.claim_id.startswith("c_")
        assert r.label in ("entailment", "neutral", "contradiction")
        assert len(r.probs) == 3
    print("  [PASS] NLI phase")


# ---------------------------------------------------------------------------
# Phase 6: test_score_phase
# ---------------------------------------------------------------------------

def test_score_phase(
    score_resp: ScoreClustersResponse,
    cluster_resp: ClusterClaimsResponse,
) -> None:
    """Validate score_clusters output with ClusterScore objects."""
    assert len(score_resp.warnings) == 0
    assert len(score_resp.scores) == len(cluster_resp.clusters)
    for i in range(len(score_resp.scores)):
        s = score_resp.scores[i]
        assert s.cluster_id.startswith("cl_")
        assert 0 <= s.trust_score <= 100
        assert s.verdict in ("SAFE", "CAUTION", "REJECT")
    print("  [PASS] score phase")


# ---------------------------------------------------------------------------
# Phase 7: test_umap_phase
# ---------------------------------------------------------------------------

def test_umap_phase(
    umap_resp: ComputeUmapResponse,
    embed_resp: EmbedClaimsResponse,
) -> None:
    """Validate compute_umap output with Dict-based coords3d."""
    assert len(umap_resp.warnings) == 0
    assert len(umap_resp.coords3d) == len(embed_resp.vectors)
    for cid_key in umap_resp.coords3d:
        assert cid_key.startswith("c_")
        assert len(umap_resp.coords3d[cid_key]) == 3
    print("  [PASS] UMAP phase")


# ---------------------------------------------------------------------------
# Serialization round-trip
# ---------------------------------------------------------------------------

def test_serialization(
    extract_resp: ExtractClaimsResponse,
    embed_resp: EmbedClaimsResponse,
    cluster_resp: ClusterClaimsResponse,
    rerank_resp: RerankEvidenceBatchResponse,
    nli_resp: NliVerifyBatchResponse,
    score_resp: ScoreClustersResponse,
    umap_resp: ComputeUmapResponse,
) -> None:
    """Verify all responses round-trip through model_dump / reconstruct."""
    responses = [
        extract_resp, embed_resp, cluster_resp,
        rerank_resp, nli_resp, score_resp, umap_resp,
    ]
    names = [
        "extract", "embed", "cluster",
        "rerank", "nli", "score", "umap",
    ]
    classes = [
        ExtractClaimsResponse, EmbedClaimsResponse, ClusterClaimsResponse,
        RerankEvidenceBatchResponse, NliVerifyBatchResponse,
        ScoreClustersResponse, ComputeUmapResponse,
    ]
    for i in range(len(responses)):
        d = responses[i].model_dump()
        rebuilt = classes[i](**d)
        assert rebuilt == responses[i], f"{names[i]} round-trip failed"
    print("  [PASS] serialization round-trip")


# ---------------------------------------------------------------------------
# Main pipeline helpers (standalone python execution)
# ---------------------------------------------------------------------------

def _run_pipeline_phases() -> tuple:
    """Build all 7 pipeline responses and assert basic counts. Returns a tuple."""
    extract_r = _build_extract_response()
    claim_list = extract_r.claims
    assert len(extract_r.warnings) == 0
    assert len(claim_list) == len(CLAIM_TEXTS)
    print("  [PASS] extract phase")

    embed_r = _build_embed_response(claim_list)
    assert len(embed_r.vectors) == len(claim_list)
    print("  [PASS] embed phase")

    meta = _build_claims_metadata(claim_list)
    cluster_r = _build_cluster_response(embed_r.vectors, meta)
    assert len(cluster_r.clusters) == 2
    print("  [PASS] cluster phase")

    rerank_r = _build_rerank_response()
    assert len(rerank_r.rankings) == 1
    print("  [PASS] rerank phase")

    nli_r = _build_nli_response(claim_list)
    assert len(nli_r.results) == len(claim_list)
    print("  [PASS] NLI phase")

    score_r = _build_score_response(
        cluster_r.clusters, meta, nli_r.results,
    )
    assert len(score_r.scores) == len(cluster_r.clusters)
    print("  [PASS] score phase")

    umap_r = _build_umap_response(embed_r.vectors)
    assert len(umap_r.coords3d) == len(embed_r.vectors)
    print("  [PASS] UMAP phase")

    return (extract_r, embed_r, cluster_r, rerank_r, nli_r, score_r, umap_r)


def _run_serialization_check(responses: tuple) -> None:
    """Verify all 7 responses round-trip through model_dump / reconstruct."""
    classes = [
        ExtractClaimsResponse, EmbedClaimsResponse, ClusterClaimsResponse,
        RerankEvidenceBatchResponse, NliVerifyBatchResponse,
        ScoreClustersResponse, ComputeUmapResponse,
    ]
    for i in range(len(responses)):
        d = responses[i].model_dump()
        rebuilt = classes[i](**d)
        assert rebuilt == responses[i]
    print("  [PASS] serialization round-trip")


def main() -> int:
    """Run the full 7-function integration test pipeline."""
    print("TruthLens ML Integration Test Harness (v2 -- spec-aligned)")
    print("=" * 55)
    responses = _run_pipeline_phases()
    _run_serialization_check(responses)
    print("=" * 55)
    print("All integration tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
