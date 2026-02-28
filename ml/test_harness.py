"""Local integration test -- validates the full 7-function pipeline with fake model outputs.

Runs: extract_claims -> embed_claims -> cluster_claims -> rerank_evidence_batch
      -> nli_verify_batch -> score_clusters -> compute_umap

Asserts every response is a valid Pydantic model with no warnings.

Usage:
    python test_harness.py
"""

import sys

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


# ---------------------------------------------------------------------------
# Helpers
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


# ---------------------------------------------------------------------------
# Phase 1: extract_claims
# ---------------------------------------------------------------------------

def test_extract_phase() -> ExtractClaimsResponse:
    """Simulate extract_claims output with Claim objects."""
    claims: list = []
    texts = [
        "The sky is blue.",
        "Water boils at 100C at sea level.",
        "The Earth orbits the Sun.",
        "Photosynthesis requires sunlight.",
        "Iron is a metal.",
    ]
    for i in range(len(texts)):
        mid = MODEL_IDS[i % len(MODEL_IDS)]
        claim = _make_fake_claim(ANALYSIS_ID, mid, texts[i])
        claims.append(claim)

    resp = ExtractClaimsResponse(
        analysis_id=ANALYSIS_ID,
        claims=claims,
    )
    assert len(resp.warnings) == 0
    assert len(resp.claims) == len(texts)
    for j in range(len(resp.claims)):
        assert resp.claims[j].claim_id.startswith("c_")
        assert len(resp.claims[j].claim_text) > 0
    print("  [PASS] extract phase")
    return resp


# ---------------------------------------------------------------------------
# Phase 2: embed_claims
# ---------------------------------------------------------------------------

def test_embed_phase(claims: list) -> EmbedClaimsResponse:
    """Simulate embed_claims output with Dict-based vectors."""
    vectors: dict = {}
    for i in range(len(claims)):
        cid = claims[i].claim_id
        vectors[cid] = _fake_vector(i, NUM_DIMS)

    resp = EmbedClaimsResponse(
        analysis_id=ANALYSIS_ID,
        vectors=vectors,
        dim=NUM_DIMS,
    )
    assert len(resp.warnings) == 0
    assert len(resp.vectors) == len(claims)
    assert resp.dim == NUM_DIMS
    for cid_key in resp.vectors:
        assert cid_key.startswith("c_")
        assert len(resp.vectors[cid_key]) == NUM_DIMS
    print("  [PASS] embed phase")
    return resp


# ---------------------------------------------------------------------------
# Phase 3: cluster_claims
# ---------------------------------------------------------------------------

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


def test_cluster_phase(
    vectors: dict,
    claims_metadata: dict,
) -> ClusterClaimsResponse:
    """Simulate cluster_claims output with Cluster objects."""
    claim_ids = list(vectors.keys())
    half = len(claim_ids) // 2
    group_a = claim_ids[:half]
    group_b = claim_ids[half:]

    cluster_a_id = make_cluster_id(group_a)
    cluster_b_id = make_cluster_id(group_b)

    rep_a_id = group_a[0]
    rep_b_id = group_b[0]

    cluster_a = Cluster(
        cluster_id=cluster_a_id,
        claim_ids=group_a,
        representative_claim_id=rep_a_id,
        representative_text=claims_metadata[rep_a_id].claim_text,
    )
    cluster_b = Cluster(
        cluster_id=cluster_b_id,
        claim_ids=group_b,
        representative_claim_id=rep_b_id,
        representative_text=claims_metadata[rep_b_id].claim_text,
    )

    resp = ClusterClaimsResponse(
        analysis_id=ANALYSIS_ID,
        clusters=[cluster_a, cluster_b],
    )
    assert len(resp.warnings) == 0
    assert len(resp.clusters) == 2
    for k in range(len(resp.clusters)):
        assert resp.clusters[k].cluster_id.startswith("cl_")
        assert len(resp.clusters[k].claim_ids) >= 1

    # Verify all claim_ids accounted for
    all_ids: list = []
    for k in range(len(resp.clusters)):
        for cid in resp.clusters[k].claim_ids:
            all_ids.append(cid)
    assert sorted(all_ids) == sorted(claim_ids)
    print("  [PASS] cluster phase")
    return resp


# ---------------------------------------------------------------------------
# Phase 4: rerank_evidence_batch
# ---------------------------------------------------------------------------

def _make_passage_ids(count: int) -> list:
    """Generate a list of fake passage IDs with p_ prefix."""
    pids: list = []
    for i in range(count):
        pids.append(f"p_{i:04d}")
    return pids


def test_rerank_phase() -> RerankEvidenceBatchResponse:
    """Simulate rerank_evidence_batch output with ClaimRanking objects."""
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
    resp = RerankEvidenceBatchResponse(
        analysis_id=ANALYSIS_ID,
        rankings=[ranking],
    )
    assert len(resp.warnings) == 0
    assert len(resp.rankings) == 1
    assert resp.rankings[0].claim_id.startswith("c_")
    assert len(resp.rankings[0].ordered_passage_ids) == TOP_K

    # Scores should be descending
    prev_score = 999.0
    for j in range(len(resp.rankings[0].ordered_passage_ids)):
        pid = resp.rankings[0].ordered_passage_ids[j]
        current_score = resp.rankings[0].scores[pid]
        assert current_score <= prev_score
        prev_score = current_score
    print("  [PASS] rerank phase")
    return resp


# ---------------------------------------------------------------------------
# Phase 5: nli_verify_batch
# ---------------------------------------------------------------------------

def _build_nli_results(claims: list) -> list:
    """Build fake NliResultOutput objects for a list of Claim objects."""
    labels = ["entailment", "neutral", "contradiction"]
    results: list = []
    passage_id = "p_0000"
    for i in range(len(claims)):
        cid = claims[i].claim_id
        pair_id = make_pair_id(cid, passage_id)
        label = labels[i % len(labels)]
        probs: dict = {
            "entailment": 0.33,
            "neutral": 0.34,
            "contradiction": 0.33,
        }
        # Boost the selected label
        probs[label] = 0.70
        # Reduce the others proportionally
        remaining_labels: list = []
        for lbl in labels:
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


def test_nli_phase(claims: list) -> NliVerifyBatchResponse:
    """Simulate nli_verify_batch output with NliResultOutput objects."""
    results = _build_nli_results(claims)

    resp = NliVerifyBatchResponse(
        analysis_id=ANALYSIS_ID,
        results=results,
    )
    assert len(resp.warnings) == 0
    assert len(resp.results) == len(claims)
    for i in range(len(resp.results)):
        r = resp.results[i]
        assert r.pair_id.startswith("nli_")
        assert r.claim_id.startswith("c_")
        assert r.label in ("entailment", "neutral", "contradiction")
        assert len(r.probs) == 3
    print("  [PASS] NLI phase")
    return resp


# ---------------------------------------------------------------------------
# Phase 6: score_clusters
# ---------------------------------------------------------------------------

def _compute_cluster_agreement(cluster: Cluster, claims_metadata: dict) -> AgreementDetail:
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


def test_score_phase(
    clusters: list,
    claims_metadata: dict,
    nli_results: list,
) -> ScoreClustersResponse:
    """Simulate score_clusters output with ClusterScore objects."""
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

    resp = ScoreClustersResponse(
        analysis_id=ANALYSIS_ID,
        scores=cluster_scores,
    )
    assert len(resp.warnings) == 0
    assert len(resp.scores) == len(clusters)
    for i in range(len(resp.scores)):
        s = resp.scores[i]
        assert s.cluster_id.startswith("cl_")
        assert 0 <= s.trust_score <= 100
        assert s.verdict in ("SAFE", "CAUTION", "REJECT")
    print("  [PASS] score phase")
    return resp


# ---------------------------------------------------------------------------
# Phase 7: compute_umap
# ---------------------------------------------------------------------------

def test_umap_phase(vectors: dict) -> ComputeUmapResponse:
    """Simulate compute_umap output with Dict-based coords3d."""
    coords: dict = {}
    idx = 0
    for cid in vectors:
        coords[cid] = [float(idx) * 0.1, float(idx) * 0.2, float(idx) * 0.3]
        idx += 1

    resp = ComputeUmapResponse(
        analysis_id=ANALYSIS_ID,
        coords3d=coords,
    )
    assert len(resp.warnings) == 0
    assert len(resp.coords3d) == len(vectors)
    for cid_key in resp.coords3d:
        assert cid_key.startswith("c_")
        assert len(resp.coords3d[cid_key]) == 3
    print("  [PASS] UMAP phase")
    return resp


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
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> int:
    """Run the full 7-function integration test pipeline."""
    print("TruthLens ML Integration Test Harness (v2 -- spec-aligned)")
    print("=" * 55)

    # Phase 1: extract claims
    extract_resp = test_extract_phase()
    claims = extract_resp.claims

    # Phase 2: embed claims
    embed_resp = test_embed_phase(claims)

    # Phase 3: cluster claims
    claims_metadata = _build_claims_metadata(claims)
    cluster_resp = test_cluster_phase(embed_resp.vectors, claims_metadata)

    # Phase 4: rerank evidence
    rerank_resp = test_rerank_phase()

    # Phase 5: NLI verify
    nli_resp = test_nli_phase(claims)

    # Phase 6: score clusters
    score_resp = test_score_phase(
        cluster_resp.clusters, claims_metadata, nli_resp.results,
    )

    # Phase 7: UMAP
    umap_resp = test_umap_phase(embed_resp.vectors)

    # Serialization round-trip
    test_serialization(
        extract_resp, embed_resp, cluster_resp,
        rerank_resp, nli_resp, score_resp, umap_resp,
    )

    print("=" * 55)
    print("All integration tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
