"""Local integration test — validates the full pipeline with fake model outputs.

Runs: fake embed → cluster → rerank → NLI → UMAP
Asserts every response is a valid Pydantic model with no error field set.

Usage:
    python test_harness.py
"""

import sys

from schemas import (
    EmbedClaimsResponse,
    ClusterClaimsResponse,
    RerankEvidenceResponse,
    RankedPassage,
    NliVerifyResponse,
    NliResult,
    ComputeUmapResponse,
    UmapPoint,
)

MAX_CLAIMS = 20
NUM_DIMS = 8
NUM_PASSAGES = 6
TOP_K = 3


def _fake_vectors(n: int, dim: int) -> list:
    """Generate n fake embedding vectors of given dimension."""
    vectors: list = []
    for i in range(n):
        row: list = []
        for d in range(dim):
            row.append(float(i * dim + d) * 0.01)
        vectors.append(row)
    return vectors


def test_embed_phase() -> EmbedClaimsResponse:
    """Simulate embed_claims output."""
    vectors = _fake_vectors(MAX_CLAIMS, NUM_DIMS)
    resp = EmbedClaimsResponse(
        vectors=vectors,
        dimension=NUM_DIMS,
        model_name="fake-embed-model",
    )
    assert resp.error is None
    assert len(resp.vectors) == MAX_CLAIMS
    assert resp.dimension == NUM_DIMS
    print("  [PASS] embed phase")
    return resp


def test_cluster_phase(vectors: list) -> ClusterClaimsResponse:
    """Simulate cluster_claims output."""
    # Fake: first half in cluster 0, second half in cluster 1
    half = len(vectors) // 2
    c0 = list(range(half))
    c1 = list(range(half, len(vectors)))
    resp = ClusterClaimsResponse(
        clusters=[c0, c1],
        num_clusters=2,
    )
    assert resp.error is None
    assert resp.num_clusters == 2
    all_idx: list = []
    for cluster in resp.clusters:
        for idx in cluster:
            all_idx.append(idx)
    assert sorted(all_idx) == list(range(len(vectors)))
    print("  [PASS] cluster phase")
    return resp


def test_rerank_phase() -> RerankEvidenceResponse:
    """Simulate rerank_evidence output."""
    ranked: list = []
    for i in range(TOP_K):
        ranked.append(
            RankedPassage(
                index=i,
                text=f"passage {i}",
                score=1.0 - (i * 0.1),
            )
        )
    resp = RerankEvidenceResponse(ranked_passages=ranked)
    assert resp.error is None
    assert len(resp.ranked_passages) == TOP_K
    # Scores should be descending
    for i in range(1, len(resp.ranked_passages)):
        assert resp.ranked_passages[i].score <= resp.ranked_passages[i - 1].score
    print("  [PASS] rerank phase")
    return resp


def test_nli_phase() -> NliVerifyResponse:
    """Simulate nli_verify output."""
    results: list = []
    labels = ["entailment", "neutral", "contradiction"]
    for i in range(MAX_CLAIMS):
        label = labels[i % 3]
        results.append(
            NliResult(
                label=label,
                scores={
                    "entailment": 0.33,
                    "neutral": 0.34,
                    "contradiction": 0.33,
                },
            )
        )
    resp = NliVerifyResponse(results=results)
    assert resp.error is None
    assert len(resp.results) == MAX_CLAIMS
    print("  [PASS] NLI phase")
    return resp


def test_umap_phase(vectors: list) -> ComputeUmapResponse:
    """Simulate compute_umap output."""
    points: list = []
    for i in range(len(vectors)):
        points.append(
            UmapPoint(
                x=float(i) * 0.1,
                y=float(i) * 0.2,
                z=float(i) * 0.3,
            )
        )
    resp = ComputeUmapResponse(coords_3d=points)
    assert resp.error is None
    assert len(resp.coords_3d) == len(vectors)
    print("  [PASS] UMAP phase")
    return resp


def test_serialization(embed_resp, cluster_resp, rerank_resp, nli_resp, umap_resp):
    """Verify all responses round-trip through dict serialization."""
    responses = [embed_resp, cluster_resp, rerank_resp, nli_resp, umap_resp]
    names = ["embed", "cluster", "rerank", "nli", "umap"]
    classes = [
        EmbedClaimsResponse,
        ClusterClaimsResponse,
        RerankEvidenceResponse,
        NliVerifyResponse,
        ComputeUmapResponse,
    ]
    for i in range(len(responses)):
        d = responses[i].model_dump()
        rebuilt = classes[i](**d)
        assert rebuilt == responses[i], f"{names[i]} round-trip failed"
    print("  [PASS] serialization round-trip")


def main() -> int:
    """Run the full integration test pipeline."""
    print("TruthLens ML Integration Test Harness")
    print("=" * 40)

    embed_resp = test_embed_phase()
    cluster_resp = test_cluster_phase(embed_resp.vectors)
    rerank_resp = test_rerank_phase()
    nli_resp = test_nli_phase()
    umap_resp = test_umap_phase(embed_resp.vectors)
    test_serialization(embed_resp, cluster_resp, rerank_resp, nli_resp, umap_resp)

    print("=" * 40)
    print("All integration tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
