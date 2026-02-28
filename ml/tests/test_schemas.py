"""Tests for schemas.py — validation, defaults, constraints, round-trip."""

import pytest
from schemas import (
    EmbedClaimsRequest,
    EmbedClaimsResponse,
    ClusterClaimsRequest,
    ClusterClaimsResponse,
    RerankEvidenceRequest,
    RerankEvidenceResponse,
    RankedPassage,
    NliVerifyRequest,
    NliVerifyResponse,
    NliPair,
    NliResult,
    ComputeUmapRequest,
    ComputeUmapResponse,
    UmapPoint,
)


# ── EmbedClaims ──────────────────────────────────────────────────────────────

class TestEmbedClaimsRequest:
    def test_valid_minimal(self):
        req = EmbedClaimsRequest(claim_texts=["hello"])
        assert req.claim_texts == ["hello"]
        assert req.batch_size == 32

    def test_custom_batch_size(self):
        req = EmbedClaimsRequest(claim_texts=["a", "b"], batch_size=64)
        assert req.batch_size == 64

    def test_empty_claim_texts_rejected(self):
        with pytest.raises(Exception):
            EmbedClaimsRequest(claim_texts=[])

    def test_batch_size_too_small(self):
        with pytest.raises(Exception):
            EmbedClaimsRequest(claim_texts=["a"], batch_size=0)

    def test_batch_size_too_large(self):
        with pytest.raises(Exception):
            EmbedClaimsRequest(claim_texts=["a"], batch_size=1000)


class TestEmbedClaimsResponse:
    def test_defaults(self):
        resp = EmbedClaimsResponse()
        assert resp.vectors == []
        assert resp.dimension == 0
        assert resp.model_name == ""
        assert resp.error is None

    def test_with_values(self):
        resp = EmbedClaimsResponse(
            vectors=[[1.0, 2.0]], dimension=2, model_name="test"
        )
        assert len(resp.vectors) == 1
        assert resp.dimension == 2

    def test_round_trip(self):
        resp = EmbedClaimsResponse(
            vectors=[[0.1, 0.2]], dimension=2, model_name="m"
        )
        d = resp.model_dump()
        rebuilt = EmbedClaimsResponse(**d)
        assert rebuilt == resp


# ── ClusterClaims ─────────────────────────────────────────────────────────────

class TestClusterClaimsRequest:
    def test_valid(self):
        req = ClusterClaimsRequest(vectors=[[1.0, 2.0]])
        assert req.threshold == 0.5

    def test_empty_vectors_rejected(self):
        with pytest.raises(Exception):
            ClusterClaimsRequest(vectors=[])

    def test_threshold_bounds(self):
        with pytest.raises(Exception):
            ClusterClaimsRequest(vectors=[[1.0]], threshold=0.0)
        with pytest.raises(Exception):
            ClusterClaimsRequest(vectors=[[1.0]], threshold=3.0)


class TestClusterClaimsResponse:
    def test_defaults(self):
        resp = ClusterClaimsResponse()
        assert resp.clusters == []
        assert resp.num_clusters == 0
        assert resp.error is None

    def test_round_trip(self):
        resp = ClusterClaimsResponse(clusters=[[0, 1], [2]], num_clusters=2)
        d = resp.model_dump()
        rebuilt = ClusterClaimsResponse(**d)
        assert rebuilt == resp


# ── RerankEvidence ────────────────────────────────────────────────────────────

class TestRankedPassage:
    def test_valid(self):
        rp = RankedPassage(index=0, text="hello", score=0.9)
        assert rp.index == 0

    def test_negative_index_rejected(self):
        with pytest.raises(Exception):
            RankedPassage(index=-1, text="x", score=0.0)


class TestRerankEvidenceRequest:
    def test_valid(self):
        req = RerankEvidenceRequest(claim="test", passages=["p1", "p2"])
        assert req.top_k == 5

    def test_empty_claim_rejected(self):
        with pytest.raises(Exception):
            RerankEvidenceRequest(claim="", passages=["p"])

    def test_empty_passages_rejected(self):
        with pytest.raises(Exception):
            RerankEvidenceRequest(claim="test", passages=[])


class TestRerankEvidenceResponse:
    def test_defaults(self):
        resp = RerankEvidenceResponse()
        assert resp.ranked_passages == []
        assert resp.error is None

    def test_round_trip(self):
        resp = RerankEvidenceResponse(
            ranked_passages=[
                RankedPassage(index=0, text="t", score=1.0)
            ]
        )
        d = resp.model_dump()
        rebuilt = RerankEvidenceResponse(**d)
        assert rebuilt == resp


# ── NliVerify ─────────────────────────────────────────────────────────────────

class TestNliPair:
    def test_valid(self):
        pair = NliPair(premise="The sky is blue.", hypothesis="It is daytime.")
        assert pair.premise == "The sky is blue."

    def test_empty_premise_rejected(self):
        with pytest.raises(Exception):
            NliPair(premise="", hypothesis="h")

    def test_empty_hypothesis_rejected(self):
        with pytest.raises(Exception):
            NliPair(premise="p", hypothesis="")


class TestNliResult:
    def test_defaults(self):
        r = NliResult()
        assert r.label == "neutral"
        assert r.scores == {}


class TestNliVerifyRequest:
    def test_valid(self):
        req = NliVerifyRequest(
            pairs=[NliPair(premise="p", hypothesis="h")]
        )
        assert req.batch_size == 16

    def test_empty_pairs_rejected(self):
        with pytest.raises(Exception):
            NliVerifyRequest(pairs=[])


class TestNliVerifyResponse:
    def test_defaults(self):
        resp = NliVerifyResponse()
        assert resp.results == []
        assert resp.error is None

    def test_round_trip(self):
        resp = NliVerifyResponse(
            results=[NliResult(label="entailment", scores={"entailment": 0.9})]
        )
        d = resp.model_dump()
        rebuilt = NliVerifyResponse(**d)
        assert rebuilt == resp


# ── ComputeUmap ───────────────────────────────────────────────────────────────

class TestUmapPoint:
    def test_defaults(self):
        p = UmapPoint()
        assert p.x == 0.0
        assert p.y == 0.0
        assert p.z == 0.0

    def test_values(self):
        p = UmapPoint(x=1.0, y=2.0, z=3.0)
        assert p.x == 1.0


class TestComputeUmapRequest:
    def test_valid(self):
        req = ComputeUmapRequest(vectors=[[1.0, 2.0, 3.0]])
        assert req.n_neighbors == 15
        assert req.min_dist == 0.1

    def test_empty_vectors_rejected(self):
        with pytest.raises(Exception):
            ComputeUmapRequest(vectors=[])

    def test_n_neighbors_bounds(self):
        with pytest.raises(Exception):
            ComputeUmapRequest(vectors=[[1.0]], n_neighbors=1)
        with pytest.raises(Exception):
            ComputeUmapRequest(vectors=[[1.0]], n_neighbors=300)


class TestComputeUmapResponse:
    def test_defaults(self):
        resp = ComputeUmapResponse()
        assert resp.coords_3d == []
        assert resp.error is None

    def test_round_trip(self):
        resp = ComputeUmapResponse(
            coords_3d=[UmapPoint(x=1.0, y=2.0, z=3.0)]
        )
        d = resp.model_dump()
        rebuilt = ComputeUmapResponse(**d)
        assert rebuilt == resp
