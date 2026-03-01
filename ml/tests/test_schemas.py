"""Tests for schemas.py — validation, defaults, constraints, round-trip.

Covers all 32 Pydantic models. Each model is tested for:
  - Valid construction with required fields
  - Default values are correct
  - Required field validation (missing fields raise ValidationError)
  - Field constraints (min_length, ge, le, gt, etc.)
  - Request types inherit schema_version="1.0" and analysis_id
  - Response types inherit warnings: List[str] = []
"""

import pytest
from pydantic import ValidationError

from schemas import (
    # Base classes
    BaseRequest,
    BaseResponse,
    # extract_claims
    ModelResponse,
    ClaimSpan,
    Claim,
    ExtractClaimsRequest,
    ExtractClaimsResponse,
    # embed_claims
    ClaimInput,
    EmbedClaimsRequest,
    EmbedClaimsResponse,
    # cluster_claims
    ClaimMetadata,
    Cluster,
    ClusterClaimsRequest,
    ClusterClaimsResponse,
    # rerank_evidence_batch
    PassageInput,
    RerankItem,
    ClaimRanking,
    RerankEvidenceBatchRequest,
    RerankEvidenceBatchResponse,
    # nli_verify_batch
    NliPairInput,
    NliResultOutput,
    NliVerifyBatchRequest,
    NliVerifyBatchResponse,
    # compute_umap
    ComputeUmapRequest,
    ComputeUmapResponse,
    # score_clusters
    ScoringWeights,
    VerdictThresholds,
    AgreementDetail,
    VerificationDetail,
    ClusterScore,
    ScoreClustersRequest,
    ScoreClustersResponse,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_model_response():
    """Build a valid ModelResponse dict."""
    return {"model_id": "m1", "response_text": "The sky is blue."}


def _make_claim():
    """Build a valid Claim dict."""
    return {
        "claim_id": "c1",
        "model_id": "m1",
        "claim_text": "Sky is blue",
    }


def _make_cluster():
    """Build a valid Cluster dict."""
    return {
        "cluster_id": "cl1",
        "claim_ids": ["c1"],
        "representative_claim_id": "c1",
        "representative_text": "Sky is blue",
    }


def _make_passage_input():
    """Build a valid PassageInput dict."""
    return {"passage_id": "p1", "text": "Evidence text"}


def _make_rerank_item():
    """Build a valid RerankItem dict."""
    return {
        "claim_id": "c1",
        "claim_text": "Sky is blue",
        "passages": [_make_passage_input()],
    }


def _make_nli_pair():
    """Build a valid NliPairInput dict."""
    return {
        "pair_id": "pr1",
        "claim_id": "c1",
        "passage_id": "p1",
        "claim_text": "Sky is blue",
        "passage_text": "Evidence text",
    }


def _make_nli_result():
    """Build a valid NliResultOutput dict."""
    return {
        "pair_id": "pr1",
        "claim_id": "c1",
        "passage_id": "p1",
    }


def _make_claim_metadata():
    """Build a valid ClaimMetadata dict."""
    return {"model_id": "m1", "claim_text": "Sky is blue"}


# ── BaseRequest ──────────────────────────────────────────────────────────────

class TestBaseRequest:
    def test_valid_construction(self):
        req = BaseRequest(analysis_id="a1")
        assert req.analysis_id == "a1"
        assert req.schema_version == "1.0"

    def test_missing_analysis_id(self):
        with pytest.raises(ValidationError):
            BaseRequest()

    def test_empty_analysis_id_rejected(self):
        with pytest.raises(ValidationError):
            BaseRequest(analysis_id="")

    def test_custom_schema_version(self):
        req = BaseRequest(analysis_id="a1", schema_version="2.0")
        assert req.schema_version == "2.0"


# ── BaseResponse ─────────────────────────────────────────────────────────────

class TestBaseResponse:
    def test_valid_construction(self):
        resp = BaseResponse(analysis_id="a1")
        assert resp.analysis_id == "a1"
        assert resp.schema_version == "1.0"
        assert resp.warnings == []

    def test_missing_analysis_id(self):
        with pytest.raises(ValidationError):
            BaseResponse()

    def test_warnings_default_empty_list(self):
        resp = BaseResponse(analysis_id="a1")
        assert isinstance(resp.warnings, list)
        assert len(resp.warnings) == 0

    def test_warnings_populated(self):
        resp = BaseResponse(analysis_id="a1", warnings=["w1", "w2"])
        assert resp.warnings == ["w1", "w2"]


# ── ModelResponse ────────────────────────────────────────────────────────────

class TestModelResponse:
    def test_valid(self):
        mr = ModelResponse(**_make_model_response())
        assert mr.model_id == "m1"
        assert mr.response_text == "The sky is blue."

    def test_missing_model_id(self):
        with pytest.raises(ValidationError):
            ModelResponse(response_text="text")

    def test_empty_response_text(self):
        with pytest.raises(ValidationError):
            ModelResponse(model_id="m1", response_text="")


# ── ClaimSpan ────────────────────────────────────────────────────────────────

class TestClaimSpan:
    def test_valid(self):
        cs = ClaimSpan(start=0, end=10)
        assert cs.start == 0
        assert cs.end == 10

    def test_negative_start_rejected(self):
        with pytest.raises(ValidationError):
            ClaimSpan(start=-1, end=5)

    def test_negative_end_rejected(self):
        with pytest.raises(ValidationError):
            ClaimSpan(start=0, end=-1)


# ── Claim ────────────────────────────────────────────────────────────────────

class TestClaim:
    def test_valid_without_span(self):
        c = Claim(**_make_claim())
        assert c.claim_id == "c1"
        assert c.span is None

    def test_valid_with_span(self):
        data = _make_claim()
        data["span"] = {"start": 0, "end": 5}
        c = Claim(**data)
        assert c.span.start == 0
        assert c.span.end == 5

    def test_missing_claim_text(self):
        with pytest.raises(ValidationError):
            Claim(claim_id="c1", model_id="m1")


# ── ExtractClaimsRequest ────────────────────────────────────────────────────

class TestExtractClaimsRequest:
    def test_valid(self):
        req = ExtractClaimsRequest(
            analysis_id="a1",
            responses=[ModelResponse(**_make_model_response())],
        )
        assert req.schema_version == "1.0"
        assert req.analysis_id == "a1"
        assert len(req.responses) == 1

    def test_empty_responses_rejected(self):
        with pytest.raises(ValidationError):
            ExtractClaimsRequest(analysis_id="a1", responses=[])


# ── ExtractClaimsResponse ───────────────────────────────────────────────────

class TestExtractClaimsResponse:
    def test_defaults(self):
        resp = ExtractClaimsResponse(analysis_id="a1")
        assert resp.claims == []
        assert resp.warnings == []
        assert resp.schema_version == "1.0"

    def test_round_trip(self):
        resp = ExtractClaimsResponse(
            analysis_id="a1",
            claims=[Claim(**_make_claim())],
        )
        rebuilt = ExtractClaimsResponse(**resp.model_dump())
        assert rebuilt == resp


# ── ClaimInput ───────────────────────────────────────────────────────────────

class TestClaimInput:
    def test_valid(self):
        ci = ClaimInput(claim_id="c1", claim_text="Sky is blue")
        assert ci.claim_id == "c1"

    def test_empty_claim_id_rejected(self):
        with pytest.raises(ValidationError):
            ClaimInput(claim_id="", claim_text="text")


# ── EmbedClaimsRequest ──────────────────────────────────────────────────────

class TestEmbedClaimsRequest:
    def test_valid_with_defaults(self):
        req = EmbedClaimsRequest(
            analysis_id="a1",
            claims=[ClaimInput(claim_id="c1", claim_text="t")],
        )
        assert req.model_name == "BAAI/bge-large-en-v1.5"
        assert req.schema_version == "1.0"

    def test_empty_claims_rejected(self):
        with pytest.raises(ValidationError):
            EmbedClaimsRequest(analysis_id="a1", claims=[])

    def test_custom_model_name(self):
        req = EmbedClaimsRequest(
            analysis_id="a1",
            claims=[ClaimInput(claim_id="c1", claim_text="t")],
            model_name="custom/model",
        )
        assert req.model_name == "custom/model"


# ── EmbedClaimsResponse ─────────────────────────────────────────────────────

class TestEmbedClaimsResponse:
    def test_defaults(self):
        resp = EmbedClaimsResponse(analysis_id="a1")
        assert resp.vectors == {}
        assert resp.dim == 0
        assert resp.warnings == []

    def test_with_values(self):
        resp = EmbedClaimsResponse(
            analysis_id="a1",
            vectors={"c1": [0.1, 0.2]},
            dim=2,
        )
        assert resp.vectors["c1"] == [0.1, 0.2]
        assert resp.dim == 2


# ── ClaimMetadata ────────────────────────────────────────────────────────────

class TestClaimMetadata:
    def test_valid(self):
        cm = ClaimMetadata(**_make_claim_metadata())
        assert cm.model_id == "m1"

    def test_empty_model_id_rejected(self):
        with pytest.raises(ValidationError):
            ClaimMetadata(model_id="", claim_text="text")


# ── Cluster ──────────────────────────────────────────────────────────────────

class TestCluster:
    def test_valid(self):
        cl = Cluster(**_make_cluster())
        assert cl.cluster_id == "cl1"
        assert cl.claim_ids == ["c1"]

    def test_empty_claim_ids_rejected(self):
        with pytest.raises(ValidationError):
            Cluster(
                cluster_id="cl1",
                claim_ids=[],
                representative_claim_id="c1",
                representative_text="text",
            )

    def test_missing_representative_text(self):
        with pytest.raises(ValidationError):
            Cluster(
                cluster_id="cl1",
                claim_ids=["c1"],
                representative_claim_id="c1",
            )


# ── ClusterClaimsRequest ────────────────────────────────────────────────────

class TestClusterClaimsRequest:
    def test_valid_with_defaults(self):
        req = ClusterClaimsRequest(
            analysis_id="a1",
            vectors={"c1": [0.1, 0.2]},
            claims={"c1": ClaimMetadata(**_make_claim_metadata())},
        )
        assert req.sim_threshold == 0.85
        assert req.schema_version == "1.0"

    def test_sim_threshold_zero_rejected(self):
        with pytest.raises(ValidationError):
            ClusterClaimsRequest(
                analysis_id="a1",
                vectors={"c1": [0.1]},
                claims={"c1": ClaimMetadata(**_make_claim_metadata())},
                sim_threshold=0.0,
            )

    def test_sim_threshold_above_one_rejected(self):
        with pytest.raises(ValidationError):
            ClusterClaimsRequest(
                analysis_id="a1",
                vectors={"c1": [0.1]},
                claims={"c1": ClaimMetadata(**_make_claim_metadata())},
                sim_threshold=1.5,
            )


# ── ClusterClaimsResponse ───────────────────────────────────────────────────

class TestClusterClaimsResponse:
    def test_defaults(self):
        resp = ClusterClaimsResponse(analysis_id="a1")
        assert resp.clusters == []
        assert resp.warnings == []

    def test_round_trip(self):
        resp = ClusterClaimsResponse(
            analysis_id="a1",
            clusters=[Cluster(**_make_cluster())],
        )
        rebuilt = ClusterClaimsResponse(**resp.model_dump())
        assert rebuilt == resp


# ── PassageInput ─────────────────────────────────────────────────────────────

class TestPassageInput:
    def test_valid(self):
        pi = PassageInput(**_make_passage_input())
        assert pi.passage_id == "p1"

    def test_empty_text_rejected(self):
        with pytest.raises(ValidationError):
            PassageInput(passage_id="p1", text="")


# ── RerankItem ───────────────────────────────────────────────────────────────

class TestRerankItem:
    def test_valid(self):
        ri = RerankItem(**_make_rerank_item())
        assert ri.claim_id == "c1"
        assert len(ri.passages) == 1

    def test_empty_passages_rejected(self):
        with pytest.raises(ValidationError):
            RerankItem(claim_id="c1", claim_text="text", passages=[])


# ── ClaimRanking ─────────────────────────────────────────────────────────────

class TestClaimRanking:
    def test_valid_with_defaults(self):
        cr = ClaimRanking(claim_id="c1")
        assert cr.ordered_passage_ids == []
        assert cr.scores == {}

    def test_with_values(self):
        cr = ClaimRanking(
            claim_id="c1",
            ordered_passage_ids=["p1", "p2"],
            scores={"p1": 0.9, "p2": 0.5},
        )
        assert cr.ordered_passage_ids == ["p1", "p2"]
        assert cr.scores["p1"] == 0.9


# ── RerankEvidenceBatchRequest ──────────────────────────────────────────────

class TestRerankEvidenceBatchRequest:
    def test_valid_with_defaults(self):
        req = RerankEvidenceBatchRequest(
            analysis_id="a1",
            items=[RerankItem(**_make_rerank_item())],
        )
        assert req.reranker_model == "cross-encoder/ms-marco-MiniLM-L-6-v2"
        assert req.top_k == 10
        assert req.schema_version == "1.0"

    def test_empty_items_rejected(self):
        with pytest.raises(ValidationError):
            RerankEvidenceBatchRequest(analysis_id="a1", items=[])

    def test_top_k_too_small(self):
        with pytest.raises(ValidationError):
            RerankEvidenceBatchRequest(
                analysis_id="a1",
                items=[RerankItem(**_make_rerank_item())],
                top_k=0,
            )

    def test_top_k_too_large(self):
        with pytest.raises(ValidationError):
            RerankEvidenceBatchRequest(
                analysis_id="a1",
                items=[RerankItem(**_make_rerank_item())],
                top_k=101,
            )


# ── RerankEvidenceBatchResponse ─────────────────────────────────────────────

class TestRerankEvidenceBatchResponse:
    def test_defaults(self):
        resp = RerankEvidenceBatchResponse(analysis_id="a1")
        assert resp.rankings == []
        assert resp.warnings == []


# ── NliPairInput ─────────────────────────────────────────────────────────────

class TestNliPairInput:
    def test_valid(self):
        npi = NliPairInput(**_make_nli_pair())
        assert npi.pair_id == "pr1"
        assert npi.claim_text == "Sky is blue"

    def test_empty_passage_text_rejected(self):
        with pytest.raises(ValidationError):
            NliPairInput(
                pair_id="pr1",
                claim_id="c1",
                passage_id="p1",
                claim_text="text",
                passage_text="",
            )


# ── NliResultOutput ──────────────────────────────────────────────────────────

class TestNliResultOutput:
    def test_valid_with_defaults(self):
        nro = NliResultOutput(**_make_nli_result())
        assert nro.label == "neutral"
        assert nro.probs == {}

    def test_custom_label_and_probs(self):
        nro = NliResultOutput(
            pair_id="pr1",
            claim_id="c1",
            passage_id="p1",
            label="entailment",
            probs={"entailment": 0.9, "neutral": 0.05, "contradiction": 0.05},
        )
        assert nro.label == "entailment"
        assert nro.probs["entailment"] == 0.9


# ── NliVerifyBatchRequest ───────────────────────────────────────────────────

class TestNliVerifyBatchRequest:
    def test_valid_with_defaults(self):
        req = NliVerifyBatchRequest(
            analysis_id="a1",
            pairs=[NliPairInput(**_make_nli_pair())],
        )
        assert req.nli_model == "cross-encoder/nli-deberta-v3-large"
        assert req.batch_size == 16
        assert req.schema_version == "1.0"

    def test_empty_pairs_rejected(self):
        with pytest.raises(ValidationError):
            NliVerifyBatchRequest(analysis_id="a1", pairs=[])

    def test_batch_size_too_small(self):
        with pytest.raises(ValidationError):
            NliVerifyBatchRequest(
                analysis_id="a1",
                pairs=[NliPairInput(**_make_nli_pair())],
                batch_size=0,
            )

    def test_batch_size_too_large(self):
        with pytest.raises(ValidationError):
            NliVerifyBatchRequest(
                analysis_id="a1",
                pairs=[NliPairInput(**_make_nli_pair())],
                batch_size=257,
            )


# ── NliVerifyBatchResponse ──────────────────────────────────────────────────

class TestNliVerifyBatchResponse:
    def test_defaults(self):
        resp = NliVerifyBatchResponse(analysis_id="a1")
        assert resp.results == []
        assert resp.warnings == []

    def test_round_trip(self):
        resp = NliVerifyBatchResponse(
            analysis_id="a1",
            results=[NliResultOutput(**_make_nli_result())],
        )
        rebuilt = NliVerifyBatchResponse(**resp.model_dump())
        assert rebuilt == resp


# ── ComputeUmapRequest ──────────────────────────────────────────────────────

class TestComputeUmapRequest:
    def test_valid_with_defaults(self):
        req = ComputeUmapRequest(
            analysis_id="a1",
            vectors={"c1": [1.0, 2.0, 3.0]},
        )
        assert req.random_state == 42
        assert req.n_neighbors == 15
        assert req.min_dist == 0.1
        assert req.schema_version == "1.0"

    def test_n_neighbors_too_small(self):
        with pytest.raises(ValidationError):
            ComputeUmapRequest(
                analysis_id="a1",
                vectors={"c1": [1.0]},
                n_neighbors=1,
            )

    def test_n_neighbors_too_large(self):
        with pytest.raises(ValidationError):
            ComputeUmapRequest(
                analysis_id="a1",
                vectors={"c1": [1.0]},
                n_neighbors=201,
            )

    def test_min_dist_zero_rejected(self):
        with pytest.raises(ValidationError):
            ComputeUmapRequest(
                analysis_id="a1",
                vectors={"c1": [1.0]},
                min_dist=0.0,
            )

    def test_min_dist_above_one_rejected(self):
        with pytest.raises(ValidationError):
            ComputeUmapRequest(
                analysis_id="a1",
                vectors={"c1": [1.0]},
                min_dist=1.5,
            )


# ── ComputeUmapResponse ─────────────────────────────────────────────────────

class TestComputeUmapResponse:
    def test_defaults(self):
        resp = ComputeUmapResponse(analysis_id="a1")
        assert resp.coords3d == {}
        assert resp.warnings == []


# ── ScoringWeights ───────────────────────────────────────────────────────────

class TestScoringWeights:
    def test_defaults(self):
        sw = ScoringWeights()
        assert sw.agreement_weight == 0.4
        assert sw.verification_weight == 0.6

    def test_weight_above_one_rejected(self):
        with pytest.raises(ValidationError):
            ScoringWeights(agreement_weight=1.5)

    def test_weight_below_zero_rejected(self):
        with pytest.raises(ValidationError):
            ScoringWeights(verification_weight=-0.1)


# ── VerdictThresholds ────────────────────────────────────────────────────────

class TestVerdictThresholds:
    def test_defaults(self):
        vt = VerdictThresholds()
        assert vt.safe_min == 75
        assert vt.caution_min == 45

    def test_safe_min_above_100_rejected(self):
        with pytest.raises(ValidationError):
            VerdictThresholds(safe_min=101)

    def test_caution_min_below_zero_rejected(self):
        with pytest.raises(ValidationError):
            VerdictThresholds(caution_min=-1)


# ── AgreementDetail ──────────────────────────────────────────────────────────

class TestAgreementDetail:
    def test_defaults(self):
        ad = AgreementDetail()
        assert ad.models_supporting == []
        assert ad.count == 0

    def test_negative_count_rejected(self):
        with pytest.raises(ValidationError):
            AgreementDetail(count=-1)


# ── VerificationDetail ──────────────────────────────────────────────────────

class TestVerificationDetail:
    def test_defaults(self):
        vd = VerificationDetail()
        assert vd.best_entailment_prob == 0.0
        assert vd.best_contradiction_prob == 0.0
        assert vd.evidence_passage_id == ""

    def test_entailment_prob_above_one_rejected(self):
        with pytest.raises(ValidationError):
            VerificationDetail(best_entailment_prob=1.1)

    def test_contradiction_prob_below_zero_rejected(self):
        with pytest.raises(ValidationError):
            VerificationDetail(best_contradiction_prob=-0.1)


# ── ClusterScore ─────────────────────────────────────────────────────────────

class TestClusterScore:
    def test_valid_with_defaults(self):
        cs = ClusterScore(cluster_id="cl1")
        assert cs.trust_score == 0
        assert cs.verdict == "REJECT"
        assert cs.agreement.count == 0
        assert cs.verification.best_entailment_prob == 0.0

    def test_trust_score_above_100_rejected(self):
        with pytest.raises(ValidationError):
            ClusterScore(cluster_id="cl1", trust_score=101)

    def test_trust_score_below_zero_rejected(self):
        with pytest.raises(ValidationError):
            ClusterScore(cluster_id="cl1", trust_score=-1)

    def test_empty_cluster_id_rejected(self):
        with pytest.raises(ValidationError):
            ClusterScore(cluster_id="")


# ── ScoreClustersRequest ────────────────────────────────────────────────────

class TestScoreClustersRequest:
    def test_valid_with_defaults(self):
        req = ScoreClustersRequest(
            analysis_id="a1",
            clusters=[Cluster(**_make_cluster())],
            claims={"c1": ClaimMetadata(**_make_claim_metadata())},
        )
        assert req.schema_version == "1.0"
        assert req.nli_results == []
        assert req.weights.agreement_weight == 0.4
        assert req.verdict_thresholds.safe_min == 75

    def test_empty_clusters_rejected(self):
        with pytest.raises(ValidationError):
            ScoreClustersRequest(
                analysis_id="a1",
                clusters=[],
                claims={"c1": ClaimMetadata(**_make_claim_metadata())},
            )

    def test_inherits_analysis_id(self):
        req = ScoreClustersRequest(
            analysis_id="test-id",
            clusters=[Cluster(**_make_cluster())],
            claims={"c1": ClaimMetadata(**_make_claim_metadata())},
        )
        assert req.analysis_id == "test-id"


# ── ScoreClustersResponse ───────────────────────────────────────────────────

class TestScoreClustersResponse:
    def test_defaults(self):
        resp = ScoreClustersResponse(analysis_id="a1")
        assert resp.scores == []
        assert resp.warnings == []
        assert resp.schema_version == "1.0"

    def test_round_trip(self):
        resp = ScoreClustersResponse(
            analysis_id="a1",
            scores=[ClusterScore(cluster_id="cl1", trust_score=80, verdict="SAFE")],
        )
        rebuilt = ScoreClustersResponse(**resp.model_dump())
        assert rebuilt == resp
