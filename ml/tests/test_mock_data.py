"""Tests for mock_data module -- schema validation and determinism."""

import pytest

from mock_data import (
    build_extract_claims_request,
    build_extract_claims_response,
    build_embed_claims_request,
    build_embed_claims_response,
    build_cluster_claims_request,
    build_cluster_claims_response,
    build_rerank_request,
    build_rerank_response,
    build_nli_request,
    build_nli_response,
    build_umap_request,
    build_umap_response,
    build_score_request,
    build_score_response,
    build_full_pipeline_data,
    ANALYSIS_ID,
)

from schemas import (
    ExtractClaimsRequest,
    ExtractClaimsResponse,
    EmbedClaimsRequest,
    EmbedClaimsResponse,
    ClusterClaimsRequest,
    ClusterClaimsResponse,
    RerankEvidenceBatchRequest,
    RerankEvidenceBatchResponse,
    NliVerifyBatchRequest,
    NliVerifyBatchResponse,
    ComputeUmapRequest,
    ComputeUmapResponse,
    ScoreClustersRequest,
    ScoreClustersResponse,
)


class TestExtractClaimsMock:
    def test_request_validates(self):
        data = build_extract_claims_request()
        req = ExtractClaimsRequest(**data)
        assert req.analysis_id == ANALYSIS_ID

    def test_request_has_five_responses(self):
        data = build_extract_claims_request()
        assert len(data["responses"]) == 5

    def test_response_validates(self):
        data = build_extract_claims_response()
        resp = ExtractClaimsResponse(**data)
        assert resp.analysis_id == ANALYSIS_ID

    def test_response_has_ten_claims(self):
        data = build_extract_claims_response()
        assert len(data["claims"]) == 10


class TestEmbedClaimsMock:
    def test_request_validates(self):
        data = build_embed_claims_request()
        req = EmbedClaimsRequest(**data)
        assert len(req.claims) == 10

    def test_response_validates(self):
        data = build_embed_claims_response()
        resp = EmbedClaimsResponse(**data)
        assert resp.dim == 8
        assert len(resp.vectors) == 10


class TestClusterClaimsMock:
    def test_request_validates(self):
        data = build_cluster_claims_request()
        req = ClusterClaimsRequest(**data)
        assert len(req.vectors) == 10

    def test_response_validates(self):
        data = build_cluster_claims_response()
        resp = ClusterClaimsResponse(**data)
        assert len(resp.clusters) == 2

    def test_cluster_sizes(self):
        data = build_cluster_claims_response()
        clusters = data["clusters"]
        assert len(clusters[0]["claim_ids"]) == 8
        assert len(clusters[1]["claim_ids"]) == 2


class TestRerankMock:
    def test_request_validates(self):
        data = build_rerank_request()
        req = RerankEvidenceBatchRequest(**data)
        assert len(req.items) == 10

    def test_response_validates(self):
        data = build_rerank_response()
        resp = RerankEvidenceBatchResponse(**data)
        assert len(resp.rankings) == 10


class TestNliMock:
    def test_request_validates(self):
        data = build_nli_request()
        req = NliVerifyBatchRequest(**data)
        assert len(req.pairs) == 10

    def test_response_validates(self):
        data = build_nli_response()
        resp = NliVerifyBatchResponse(**data)
        assert len(resp.results) == 10

    def test_correct_model_gets_entailment(self):
        data = build_nli_response()
        first_result = data["results"][0]
        assert first_result["label"] == "entailment"

    def test_flat_bot_gets_contradiction(self):
        data = build_nli_response()
        flat_results = [r for r in data["results"] if r["label"] == "contradiction"]
        assert len(flat_results) == 2


class TestUmapMock:
    def test_request_validates(self):
        data = build_umap_request()
        req = ComputeUmapRequest(**data)
        assert len(req.vectors) == 10

    def test_response_validates(self):
        data = build_umap_response()
        resp = ComputeUmapResponse(**data)
        assert len(resp.coords3d) == 10


class TestScoreMock:
    def test_request_validates(self):
        data = build_score_request()
        req = ScoreClustersRequest(**data)
        assert len(req.clusters) == 2

    def test_response_validates(self):
        data = build_score_response()
        resp = ScoreClustersResponse(**data)
        assert len(resp.scores) == 2

    def test_correct_cluster_is_safe(self):
        data = build_score_response()
        assert data["scores"][0]["verdict"] == "SAFE"

    def test_flat_cluster_is_reject(self):
        data = build_score_response()
        assert data["scores"][1]["verdict"] == "REJECT"


class TestDeterminism:
    def test_full_pipeline_deterministic(self):
        data1 = build_full_pipeline_data()
        data2 = build_full_pipeline_data()
        assert data1 == data2

    def test_full_pipeline_has_14_keys(self):
        data = build_full_pipeline_data()
        assert len(data) == 14


class TestIdPrefixes:
    def test_claim_ids_start_with_c(self):
        data = build_extract_claims_response()
        for i in range(len(data["claims"])):
            assert data["claims"][i]["claim_id"].startswith("c_")

    def test_cluster_ids_start_with_cl(self):
        data = build_cluster_claims_response()
        for i in range(len(data["clusters"])):
            assert data["clusters"][i]["cluster_id"].startswith("cl_")

    def test_pair_ids_start_with_nli(self):
        data = build_nli_request()
        for i in range(len(data["pairs"])):
            assert data["pairs"][i]["pair_id"].startswith("nli_")
