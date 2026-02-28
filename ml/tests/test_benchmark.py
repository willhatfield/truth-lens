"""Tests for benchmark.py synthetic payload generators.

Validates that every generator produces a dict that passes Pydantic
validation against the corresponding schema from schemas.py.
Does NOT call Modal or time anything.
"""

import sys
import os

# Ensure project root is on sys.path for schema imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from benchmark import (
    make_extract_claims_payload,
    make_embed_claims_payload,
    make_cluster_claims_payload,
    make_rerank_evidence_batch_payload,
    make_nli_verify_batch_payload,
    make_compute_umap_payload,
    make_score_clusters_payload,
    PAYLOAD_GENERATORS,
    FUNCTION_NAMES,
    EMBEDDING_DIM,
)
from schemas import (
    ExtractClaimsRequest,
    EmbedClaimsRequest,
    ClusterClaimsRequest,
    RerankEvidenceBatchRequest,
    NliVerifyBatchRequest,
    ComputeUmapRequest,
    ScoreClustersRequest,
)


# -- extract_claims payload -------------------------------------------------

class TestExtractClaimsPayload:
    """Validate make_extract_claims_payload output."""

    def test_returns_dict(self) -> None:
        payload = make_extract_claims_payload()
        assert isinstance(payload, dict)

    def test_has_analysis_id(self) -> None:
        payload = make_extract_claims_payload()
        assert "analysis_id" in payload
        assert len(payload["analysis_id"]) > 0

    def test_has_responses_list(self) -> None:
        payload = make_extract_claims_payload()
        assert "responses" in payload
        assert len(payload["responses"]) >= 1

    def test_response_has_required_keys(self) -> None:
        payload = make_extract_claims_payload()
        resp = payload["responses"][0]
        assert "model_id" in resp
        assert "response_text" in resp

    def test_pydantic_validation(self) -> None:
        payload = make_extract_claims_payload()
        req = ExtractClaimsRequest(**payload)
        assert req.analysis_id == payload["analysis_id"]
        assert len(req.responses) == len(payload["responses"])


# -- embed_claims payload ---------------------------------------------------

class TestEmbedClaimsPayload:
    """Validate make_embed_claims_payload output."""

    def test_returns_dict(self) -> None:
        payload = make_embed_claims_payload()
        assert isinstance(payload, dict)

    def test_has_claims_list(self) -> None:
        payload = make_embed_claims_payload()
        assert "claims" in payload
        assert len(payload["claims"]) >= 1

    def test_claim_has_required_keys(self) -> None:
        payload = make_embed_claims_payload()
        claim = payload["claims"][0]
        assert "claim_id" in claim
        assert "claim_text" in claim

    def test_pydantic_validation(self) -> None:
        payload = make_embed_claims_payload()
        req = EmbedClaimsRequest(**payload)
        assert req.analysis_id == payload["analysis_id"]
        assert len(req.claims) == len(payload["claims"])


# -- cluster_claims payload -------------------------------------------------

class TestClusterClaimsPayload:
    """Validate make_cluster_claims_payload output."""

    def test_returns_dict(self) -> None:
        payload = make_cluster_claims_payload()
        assert isinstance(payload, dict)

    def test_has_vectors_dict(self) -> None:
        payload = make_cluster_claims_payload()
        assert "vectors" in payload
        assert len(payload["vectors"]) >= 1

    def test_has_claims_dict(self) -> None:
        payload = make_cluster_claims_payload()
        assert "claims" in payload
        assert len(payload["claims"]) >= 1

    def test_vector_dimension_matches(self) -> None:
        payload = make_cluster_claims_payload()
        for vec in payload["vectors"].values():
            assert len(vec) == EMBEDDING_DIM

    def test_pydantic_validation(self) -> None:
        payload = make_cluster_claims_payload()
        req = ClusterClaimsRequest(**payload)
        assert req.analysis_id == payload["analysis_id"]
        assert len(req.vectors) == len(payload["vectors"])


# -- rerank_evidence_batch payload ------------------------------------------

class TestRerankEvidenceBatchPayload:
    """Validate make_rerank_evidence_batch_payload output."""

    def test_returns_dict(self) -> None:
        payload = make_rerank_evidence_batch_payload()
        assert isinstance(payload, dict)

    def test_has_items_list(self) -> None:
        payload = make_rerank_evidence_batch_payload()
        assert "items" in payload
        assert len(payload["items"]) >= 1

    def test_item_has_required_keys(self) -> None:
        payload = make_rerank_evidence_batch_payload()
        item = payload["items"][0]
        assert "claim_id" in item
        assert "claim_text" in item
        assert "passages" in item
        assert len(item["passages"]) >= 1

    def test_passage_has_required_keys(self) -> None:
        payload = make_rerank_evidence_batch_payload()
        passage = payload["items"][0]["passages"][0]
        assert "passage_id" in passage
        assert "text" in passage

    def test_pydantic_validation(self) -> None:
        payload = make_rerank_evidence_batch_payload()
        req = RerankEvidenceBatchRequest(**payload)
        assert req.analysis_id == payload["analysis_id"]
        assert len(req.items) == len(payload["items"])


# -- nli_verify_batch payload -----------------------------------------------

class TestNliVerifyBatchPayload:
    """Validate make_nli_verify_batch_payload output."""

    def test_returns_dict(self) -> None:
        payload = make_nli_verify_batch_payload()
        assert isinstance(payload, dict)

    def test_has_pairs_list(self) -> None:
        payload = make_nli_verify_batch_payload()
        assert "pairs" in payload
        assert len(payload["pairs"]) >= 1

    def test_pair_has_required_keys(self) -> None:
        payload = make_nli_verify_batch_payload()
        pair = payload["pairs"][0]
        assert "pair_id" in pair
        assert "claim_id" in pair
        assert "passage_id" in pair
        assert "claim_text" in pair
        assert "passage_text" in pair

    def test_pydantic_validation(self) -> None:
        payload = make_nli_verify_batch_payload()
        req = NliVerifyBatchRequest(**payload)
        assert req.analysis_id == payload["analysis_id"]
        assert len(req.pairs) == len(payload["pairs"])


# -- compute_umap payload --------------------------------------------------

class TestComputeUmapPayload:
    """Validate make_compute_umap_payload output."""

    def test_returns_dict(self) -> None:
        payload = make_compute_umap_payload()
        assert isinstance(payload, dict)

    def test_has_vectors_dict(self) -> None:
        payload = make_compute_umap_payload()
        assert "vectors" in payload
        assert len(payload["vectors"]) >= 1

    def test_vector_dimension_matches(self) -> None:
        payload = make_compute_umap_payload()
        for vec in payload["vectors"].values():
            assert len(vec) == EMBEDDING_DIM

    def test_n_neighbors_present(self) -> None:
        payload = make_compute_umap_payload()
        assert "n_neighbors" in payload
        assert payload["n_neighbors"] >= 2

    def test_pydantic_validation(self) -> None:
        payload = make_compute_umap_payload()
        req = ComputeUmapRequest(**payload)
        assert req.analysis_id == payload["analysis_id"]
        assert len(req.vectors) == len(payload["vectors"])


# -- score_clusters payload -------------------------------------------------

class TestScoreClustersPayload:
    """Validate make_score_clusters_payload output."""

    def test_returns_dict(self) -> None:
        payload = make_score_clusters_payload()
        assert isinstance(payload, dict)

    def test_has_clusters_list(self) -> None:
        payload = make_score_clusters_payload()
        assert "clusters" in payload
        assert len(payload["clusters"]) >= 1

    def test_has_claims_dict(self) -> None:
        payload = make_score_clusters_payload()
        assert "claims" in payload
        assert len(payload["claims"]) >= 1

    def test_cluster_has_required_keys(self) -> None:
        payload = make_score_clusters_payload()
        cluster = payload["clusters"][0]
        assert "cluster_id" in cluster
        assert "claim_ids" in cluster
        assert "representative_claim_id" in cluster
        assert "representative_text" in cluster

    def test_nli_results_present(self) -> None:
        payload = make_score_clusters_payload()
        assert "nli_results" in payload
        assert isinstance(payload["nli_results"], list)

    def test_pydantic_validation(self) -> None:
        payload = make_score_clusters_payload()
        req = ScoreClustersRequest(**payload)
        assert req.analysis_id == payload["analysis_id"]
        assert len(req.clusters) == len(payload["clusters"])


# -- Dispatch table coverage ------------------------------------------------

class TestPayloadGeneratorsTable:
    """Validate the PAYLOAD_GENERATORS dispatch table."""

    def test_all_functions_have_generators(self) -> None:
        for fn_name in FUNCTION_NAMES:
            assert fn_name in PAYLOAD_GENERATORS

    def test_generators_return_dicts(self) -> None:
        for fn_name in FUNCTION_NAMES:
            gen = PAYLOAD_GENERATORS[fn_name]
            payload = gen()
            assert isinstance(payload, dict)

    def test_all_payloads_have_analysis_id(self) -> None:
        for fn_name in FUNCTION_NAMES:
            gen = PAYLOAD_GENERATORS[fn_name]
            payload = gen()
            assert "analysis_id" in payload
            assert len(payload["analysis_id"]) > 0
