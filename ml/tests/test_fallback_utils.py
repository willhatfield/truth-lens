"""Tests for fallback_utils.py — error response construction."""

import pytest
from fallback_utils import build_error_response
from schemas import (
    EmbedClaimsResponse,
    ClusterClaimsResponse,
    RerankEvidenceResponse,
    NliVerifyResponse,
    ComputeUmapResponse,
)


class TestBuildErrorResponse:
    def test_embed_error_response(self):
        resp = build_error_response(EmbedClaimsResponse, "boom")
        assert resp.error == "boom"
        assert resp.vectors == []
        assert resp.dimension == 0
        assert resp.model_name == ""

    def test_cluster_error_response(self):
        resp = build_error_response(ClusterClaimsResponse, "fail")
        assert resp.error == "fail"
        assert resp.clusters == []
        assert resp.num_clusters == 0

    def test_rerank_error_response(self):
        resp = build_error_response(RerankEvidenceResponse, "err")
        assert resp.error == "err"
        assert resp.ranked_passages == []

    def test_nli_error_response(self):
        resp = build_error_response(NliVerifyResponse, "nli fail")
        assert resp.error == "nli fail"
        assert resp.results == []

    def test_umap_error_response(self):
        resp = build_error_response(ComputeUmapResponse, "umap err")
        assert resp.error == "umap err"
        assert resp.coords_3d == []

    def test_error_message_preserved(self):
        msg = "something went very wrong"
        resp = build_error_response(EmbedClaimsResponse, msg)
        assert resp.error == msg

    def test_serializable(self):
        resp = build_error_response(EmbedClaimsResponse, "test")
        d = resp.model_dump()
        assert isinstance(d, dict)
        assert d["error"] == "test"
