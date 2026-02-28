"""Tests for fallback_utils.py — build_warning_response construction.

Covers all seven response types plus cross-cutting checks for
analysis_id preservation and schema_version == '1.0'.
"""

import pytest
from fallback_utils import build_warning_response
from schemas import (
    ExtractClaimsResponse,
    EmbedClaimsResponse,
    ClusterClaimsResponse,
    RerankEvidenceBatchResponse,
    NliVerifyBatchResponse,
    ComputeUmapResponse,
    ScoreClustersResponse,
)

ANALYSIS_ID = "test-analysis-001"
WARNING_MSG = "something went wrong"


class TestBuildWarningExtractClaims:
    """build_warning_response for ExtractClaimsResponse."""

    def test_warning_set_and_claims_empty(self):
        resp = build_warning_response(
            ExtractClaimsResponse, ANALYSIS_ID, WARNING_MSG,
        )
        assert resp.warnings == [WARNING_MSG]
        assert resp.claims == []
        assert resp.analysis_id == ANALYSIS_ID


class TestBuildWarningEmbedClaims:
    """build_warning_response for EmbedClaimsResponse."""

    def test_warning_set_vectors_empty_dim_zero(self):
        resp = build_warning_response(
            EmbedClaimsResponse, ANALYSIS_ID, WARNING_MSG,
        )
        assert resp.warnings == [WARNING_MSG]
        assert resp.vectors == {}
        assert resp.dim == 0
        assert resp.analysis_id == ANALYSIS_ID


class TestBuildWarningClusterClaims:
    """build_warning_response for ClusterClaimsResponse."""

    def test_warning_set_and_clusters_empty(self):
        resp = build_warning_response(
            ClusterClaimsResponse, ANALYSIS_ID, WARNING_MSG,
        )
        assert resp.warnings == [WARNING_MSG]
        assert resp.clusters == []
        assert resp.analysis_id == ANALYSIS_ID


class TestBuildWarningRerankEvidenceBatch:
    """build_warning_response for RerankEvidenceBatchResponse."""

    def test_warning_set_and_rankings_empty(self):
        resp = build_warning_response(
            RerankEvidenceBatchResponse, ANALYSIS_ID, WARNING_MSG,
        )
        assert resp.warnings == [WARNING_MSG]
        assert resp.rankings == []
        assert resp.analysis_id == ANALYSIS_ID


class TestBuildWarningNliVerifyBatch:
    """build_warning_response for NliVerifyBatchResponse."""

    def test_warning_set_and_results_empty(self):
        resp = build_warning_response(
            NliVerifyBatchResponse, ANALYSIS_ID, WARNING_MSG,
        )
        assert resp.warnings == [WARNING_MSG]
        assert resp.results == []
        assert resp.analysis_id == ANALYSIS_ID


class TestBuildWarningComputeUmap:
    """build_warning_response for ComputeUmapResponse."""

    def test_warning_set_and_coords3d_empty(self):
        resp = build_warning_response(
            ComputeUmapResponse, ANALYSIS_ID, WARNING_MSG,
        )
        assert resp.warnings == [WARNING_MSG]
        assert resp.coords3d == {}
        assert resp.analysis_id == ANALYSIS_ID


class TestBuildWarningScoreClusters:
    """build_warning_response for ScoreClustersResponse."""

    def test_warning_set_and_scores_empty(self):
        resp = build_warning_response(
            ScoreClustersResponse, ANALYSIS_ID, WARNING_MSG,
        )
        assert resp.warnings == [WARNING_MSG]
        assert resp.scores == []
        assert resp.analysis_id == ANALYSIS_ID


class TestCrossCuttingWarningResponse:
    """analysis_id preserved and schema_version == '1.0' for all types."""

    ALL_RESPONSE_CLASSES = [
        ExtractClaimsResponse,
        EmbedClaimsResponse,
        ClusterClaimsResponse,
        RerankEvidenceBatchResponse,
        NliVerifyBatchResponse,
        ComputeUmapResponse,
        ScoreClustersResponse,
    ]

    @pytest.mark.parametrize("response_class", ALL_RESPONSE_CLASSES)
    def test_analysis_id_preserved_and_schema_version(self, response_class):
        aid = "cross-cut-id-42"
        resp = build_warning_response(response_class, aid, "warn")
        assert resp.analysis_id == aid
        assert resp.schema_version == "1.0"
        assert isinstance(resp.warnings, list)
        assert len(resp.warnings) == 1
