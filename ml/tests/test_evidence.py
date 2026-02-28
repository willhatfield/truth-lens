"""Tests for evidence retrieval stub."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from evidence import retrieve_evidence


class TestRetrieveEvidence:
    """Tests for the retrieve_evidence function."""

    def test_returns_empty_lists(self):
        """Given 3 claims, returns dict with 3 keys each mapping to []."""
        claims = [
            {"claim_id": "c1", "claim_text": "The sky is blue."},
            {"claim_id": "c2", "claim_text": "Water is wet."},
            {"claim_id": "c3", "claim_text": "Fire is hot."},
        ]
        result = retrieve_evidence(claims, "analysis-001")

        assert isinstance(result, dict)
        assert len(result) == 3
        assert result["c1"] == []
        assert result["c2"] == []
        assert result["c3"] == []

    def test_empty_claims_input(self):
        """Given [], returns {}."""
        result = retrieve_evidence([], "analysis-002")

        assert isinstance(result, dict)
        assert len(result) == 0
        assert result == {}

    def test_ignores_missing_claim_id(self):
        """Claims without 'claim_id' key are skipped."""
        claims = [
            {"claim_id": "c1", "claim_text": "Valid claim."},
            {"claim_text": "No id here."},
            {"claim_id": "", "claim_text": "Empty id."},
            {"claim_id": "c4", "claim_text": "Another valid claim."},
        ]
        result = retrieve_evidence(claims, "analysis-003")

        assert isinstance(result, dict)
        assert len(result) == 2
        assert "c1" in result
        assert "c4" in result
        assert result["c1"] == []
        assert result["c4"] == []
