"""Tests for evidence_retrieval.retrieve_evidence."""

import asyncio
import sys
import os
from hashlib import sha1
from unittest.mock import MagicMock, patch

# Stub duckduckgo_search before importing the module so tests run
# without the package installed.
if "duckduckgo_search" not in sys.modules:
    sys.modules["duckduckgo_search"] = MagicMock()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.evidence_retrieval import retrieve_evidence


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FAKE_RESULTS = [
    {"href": "https://example.com/a", "body": "First snippet text."},
    {"href": "https://example.com/b", "body": "Second snippet text."},
]


def _pid(claim_id: str, url: str) -> str:
    """Replicate the passage_id formula from evidence_retrieval."""
    return "p_" + sha1(f"{claim_id}:{url}".encode()).hexdigest()[:16]


def _ddgs_mock(results=_FAKE_RESULTS):
    """Return a DDGS class mock whose .text() returns results."""
    instance = MagicMock()
    instance.text.return_value = results
    return MagicMock(return_value=instance)


# ---------------------------------------------------------------------------
# TestPassageStructure
# ---------------------------------------------------------------------------


class TestPassageStructure:
    def test_returns_passages_for_valid_claim(self):
        with patch("duckduckgo_search.DDGS", _ddgs_mock()):
            result = asyncio.run(
                retrieve_evidence([{"claim_id": "c1", "claim_text": "speed of light"}])
            )
        assert "c1" in result
        assert len(result["c1"]) == 2

    def test_passage_has_required_keys(self):
        with patch("duckduckgo_search.DDGS", _ddgs_mock()):
            result = asyncio.run(
                retrieve_evidence([{"claim_id": "c1", "claim_text": "x"}])
            )
        p = result["c1"][0]
        assert "passage_id" in p
        assert "text" in p
        assert "source" in p

    def test_passage_text_and_source_correct(self):
        with patch("duckduckgo_search.DDGS", _ddgs_mock()):
            result = asyncio.run(
                retrieve_evidence([{"claim_id": "c1", "claim_text": "x"}])
            )
        assert result["c1"][0]["text"] == "First snippet text."
        assert result["c1"][0]["source"] == "https://example.com/a"

    def test_passage_id_matches_formula(self):
        with patch("duckduckgo_search.DDGS", _ddgs_mock()):
            result = asyncio.run(
                retrieve_evidence([{"claim_id": "c1", "claim_text": "x"}])
            )
        expected = _pid("c1", "https://example.com/a")
        assert result["c1"][0]["passage_id"] == expected

    def test_passage_id_is_deterministic(self):
        with patch("duckduckgo_search.DDGS", _ddgs_mock()):
            r1 = asyncio.run(
                retrieve_evidence([{"claim_id": "c1", "claim_text": "x"}])
            )
        with patch("duckduckgo_search.DDGS", _ddgs_mock()):
            r2 = asyncio.run(
                retrieve_evidence([{"claim_id": "c1", "claim_text": "x"}])
            )
        assert r1["c1"][0]["passage_id"] == r2["c1"][0]["passage_id"]

    def test_passage_ids_differ_across_claims(self):
        with patch("duckduckgo_search.DDGS", _ddgs_mock()):
            result = asyncio.run(
                retrieve_evidence([
                    {"claim_id": "c1", "claim_text": "x"},
                    {"claim_id": "c2", "claim_text": "x"},
                ])
            )
        assert result["c1"][0]["passage_id"] != result["c2"][0]["passage_id"]


# ---------------------------------------------------------------------------
# TestKeyFallbacks
# ---------------------------------------------------------------------------


class TestKeyFallbacks:
    def test_snippet_used_when_body_missing(self):
        results = [{"href": "https://example.com/s", "snippet": "Snippet text."}]
        with patch("duckduckgo_search.DDGS", _ddgs_mock(results)):
            result = asyncio.run(
                retrieve_evidence([{"claim_id": "c1", "claim_text": "x"}])
            )
        assert result["c1"][0]["text"] == "Snippet text."

    def test_url_key_used_when_href_missing(self):
        results = [{"url": "https://alt.com/1", "body": "Alt text."}]
        with patch("duckduckgo_search.DDGS", _ddgs_mock(results)):
            result = asyncio.run(
                retrieve_evidence([{"claim_id": "c1", "claim_text": "x"}])
            )
        assert result["c1"][0]["source"] == "https://alt.com/1"

    def test_result_with_empty_body_and_snippet_skipped(self):
        results = [
            {"href": "https://example.com/empty", "body": ""},
            {"href": "https://example.com/valid", "body": "Valid text."},
        ]
        with patch("duckduckgo_search.DDGS", _ddgs_mock(results)):
            result = asyncio.run(
                retrieve_evidence([{"claim_id": "c1", "claim_text": "x"}])
            )
        assert len(result["c1"]) == 1
        assert result["c1"][0]["text"] == "Valid text."

    def test_all_empty_body_returns_empty_list(self):
        results = [{"href": "https://example.com/x", "body": ""}]
        with patch("duckduckgo_search.DDGS", _ddgs_mock(results)):
            result = asyncio.run(
                retrieve_evidence([{"claim_id": "c1", "claim_text": "x"}])
            )
        assert result.get("c1") == []


# ---------------------------------------------------------------------------
# TestInputEdgeCases
# ---------------------------------------------------------------------------


class TestInputEdgeCases:
    def test_empty_claim_id_excluded_from_result(self):
        with patch("duckduckgo_search.DDGS", _ddgs_mock()):
            result = asyncio.run(
                retrieve_evidence([{"claim_id": "", "claim_text": "something"}])
            )
        assert "" not in result

    def test_empty_claim_text_skips_search_returns_empty_list(self):
        with patch("duckduckgo_search.DDGS", _ddgs_mock()) as mock_cls:
            result = asyncio.run(
                retrieve_evidence([{"claim_id": "c1", "claim_text": ""}])
            )
        assert result.get("c1") == []
        mock_cls.assert_not_called()

    def test_empty_claims_list(self):
        result = asyncio.run(retrieve_evidence([]))
        assert result == {}

    def test_multiple_claims_all_returned(self):
        with patch("duckduckgo_search.DDGS", _ddgs_mock()):
            result = asyncio.run(
                retrieve_evidence([
                    {"claim_id": "c1", "claim_text": "claim one"},
                    {"claim_id": "c2", "claim_text": "claim two"},
                ])
            )
        assert "c1" in result
        assert "c2" in result

    def test_no_results_returns_empty_list_for_claim(self):
        with patch("duckduckgo_search.DDGS", _ddgs_mock([])):
            result = asyncio.run(
                retrieve_evidence([{"claim_id": "c1", "claim_text": "x"}])
            )
        assert result.get("c1") == []


# ---------------------------------------------------------------------------
# TestErrorHandling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    def test_ddgs_constructor_raises_returns_empty_dict(self):
        failing_cls = MagicMock(side_effect=Exception("network error"))
        with patch("duckduckgo_search.DDGS", failing_cls):
            result = asyncio.run(
                retrieve_evidence([{"claim_id": "c1", "claim_text": "something"}])
            )
        assert result == {}

    def test_ddgs_text_raises_returns_empty_dict(self):
        instance = MagicMock()
        instance.text.side_effect = Exception("timeout")
        with patch("duckduckgo_search.DDGS", MagicMock(return_value=instance)):
            result = asyncio.run(
                retrieve_evidence([{"claim_id": "c1", "claim_text": "something"}])
            )
        assert result == {}

    def test_error_returns_empty_not_raises(self):
        with patch("duckduckgo_search.DDGS", MagicMock(side_effect=RuntimeError)):
            try:
                result = asyncio.run(
                    retrieve_evidence([{"claim_id": "c1", "claim_text": "x"}])
                )
            except Exception:
                assert False, "retrieve_evidence should not raise"
            assert result == {}
