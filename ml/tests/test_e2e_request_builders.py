"""Tests for e2e_request_builders module.

Covers all 7 builders, the dispatch function, field validation helpers,
and chaining-vs-mock detection logic.
"""

import pytest

from e2e_request_builders import (
    build_request,
    VALID_BUILDER_KEYS,
    CHAIN_DEPS,
    _has_deps,
    _build_extract,
    _build_embed,
    _build_cluster,
    _build_rerank,
    _build_nli,
    _build_umap,
    _build_score,
)
from e2e_modal_test import (
    _validate_response_fields,
    _is_chained,
)
from mock_data import (
    ANALYSIS_ID,
    build_extract_claims_response,
    build_embed_claims_response,
    build_cluster_claims_response,
    build_rerank_response,
    build_nli_response,
)

MAX_BUILDERS = 20


# ── Helpers to build fake upstream responses ─────────────────────────

def _fake_extract_response() -> dict:
    return build_extract_claims_response()


def _fake_embed_response() -> dict:
    return build_embed_claims_response()


def _fake_cluster_response() -> dict:
    return build_cluster_claims_response()


def _fake_rerank_response() -> dict:
    return build_rerank_response()


def _fake_nli_response() -> dict:
    return build_nli_response()


def _all_responses() -> dict:
    """Return a collected_responses dict as if every phase succeeded."""
    return {
        "extract_claims": _fake_extract_response(),
        "embed_claims": _fake_embed_response(),
        "cluster_claims": _fake_cluster_response(),
        "rerank_evidence_batch": _fake_rerank_response(),
        "nli_verify_batch": _fake_nli_response(),
    }


# ── Test: extract builder ────────────────────────────────────────────

class TestBuildExtract:
    def test_returns_dict_with_analysis_id(self):
        result = _build_extract({})
        assert result["analysis_id"] == ANALYSIS_ID

    def test_returns_responses_list(self):
        result = _build_extract({})
        assert "responses" in result
        assert isinstance(result["responses"], list)
        assert len(result["responses"]) > 0

    def test_ignores_prior_responses(self):
        result_empty = _build_extract({})
        result_full = _build_extract(_all_responses())
        assert result_empty == result_full


# ── Test: embed builder ──────────────────────────────────────────────

class TestBuildEmbed:
    def test_mock_fallback_when_no_deps(self):
        result = _build_embed({})
        assert "analysis_id" in result
        assert "claims" in result

    def test_chained_uses_extract_claims(self):
        responses = {"extract_claims": _fake_extract_response()}
        result = _build_embed(responses)
        assert "claims" in result
        extract_claims = responses["extract_claims"]["claims"]
        for i in range(len(result["claims"])):
            if i >= 10:
                break
            assert result["claims"][i]["claim_id"] == extract_claims[i]["claim_id"]

    def test_chained_preserves_analysis_id(self):
        resp = _fake_extract_response()
        resp["analysis_id"] = "custom-id-123"
        result = _build_embed({"extract_claims": resp})
        assert result["analysis_id"] == "custom-id-123"


# ── Test: cluster builder ────────────────────────────────────────────

class TestBuildCluster:
    def test_mock_fallback_when_no_deps(self):
        result = _build_cluster({})
        assert "analysis_id" in result
        assert "vectors" in result
        assert "claims" in result

    def test_chained_has_vectors_and_claims_meta(self):
        responses = {
            "embed_claims": _fake_embed_response(),
            "extract_claims": _fake_extract_response(),
        }
        result = _build_cluster(responses)
        assert isinstance(result["vectors"], dict)
        assert isinstance(result["claims"], dict)
        assert len(result["vectors"]) > 0
        assert len(result["claims"]) > 0

    def test_partial_deps_falls_back_to_mock(self):
        responses = {"embed_claims": _fake_embed_response()}
        result = _build_cluster(responses)
        assert "analysis_id" in result


# ── Test: rerank builder ─────────────────────────────────────────────

class TestBuildRerank:
    def test_mock_fallback_when_no_deps(self):
        result = _build_rerank({})
        assert "items" in result

    def test_chained_items_have_passages(self):
        responses = {"extract_claims": _fake_extract_response()}
        result = _build_rerank(responses)
        assert len(result["items"]) > 0
        for i in range(len(result["items"])):
            if i >= 10:
                break
            assert "passages" in result["items"][i]
            assert len(result["items"][i]["passages"]) > 0

    def test_chained_claim_ids_match_extract(self):
        extract_resp = _fake_extract_response()
        responses = {"extract_claims": extract_resp}
        result = _build_rerank(responses)
        extract_ids = set()
        for i in range(len(extract_resp["claims"])):
            extract_ids.add(extract_resp["claims"][i]["claim_id"])
        for i in range(len(result["items"])):
            if i >= 10:
                break
            assert result["items"][i]["claim_id"] in extract_ids


# ── Test: nli builder ────────────────────────────────────────────────

class TestBuildNli:
    def test_mock_fallback_when_no_deps(self):
        result = _build_nli({})
        assert "pairs" in result

    def test_chained_pairs_have_required_fields(self):
        responses = {
            "extract_claims": _fake_extract_response(),
            "rerank_evidence_batch": _fake_rerank_response(),
        }
        result = _build_nli(responses)
        assert len(result["pairs"]) > 0
        required = ["pair_id", "claim_id", "passage_id",
                     "claim_text", "passage_text"]
        for i in range(len(result["pairs"])):
            if i >= 10:
                break
            for field in required:
                assert field in result["pairs"][i], (
                    f"Missing {field} in pair {i}"
                )

    def test_chained_uses_top_ranked_passage(self):
        rerank_resp = _fake_rerank_response()
        responses = {
            "extract_claims": _fake_extract_response(),
            "rerank_evidence_batch": rerank_resp,
        }
        result = _build_nli(responses)
        top_ids = set()
        for i in range(len(rerank_resp["rankings"])):
            if i >= 10:
                break
            ordered = rerank_resp["rankings"][i]["ordered_passage_ids"]
            if len(ordered) > 0:
                top_ids.add(ordered[0])
        for i in range(len(result["pairs"])):
            if i >= 10:
                break
            assert result["pairs"][i]["passage_id"] in top_ids


# ── Test: umap builder ──────────────────────────────────────────────

class TestBuildUmap:
    def test_mock_fallback_when_no_deps(self):
        result = _build_umap({})
        assert "vectors" in result
        assert "n_neighbors" in result

    def test_chained_n_neighbors_bounded(self):
        responses = {"embed_claims": _fake_embed_response()}
        result = _build_umap(responses)
        n_vecs = len(result["vectors"])
        assert result["n_neighbors"] <= 15
        assert result["n_neighbors"] < n_vecs

    def test_chained_vectors_from_embed(self):
        embed_resp = _fake_embed_response()
        responses = {"embed_claims": embed_resp}
        result = _build_umap(responses)
        assert result["vectors"] == embed_resp["vectors"]


# ── Test: score builder ─────────────────────────────────────────────

class TestBuildScore:
    def test_mock_fallback_when_no_deps(self):
        result = _build_score({})
        assert "clusters" in result
        assert "claims" in result
        assert "nli_results" in result

    def test_chained_has_all_fields(self):
        responses = {
            "cluster_claims": _fake_cluster_response(),
            "extract_claims": _fake_extract_response(),
            "nli_verify_batch": _fake_nli_response(),
        }
        result = _build_score(responses)
        assert isinstance(result["clusters"], list)
        assert isinstance(result["claims"], dict)
        assert isinstance(result["nli_results"], list)

    def test_partial_deps_falls_back_to_mock(self):
        responses = {"cluster_claims": _fake_cluster_response()}
        result = _build_score(responses)
        assert "analysis_id" in result


# ── Test: dispatch function ──────────────────────────────────────────

class TestBuildRequestDispatch:
    def test_all_valid_keys_dispatch(self):
        keys = list(VALID_BUILDER_KEYS)
        for i in range(len(keys)):
            if i >= MAX_BUILDERS:
                break
            result = build_request(keys[i], {})
            assert isinstance(result, dict)
            assert "analysis_id" in result

    def test_invalid_key_raises(self):
        with pytest.raises(ValueError):
            build_request("nonexistent_phase", {})

    def test_dispatch_with_chained_data(self):
        responses = _all_responses()
        result = build_request("score", responses)
        assert "clusters" in result
        assert "nli_results" in result


# ── Test: _has_deps helper ───────────────────────────────────────────

class TestHasDeps:
    def test_extract_always_true(self):
        assert _has_deps("extract", {}) is True

    def test_embed_false_without_extract(self):
        assert _has_deps("embed", {}) is False

    def test_embed_true_with_extract(self):
        assert _has_deps("embed", {"extract_claims": {}}) is True

    def test_score_needs_three_deps(self):
        assert _has_deps("score", {}) is False
        partial = {"cluster_claims": {}, "extract_claims": {}}
        assert _has_deps("score", partial) is False
        full = {
            "cluster_claims": {},
            "extract_claims": {},
            "nli_verify_batch": {},
        }
        assert _has_deps("score", full) is True


# ── Test: _validate_response_fields ──────────────────────────────────

class TestValidateResponseFields:
    def test_all_present_returns_valid(self):
        data = {"claims": [1, 2], "dim": 8}
        valid, detail = _validate_response_fields(data, ["claims", "dim"])
        assert valid is True
        assert detail == ""

    def test_missing_one_field(self):
        data = {"claims": [1, 2]}
        valid, detail = _validate_response_fields(data, ["claims", "dim"])
        assert valid is False
        assert "dim" in detail
        assert detail.startswith("MISSING:")

    def test_all_missing(self):
        valid, detail = _validate_response_fields({}, ["a", "b"])
        assert valid is False
        assert "a" in detail
        assert "b" in detail

    def test_empty_field_list_is_valid(self):
        valid, detail = _validate_response_fields({}, [])
        assert valid is True


# ── Test: _is_chained ────────────────────────────────────────────────

class TestIsChained:
    def test_extract_always_chained(self):
        assert _is_chained("extract", {}) is True

    def test_embed_not_chained_without_extract(self):
        assert _is_chained("embed", {}) is False

    def test_embed_chained_with_extract(self):
        assert _is_chained("embed", {"extract_claims": {}}) is True

    def test_score_chained_with_all_deps(self):
        responses = {
            "cluster_claims": {},
            "extract_claims": {},
            "nli_verify_batch": {},
        }
        assert _is_chained("score", responses) is True


# ── Test: chaining detection across full pipeline ────────────────────

class TestChainingDetection:
    def test_first_phase_always_chained(self):
        assert _is_chained("extract", {}) is True

    def test_second_phase_chained_after_first_succeeds(self):
        collected = {"extract_claims": _fake_extract_response()}
        assert _is_chained("embed", collected) is True

    def test_second_phase_mock_when_first_missing(self):
        assert _is_chained("embed", {}) is False

    def test_full_pipeline_all_chained(self):
        responses = _all_responses()
        keys_in_order = [
            "extract", "embed", "cluster",
            "rerank", "nli", "umap", "score",
        ]
        for i in range(len(keys_in_order)):
            key = keys_in_order[i]
            assert _is_chained(key, responses) is True, (
                f"{key} should be chained with all responses present"
            )


# ── Test: CHAIN_DEPS completeness ───────────────────────────────────

class TestChainDepsCompleteness:
    def test_every_builder_key_has_deps_entry(self):
        for key in VALID_BUILDER_KEYS:
            assert key in CHAIN_DEPS, f"{key} missing from CHAIN_DEPS"

    def test_deps_are_lists_of_strings(self):
        for key in CHAIN_DEPS:
            deps = CHAIN_DEPS[key]
            assert isinstance(deps, list)
            for i in range(len(deps)):
                assert isinstance(deps[i], str)
