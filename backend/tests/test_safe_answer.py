"""Tests for the safe-answer builder."""

import sys
import os

# Ensure backend root is on path so 'app.safe_answer' resolves
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.safe_answer import (
    _ClusterData,
    _ScoreData,
    _apply_transition_rewrite,
    _base_sentences,
    _dedup_clusters,
    _dedup_scores,
    _FALLBACK_TEXT,
    _MAX_CLUSTERS,
    _sorted_scores,
    _try_parse_cluster,
    _try_parse_score,
    _validate_prefix,
    _verdict_rank,
    build_safe_answer,
)


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------


def _make_cluster(cluster_id, text):
    """Return a valid cluster dict."""
    return {
        "cluster_id": cluster_id,
        "claim_ids": ["c1"],
        "representative_claim_id": "c1",
        "representative_text": text,
    }


def _make_score(cluster_id, trust_score, verdict):
    """Return a valid score dict."""
    return {
        "cluster_id": cluster_id,
        "trust_score": trust_score,
        "verdict": verdict,
    }


# ---------------------------------------------------------------------------
# TestTryParseCluster
# ---------------------------------------------------------------------------


class TestTryParseCluster:
    def test_valid_dict(self):
        raw = {"cluster_id": "c1", "representative_text": "hello"}
        result = _try_parse_cluster(raw)
        assert result == _ClusterData(cluster_id="c1", representative_text="hello")

    def test_missing_cluster_id(self):
        raw = {"representative_text": "hello"}
        assert _try_parse_cluster(raw) is None

    def test_missing_representative_text(self):
        raw = {"cluster_id": "c1"}
        assert _try_parse_cluster(raw) is None

    def test_empty_cluster_id(self):
        raw = {"cluster_id": "", "representative_text": "hello"}
        assert _try_parse_cluster(raw) is None

    def test_empty_representative_text(self):
        raw = {"cluster_id": "c1", "representative_text": ""}
        assert _try_parse_cluster(raw) is None

    def test_non_str_cluster_id(self):
        raw = {"cluster_id": 123, "representative_text": "hello"}
        assert _try_parse_cluster(raw) is None

    def test_non_dict_input(self):
        assert _try_parse_cluster("not a dict") is None
        assert _try_parse_cluster(None) is None
        assert _try_parse_cluster(42) is None
        assert _try_parse_cluster([]) is None


# ---------------------------------------------------------------------------
# TestTryParseScore
# ---------------------------------------------------------------------------


class TestTryParseScore:
    def test_valid_dict(self):
        raw = {"cluster_id": "c1", "trust_score": 80, "verdict": "SAFE"}
        result = _try_parse_score(raw)
        assert result == _ScoreData(cluster_id="c1", trust_score=80, verdict="SAFE")

    def test_missing_cluster_id(self):
        raw = {"trust_score": 80, "verdict": "SAFE"}
        assert _try_parse_score(raw) is None

    def test_missing_trust_score(self):
        raw = {"cluster_id": "c1", "verdict": "SAFE"}
        assert _try_parse_score(raw) is None

    def test_missing_verdict(self):
        raw = {"cluster_id": "c1", "trust_score": 80}
        assert _try_parse_score(raw) is None

    def test_non_int_trust_score(self):
        raw = {"cluster_id": "c1", "trust_score": 80.5, "verdict": "SAFE"}
        assert _try_parse_score(raw) is None

    def test_bool_trust_score(self):
        raw = {"cluster_id": "c1", "trust_score": True, "verdict": "SAFE"}
        assert _try_parse_score(raw) is None

    def test_trust_score_below_zero(self):
        raw = {"cluster_id": "c1", "trust_score": -1, "verdict": "SAFE"}
        assert _try_parse_score(raw) is None

    def test_trust_score_above_100(self):
        raw = {"cluster_id": "c1", "trust_score": 101, "verdict": "SAFE"}
        assert _try_parse_score(raw) is None

    def test_empty_verdict(self):
        raw = {"cluster_id": "c1", "trust_score": 80, "verdict": ""}
        assert _try_parse_score(raw) is None

    def test_non_dict_input(self):
        assert _try_parse_score("not a dict") is None
        assert _try_parse_score(None) is None


# ---------------------------------------------------------------------------
# TestDedupClusters
# ---------------------------------------------------------------------------


class TestDedupClusters:
    def test_no_duplicates(self):
        parsed = [
            _ClusterData("c1", "hello"),
            _ClusterData("c2", "world"),
        ]
        warnings = []
        result = _dedup_clusters(parsed, warnings)
        assert len(result) == 2
        assert len(warnings) == 0

    def test_duplicate_keeps_longest_text(self):
        parsed = [
            _ClusterData("c1", "short"),
            _ClusterData("c1", "much longer text"),
        ]
        warnings = []
        result = _dedup_clusters(parsed, warnings)
        assert len(result) == 1
        assert result["c1"].representative_text == "much longer text"
        assert len(warnings) == 1

    def test_equal_length_keeps_lex_smallest(self):
        parsed = [
            _ClusterData("c1", "beta"),
            _ClusterData("c1", "alfa"),
        ]
        warnings = []
        result = _dedup_clusters(parsed, warnings)
        assert result["c1"].representative_text == "alfa"
        assert len(warnings) == 1

    def test_warning_per_duplicate(self):
        parsed = [
            _ClusterData("c1", "aaa"),
            _ClusterData("c1", "bbb"),
            _ClusterData("c1", "ccc"),
        ]
        warnings = []
        _dedup_clusters(parsed, warnings)
        assert len(warnings) == 2


# ---------------------------------------------------------------------------
# TestDedupScores
# ---------------------------------------------------------------------------


class TestDedupScores:
    def test_no_duplicates(self):
        parsed = [
            _ScoreData("c1", 80, "SAFE"),
            _ScoreData("c2", 60, "CAUTION"),
        ]
        warnings = []
        result = _dedup_scores(parsed, warnings)
        assert len(result) == 2
        assert len(warnings) == 0

    def test_duplicate_keeps_highest_trust(self):
        parsed = [
            _ScoreData("c1", 60, "SAFE"),
            _ScoreData("c1", 90, "SAFE"),
        ]
        warnings = []
        result = _dedup_scores(parsed, warnings)
        assert len(result) == 1
        assert result[0].trust_score == 90
        assert len(warnings) == 1

    def test_equal_trust_keeps_best_verdict(self):
        parsed = [
            _ScoreData("c1", 80, "CAUTION"),
            _ScoreData("c1", 80, "SAFE"),
        ]
        warnings = []
        result = _dedup_scores(parsed, warnings)
        assert len(result) == 1
        assert result[0].verdict == "SAFE"
        assert len(warnings) == 1

    def test_warning_per_duplicate(self):
        parsed = [
            _ScoreData("c1", 80, "SAFE"),
            _ScoreData("c1", 90, "SAFE"),
            _ScoreData("c1", 95, "SAFE"),
        ]
        warnings = []
        _dedup_scores(parsed, warnings)
        assert len(warnings) == 2


# ---------------------------------------------------------------------------
# TestVerdictRank
# ---------------------------------------------------------------------------


class TestVerdictRank:
    def test_safe(self):
        assert _verdict_rank("SAFE") == 0

    def test_caution(self):
        assert _verdict_rank("CAUTION") == 1

    def test_reject(self):
        assert _verdict_rank("REJECT") == 2

    def test_unknown(self):
        assert _verdict_rank("UNKNOWN_VERDICT") == 2
        assert _verdict_rank("") == 2


# ---------------------------------------------------------------------------
# TestSortedScores
# ---------------------------------------------------------------------------


class TestSortedScores:
    def test_verdict_ordering(self):
        scores = [
            _ScoreData("c3", 50, "REJECT"),
            _ScoreData("c1", 90, "SAFE"),
            _ScoreData("c2", 70, "CAUTION"),
        ]
        result = _sorted_scores(scores)
        assert result[0].verdict == "SAFE"
        assert result[1].verdict == "CAUTION"
        assert result[2].verdict == "REJECT"

    def test_trust_descending_within_verdict(self):
        scores = [
            _ScoreData("c1", 60, "SAFE"),
            _ScoreData("c2", 90, "SAFE"),
        ]
        result = _sorted_scores(scores)
        assert result[0].trust_score == 90
        assert result[1].trust_score == 60

    def test_cluster_id_tiebreak(self):
        scores = [
            _ScoreData("c2", 80, "SAFE"),
            _ScoreData("c1", 80, "SAFE"),
        ]
        result = _sorted_scores(scores)
        assert result[0].cluster_id == "c1"
        assert result[1].cluster_id == "c2"

    def test_empty_input(self):
        assert _sorted_scores([]) == []

    def test_deterministic(self):
        scores = [
            _ScoreData("c3", 50, "REJECT"),
            _ScoreData("c1", 90, "SAFE"),
            _ScoreData("c2", 70, "CAUTION"),
        ]
        first = _sorted_scores(scores)
        second = _sorted_scores(scores)
        for i in range(len(first)):
            assert first[i].cluster_id == second[i].cluster_id


# ---------------------------------------------------------------------------
# TestValidatePrefix
# ---------------------------------------------------------------------------


class TestValidatePrefix:
    def test_allowed_prefix_at_index_gt_0(self):
        assert _validate_prefix("In addition,", 1) is True
        assert _validate_prefix("Additionally,", 2) is True
        assert _validate_prefix("However,", 1) is True
        assert _validate_prefix("Because of this,", 3) is True
        assert _validate_prefix("As a result,", 1) is True
        assert _validate_prefix("Meanwhile,", 1) is True
        assert _validate_prefix("Overall,", 1) is True
        assert _validate_prefix("Still,", 1) is True

    def test_empty_string_any_index(self):
        assert _validate_prefix("", 0) is True
        assert _validate_prefix("", 5) is True

    def test_disallowed_string(self):
        assert _validate_prefix("Not a real prefix,", 1) is False
        assert _validate_prefix("Random", 0) is False

    def test_contrastive_at_index_0(self):
        assert _validate_prefix("However,", 0) is False
        assert _validate_prefix("Still,", 0) is False
        assert _validate_prefix("Because of this,", 0) is False
        assert _validate_prefix("As a result,", 0) is False

    def test_opener_at_index_0(self):
        assert _validate_prefix("Overall,", 0) is True
        assert _validate_prefix("Meanwhile,", 0) is True
        assert _validate_prefix("In addition,", 0) is True
        assert _validate_prefix("Additionally,", 0) is True


# ---------------------------------------------------------------------------
# TestApplyTransitionRewrite
# ---------------------------------------------------------------------------


class TestApplyTransitionRewrite:
    def test_valid_prefixes_prepended(self):
        sentences = ["Claim A.", "Claim B."]
        client = lambda s: ["", "However,"]
        result = _apply_transition_rewrite(sentences, client)
        assert result[0] == "Claim A."
        assert result[1] == "However, Claim B."

    def test_length_mismatch_full_fallback(self):
        sentences = ["Claim A.", "Claim B."]
        client = lambda s: [""]  # wrong length
        result = _apply_transition_rewrite(sentences, client)
        assert result == ["Claim A.", "Claim B."]

    def test_invalid_prefix_per_sentence_fallback(self):
        sentences = ["Claim A.", "Claim B.", "Claim C."]
        client = lambda s: ["", "INVALID_PREFIX", "Additionally,"]
        result = _apply_transition_rewrite(sentences, client)
        assert result[0] == "Claim A."
        assert result[1] == "Claim B."  # invalid kept original
        assert result[2] == "Additionally, Claim C."

    def test_contrastive_at_index_0_fallback(self):
        sentences = ["Claim A.", "Claim B."]
        client = lambda s: ["However,", "Additionally,"]
        result = _apply_transition_rewrite(sentences, client)
        assert result[0] == "Claim A."  # However at index 0 rejected
        assert result[1] == "Additionally, Claim B."

    def test_exception_full_fallback(self):
        sentences = ["Claim A.", "Claim B."]
        def bad_client(s):
            raise ValueError("boom")
        result = _apply_transition_rewrite(sentences, bad_client)
        assert result == ["Claim A.", "Claim B."]

    def test_empty_list(self):
        result = _apply_transition_rewrite([], lambda s: [])
        assert result == []


# ---------------------------------------------------------------------------
# TestBaseSentences
# ---------------------------------------------------------------------------


class TestBaseSentences:
    def _make_index(self, pairs):
        """Build cluster_index from (cluster_id, text) pairs."""
        return {cid: _ClusterData(cid, text) for cid, text in pairs}

    def test_safe_lead(self):
        scores = [_ScoreData("c1", 90, "SAFE")]
        index = self._make_index([("c1", "the earth is round")])
        result = _base_sentences(scores, index)
        assert len(result) == 1
        assert "Verified sources support the following:" in result[0]
        assert "the earth is round" in result[0]

    def test_multiple_safe_support(self):
        scores = [
            _ScoreData("c1", 90, "SAFE"),
            _ScoreData("c2", 85, "SAFE"),
            _ScoreData("c3", 80, "SAFE"),
        ]
        index = self._make_index([
            ("c1", "claim one"), ("c2", "claim two"), ("c3", "claim three"),
        ])
        result = _base_sentences(scores, index)
        assert len(result) == 3  # lead + 2 support
        assert "Verified sources" in result[0]
        assert "claim two." == result[1]
        assert "claim three." == result[2]

    def test_safe_with_caution_caveat(self):
        scores = [
            _ScoreData("c1", 90, "SAFE"),
            _ScoreData("c2", 50, "CAUTION"),
        ]
        index = self._make_index([("c1", "safe claim"), ("c2", "maybe claim")])
        result = _base_sentences(scores, index)
        assert len(result) == 2
        assert "It is worth noting that" in result[1]
        assert "limited support" in result[1]

    def test_safe_with_reject_conflict(self):
        scores = [
            _ScoreData("c1", 90, "SAFE"),
            _ScoreData("c2", 20, "REJECT"),
        ]
        index = self._make_index([("c1", "safe claim"), ("c2", "bad claim")])
        result = _base_sentences(scores, index)
        assert len(result) == 2
        assert "was not verified" in result[1]

    def test_caution_only_lead(self):
        scores = [_ScoreData("c1", 50, "CAUTION")]
        index = self._make_index([("c1", "uncertain claim")])
        result = _base_sentences(scores, index)
        assert "The available evidence is uncertain" in result[0]
        assert "uncertain claim" in result[0]

    def test_all_reject_fallback(self):
        scores = [_ScoreData("c1", 10, "REJECT")]
        index = self._make_index([("c1", "bad claim")])
        result = _base_sentences(scores, index)
        assert result == [_FALLBACK_TEXT]

    def test_empty_input_fallback(self):
        result = _base_sentences([], {})
        assert result == [_FALLBACK_TEXT]

    def test_missing_cluster_id_skipped(self):
        scores = [
            _ScoreData("c1", 90, "SAFE"),
            _ScoreData("missing", 80, "SAFE"),
        ]
        index = self._make_index([("c1", "real claim")])
        result = _base_sentences(scores, index)
        assert len(result) == 1
        assert "real claim" in result[0]


# ---------------------------------------------------------------------------
# TestBuildSafeAnswer (integration)
# ---------------------------------------------------------------------------


class TestBuildSafeAnswer:
    def test_returns_tuple(self):
        clusters = [_make_cluster("c1", "hello")]
        scores = [_make_score("c1", 90, "SAFE")]
        result = build_safe_answer(clusters, scores)
        assert isinstance(result, tuple)
        assert len(result) == 2
        sa, warns = result
        assert "text" in sa
        assert "supported_cluster_ids" in sa
        assert "rejected_cluster_ids" in sa
        assert isinstance(warns, list)

    def test_deterministic_without_rewrite(self):
        clusters = [
            _make_cluster("c1", "first"),
            _make_cluster("c2", "second"),
        ]
        scores = [
            _make_score("c1", 90, "SAFE"),
            _make_score("c2", 50, "CAUTION"),
        ]
        r1 = build_safe_answer(clusters, scores)
        r2 = build_safe_answer(clusters, scores)
        assert r1 == r2

    def test_supported_cluster_ids(self):
        clusters = [
            _make_cluster("c1", "safe one"),
            _make_cluster("c2", "safe two"),
        ]
        scores = [
            _make_score("c1", 90, "SAFE"),
            _make_score("c2", 80, "SAFE"),
        ]
        sa, _ = build_safe_answer(clusters, scores)
        assert "c1" in sa["supported_cluster_ids"]
        assert "c2" in sa["supported_cluster_ids"]

    def test_rejected_cluster_ids(self):
        clusters = [_make_cluster("c1", "bad claim")]
        scores = [_make_score("c1", 10, "REJECT")]
        sa, _ = build_safe_answer(clusters, scores)
        assert "c1" in sa["rejected_cluster_ids"]
        assert sa["supported_cluster_ids"] == []

    def test_caution_not_in_either_list(self):
        clusters = [_make_cluster("c1", "maybe")]
        scores = [_make_score("c1", 50, "CAUTION")]
        sa, _ = build_safe_answer(clusters, scores)
        assert "c1" not in sa["supported_cluster_ids"]
        assert "c1" not in sa["rejected_cluster_ids"]

    def test_score_missing_cluster_excluded(self):
        clusters = [_make_cluster("c1", "real")]
        scores = [
            _make_score("c1", 90, "SAFE"),
            _make_score("c_missing", 80, "SAFE"),
        ]
        sa, _ = build_safe_answer(clusters, scores)
        assert "c_missing" not in sa["supported_cluster_ids"]

    def test_no_safe_fallback(self):
        clusters = [_make_cluster("c1", "rejected")]
        scores = [_make_score("c1", 10, "REJECT")]
        sa, _ = build_safe_answer(clusters, scores)
        assert _FALLBACK_TEXT in sa["text"]

    def test_rewrite_accepted(self):
        clusters = [
            _make_cluster("c1", "claim one"),
            _make_cluster("c2", "claim two"),
        ]
        scores = [
            _make_score("c1", 90, "SAFE"),
            _make_score("c2", 80, "SAFE"),
        ]
        client = lambda s: [""] * len(s)
        sa, _ = build_safe_answer(clusters, scores, rewrite_client=client)
        assert "claim one" in sa["text"]

    def test_rewrite_partial(self):
        clusters = [
            _make_cluster("c1", "claim one"),
            _make_cluster("c2", "claim two"),
        ]
        scores = [
            _make_score("c1", 90, "SAFE"),
            _make_score("c2", 85, "SAFE"),
        ]
        client = lambda s: ["", "INVALID_PREFIX"]
        sa, _ = build_safe_answer(clusters, scores, rewrite_client=client)
        assert "claim one" in sa["text"]
        assert "claim two" in sa["text"]

    def test_malformed_cluster_skipped_with_warning(self):
        clusters = [{"bad": "data"}, _make_cluster("c1", "good")]
        scores = [_make_score("c1", 90, "SAFE")]
        sa, warns = build_safe_answer(clusters, scores)
        assert any("malformed cluster" in w.lower() for w in warns)
        assert "good" in sa["text"]

    def test_malformed_score_skipped_with_warning(self):
        clusters = [_make_cluster("c1", "good")]
        scores = [{"bad": "data"}, _make_score("c1", 90, "SAFE")]
        sa, warns = build_safe_answer(clusters, scores)
        assert any("malformed score" in w.lower() for w in warns)
        assert "good" in sa["text"]

    def test_duplicate_cluster_deduped_with_warning(self):
        clusters = [
            _make_cluster("c1", "short"),
            _make_cluster("c1", "longer text"),
        ]
        scores = [_make_score("c1", 90, "SAFE")]
        sa, warns = build_safe_answer(clusters, scores)
        assert any("duplicate" in w.lower() for w in warns)
        assert "longer text" in sa["text"]

    def test_duplicate_score_deduped_with_warning(self):
        clusters = [_make_cluster("c1", "claim")]
        scores = [
            _make_score("c1", 60, "SAFE"),
            _make_score("c1", 90, "SAFE"),
        ]
        sa, warns = build_safe_answer(clusters, scores)
        assert any("duplicate" in w.lower() for w in warns)

    def test_truncation_warning(self):
        clusters = []
        scores = []
        for i in range(_MAX_CLUSTERS + 5):
            cid = f"c{i:05d}"
            clusters.append(_make_cluster(cid, f"text {i}"))
            scores.append(_make_score(cid, max(0, 100 - i), "SAFE"))
        sa, warns = build_safe_answer(clusters, scores)
        assert any("truncated" in w.lower() for w in warns)

    def test_empty_inputs_fallback(self):
        sa, warns = build_safe_answer([], [])
        assert sa["text"] == _FALLBACK_TEXT
        assert sa["supported_cluster_ids"] == []
        assert sa["rejected_cluster_ids"] == []

    def test_warnings_separate_from_safe_answer(self):
        clusters = [{"bad": True}]
        scores = []
        sa, warns = build_safe_answer(clusters, scores)
        assert "warnings" not in sa
        assert isinstance(warns, list)

    def test_never_raises_on_none_input(self):
        sa, warns = build_safe_answer(None, None)
        assert isinstance(sa, dict)
        assert isinstance(warns, list)

    def test_never_raises_on_int_input(self):
        sa, warns = build_safe_answer(42, 99)
        assert isinstance(sa, dict)
        assert isinstance(warns, list)

    def test_safe_caution_reject_full_mix(self):
        clusters = [
            _make_cluster("c1", "safe claim"),
            _make_cluster("c2", "caution claim"),
            _make_cluster("c3", "reject claim"),
        ]
        scores = [
            _make_score("c1", 90, "SAFE"),
            _make_score("c2", 50, "CAUTION"),
            _make_score("c3", 10, "REJECT"),
        ]
        sa, warns = build_safe_answer(clusters, scores)
        assert "c1" in sa["supported_cluster_ids"]
        assert "c3" in sa["rejected_cluster_ids"]
        assert "c2" not in sa["supported_cluster_ids"]
        assert "c2" not in sa["rejected_cluster_ids"]
        assert "safe claim" in sa["text"]
        assert "caution claim" in sa["text"]
        assert "reject claim" in sa["text"]
