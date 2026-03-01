"""Tests for scoring.py trust-score algorithm functions."""

import types

from scoring import (
    TOTAL_MODELS,
    check_has_evidence,
    clamp,
    compute_agreement_score,
    compute_consistency_score,
    compute_independence_score,
    compute_trust_score,
    compute_verification_score,
    determine_verdict,
    find_best_nli_for_cluster,
    find_supporting_models,
)


# ---------------------------------------------------------------------------
# clamp
# ---------------------------------------------------------------------------


def test_clamp_below_min():
    """Value below min is clamped to min."""
    assert clamp(-5.0, 0.0, 100.0) == 0.0


def test_clamp_above_max():
    """Value above max is clamped to max."""
    assert clamp(150.0, 0.0, 100.0) == 100.0


def test_clamp_within_range():
    """Value within range is returned unchanged."""
    assert clamp(42.0, 0.0, 100.0) == 42.0


# ---------------------------------------------------------------------------
# check_has_evidence
# ---------------------------------------------------------------------------


def test_check_has_evidence_empty_string():
    """Empty passage ID means no evidence."""
    assert check_has_evidence("") is False


def test_check_has_evidence_stub_prefix():
    """Passage IDs starting with p_default_ are stubs."""
    assert check_has_evidence("p_default_claim1") is False


def test_check_has_evidence_real_passage():
    """A real passage ID returns True."""
    assert check_has_evidence("wiki-earth-shape-001") is True


# ---------------------------------------------------------------------------
# compute_agreement_score
# ---------------------------------------------------------------------------


def test_agreement_zero_supporting():
    """Zero supporting models produces score of 0."""
    assert compute_agreement_score(0, TOTAL_MODELS) == 0.0


def test_agreement_three_of_five():
    """Three of five models produces 60.0."""
    assert compute_agreement_score(3, TOTAL_MODELS) == 60.0


def test_agreement_five_of_five():
    """All models supporting produces 100.0."""
    assert compute_agreement_score(5, TOTAL_MODELS) == 100.0


def test_agreement_zero_total_models():
    """Zero total_models edge case returns 0.0 (no division by zero)."""
    assert compute_agreement_score(3, 0) == 0.0


# ---------------------------------------------------------------------------
# compute_verification_score
# ---------------------------------------------------------------------------


def test_verification_high_ent_low_contra():
    """High entailment, low contradiction yields a positive score."""
    score = compute_verification_score(0.9, 0.05)
    assert score == (100.0 * 0.9 - 100.0 * 0.05)


def test_verification_low_ent_high_contra():
    """Low entailment, high contradiction yields a negative score."""
    score = compute_verification_score(0.1, 0.8)
    assert score < 0.0


def test_verification_both_zero():
    """Both zero produces exactly 0.0."""
    assert compute_verification_score(0.0, 0.0) == 0.0


# ---------------------------------------------------------------------------
# compute_independence_score
# ---------------------------------------------------------------------------


def test_independence_all_models_support():
    """All models supporting yields 100.0."""
    assert compute_independence_score(5, 5) == 100.0


def test_independence_partial_support():
    """Three of five models yields 60.0."""
    assert compute_independence_score(3, 5) == 60.0


def test_independence_zero_total():
    """Zero total models returns 0.0 without division by zero."""
    assert compute_independence_score(3, 0) == 0.0


# ---------------------------------------------------------------------------
# compute_consistency_score
# ---------------------------------------------------------------------------


def test_consistency_no_contradiction():
    """Zero contradiction yields 100.0."""
    assert compute_consistency_score(0.0) == 100.0


def test_consistency_full_contradiction():
    """Full contradiction yields 0.0."""
    assert compute_consistency_score(1.0) == 0.0


def test_consistency_partial_contradiction():
    """0.2 contradiction yields 80.0."""
    assert compute_consistency_score(0.2) == 80.0


# ---------------------------------------------------------------------------
# compute_trust_score
# ---------------------------------------------------------------------------


def test_trust_score_typical():
    """Formula: 0.35*agreement + 0.35*verification + 0.15*independence + 0.15*consistency + 0.35."""
    agreement = 80.0
    verification = 70.0
    independence = 60.0
    consistency = 90.0
    result = compute_trust_score(agreement, verification, independence, consistency, has_evidence=True)
    expected = round(0.35 * 80.0 + 0.35 * 70.0 + 0.15 * 60.0 + 0.15 * 90.0 + 0.35)
    assert result == expected


def test_trust_score_all_zero():
    """All zero inputs produce a trust score of 0."""
    result = compute_trust_score(0.0, 0.0, 0.0, 0.0)
    assert result == 0


def test_trust_score_clamped_at_100():
    """Scores above 100 are clamped to 100 (e.g. agreement=200 would overflow without clamping)."""
    result = compute_trust_score(200.0, 200.0, 200.0, 200.0)
    assert result == 100


def test_trust_score_negative_verification_clamped():
    """Negative verification is clamped to 0 before weighting."""
    result = compute_trust_score(100.0, -50.0, 100.0, 100.0, has_evidence=True)
    expected = round(0.35 * 100.0 + 0.35 * 0.0 + 0.15 * 100.0 + 0.15 * 100.0 + 0.35)
    assert result == expected


# ---------------------------------------------------------------------------
# compute_trust_score — no evidence (2-component formula)
# ---------------------------------------------------------------------------


def test_trust_no_evidence_5_of_5():
    """5/5 agree, no evidence: 0.70*100 + 0.30*100 = 100."""
    agreement = compute_agreement_score(5, 5)      # 100
    independence = compute_independence_score(5, 5)  # 100
    result = compute_trust_score(agreement, 0.0, independence, 0.0, has_evidence=False)
    assert result == 100


def test_trust_no_evidence_4_of_5():
    """4/5 agree, no evidence: 0.70*80 + 0.30*80 = 80."""
    agreement = compute_agreement_score(4, 5)      # 80
    independence = compute_independence_score(4, 5)  # 80
    result = compute_trust_score(agreement, 0.0, independence, 0.0, has_evidence=False)
    assert result == 80


def test_trust_no_evidence_3_of_5():
    """3/5 agree, no evidence: 0.70*60 + 0.30*60 = 60."""
    agreement = compute_agreement_score(3, 5)      # 60
    independence = compute_independence_score(3, 5)  # 60
    result = compute_trust_score(agreement, 0.0, independence, 0.0, has_evidence=False)
    assert result == 60


def test_trust_no_evidence_2_of_5():
    """2/5 agree, no evidence: 0.70*40 + 0.30*40 = 40."""
    agreement = compute_agreement_score(2, 5)      # 40
    independence = compute_independence_score(2, 5)  # 40
    result = compute_trust_score(agreement, 0.0, independence, 0.0, has_evidence=False)
    assert result == 40


def test_trust_no_evidence_1_of_5():
    """1/5 agree, no evidence: 0.70*20 + 0.30*20 = 20."""
    agreement = compute_agreement_score(1, 5)      # 20
    independence = compute_independence_score(1, 5)  # 20
    result = compute_trust_score(agreement, 0.0, independence, 0.0, has_evidence=False)
    assert result == 20


def test_trust_has_evidence_defaults_false():
    """has_evidence defaults to False, using 2-component formula."""
    agreement = compute_agreement_score(4, 5)      # 80
    independence = compute_independence_score(4, 5)  # 80
    result = compute_trust_score(agreement, 0.0, independence, 0.0)
    expected = round(0.70 * 80.0 + 0.30 * 80.0 + 0.35)
    assert result == expected


# ---------------------------------------------------------------------------
# determine_verdict
# ---------------------------------------------------------------------------


def test_verdict_safe():
    """SAFE when score >= safe_min and contradiction <= 0.2."""
    verdict = determine_verdict(80, 0.1, safe_min=70, caution_min=40)
    assert verdict == "SAFE"


def test_verdict_caution():
    """CAUTION when score >= caution_min but below safe threshold."""
    verdict = determine_verdict(50, 0.5, safe_min=70, caution_min=40)
    assert verdict == "CAUTION"


def test_verdict_reject():
    """REJECT when score is below caution_min."""
    verdict = determine_verdict(30, 0.3, safe_min=70, caution_min=40)
    assert verdict == "REJECT"


def test_verdict_caution_high_score_high_contradiction():
    """CAUTION even with high trust score if contradiction > 0.2."""
    verdict = determine_verdict(90, 0.5, safe_min=70, caution_min=40)
    assert verdict == "CAUTION"


# ---------------------------------------------------------------------------
# find_supporting_models
# ---------------------------------------------------------------------------


def _make_claim(model_id: str) -> types.SimpleNamespace:
    """Create a minimal claim object with a model_id attribute."""
    return types.SimpleNamespace(model_id=model_id)


def test_supporting_models_multiple():
    """Multiple unique model_ids are collected."""
    claims = {
        "c1": _make_claim("gpt-4"),
        "c2": _make_claim("claude"),
        "c3": _make_claim("gpt-4"),  # duplicate model
    }
    result = find_supporting_models(["c1", "c2", "c3"], claims)
    assert result == ["gpt-4", "claude"]


def test_supporting_models_single():
    """Single claim yields a single model."""
    claims = {"c1": _make_claim("gemini")}
    result = find_supporting_models(["c1"], claims)
    assert result == ["gemini"]


def test_supporting_models_unknown_claim_skipped():
    """Unknown claim_id is silently skipped."""
    claims = {"c1": _make_claim("gpt-4")}
    result = find_supporting_models(["c1", "c_unknown"], claims)
    assert result == ["gpt-4"]


# ---------------------------------------------------------------------------
# find_best_nli_for_cluster
# ---------------------------------------------------------------------------


def _make_nli(claim_id: str, passage_id: str, ent: float, contra: float):
    """Create a minimal NLI result with probs dict."""
    return types.SimpleNamespace(
        claim_id=claim_id,
        passage_id=passage_id,
        probs={"entailment": ent, "contradiction": contra},
    )


def test_best_nli_multiple_results():
    """Best entailment and worst contradiction are found across results."""
    nli_results = [
        _make_nli("c1", "p1", 0.6, 0.1),
        _make_nli("c1", "p2", 0.9, 0.3),
        _make_nli("c2", "p3", 0.7, 0.5),
    ]
    ent, contra, pid = find_best_nli_for_cluster(["c1", "c2"], nli_results)
    assert ent == 0.9
    assert contra == 0.5
    assert pid == "p2"


def test_best_nli_no_results():
    """Empty nli_results returns zeros and empty passage id."""
    ent, contra, pid = find_best_nli_for_cluster(["c1"], [])
    assert ent == 0.0
    assert contra == 0.0
    assert pid == ""


def test_best_nli_claim_not_in_cluster_skipped():
    """NLI results for claims outside the cluster are ignored."""
    nli_results = [
        _make_nli("c_other", "p1", 0.95, 0.8),
        _make_nli("c1", "p2", 0.5, 0.1),
    ]
    ent, contra, pid = find_best_nli_for_cluster(["c1"], nli_results)
    assert ent == 0.5
    assert contra == 0.1
    assert pid == "p2"
