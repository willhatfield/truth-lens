"""Tests for model_metrics.py per-model claim-count metrics."""

from model_metrics import compute_model_metrics


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _claim(claim_id, model_id, text="some claim"):
    """Build a minimal claim dict."""
    return {"claim_id": claim_id, "model_id": model_id, "claim_text": text}


def _score(cluster_id, verdict):
    """Build a minimal cluster-score dict."""
    return {"cluster_id": cluster_id, "verdict": verdict}


def _cluster(cluster_id, claim_ids):
    """Build a minimal cluster dict."""
    return {"cluster_id": cluster_id, "claim_ids": claim_ids}


# ---------------------------------------------------------------------------
# test_basic_counts
# ---------------------------------------------------------------------------


def test_basic_counts():
    """Two models, 3 claims across 2 clusters with SAFE and REJECT verdicts."""
    claims = [
        _claim("c1", "gpt-4"),
        _claim("c2", "claude"),
        _claim("c3", "gpt-4"),
    ]
    cluster_scores = [
        _score("cl1", "SAFE"),
        _score("cl2", "REJECT"),
    ]
    clusters = [
        _cluster("cl1", ["c1", "c2"]),
        _cluster("cl2", ["c3"]),
    ]

    result = compute_model_metrics(claims, cluster_scores, clusters)

    assert len(result) == 2
    claude_entry = result[0]
    gpt_entry = result[1]

    assert claude_entry["model_id"] == "claude"
    assert claude_entry["claim_counts"]["total"] == 1
    assert claude_entry["claim_counts"]["supported"] == 1
    assert claude_entry["claim_counts"]["caution"] == 0
    assert claude_entry["claim_counts"]["rejected"] == 0

    assert gpt_entry["model_id"] == "gpt-4"
    assert gpt_entry["claim_counts"]["total"] == 2
    assert gpt_entry["claim_counts"]["supported"] == 1
    assert gpt_entry["claim_counts"]["rejected"] == 1


# ---------------------------------------------------------------------------
# test_empty_inputs
# ---------------------------------------------------------------------------


def test_empty_inputs():
    """All empty lists return []."""
    assert compute_model_metrics([], [], []) == []


# ---------------------------------------------------------------------------
# test_single_model_all_verdicts
# ---------------------------------------------------------------------------


def test_single_model_all_verdicts():
    """One model with SAFE, CAUTION, and REJECT claims."""
    claims = [
        _claim("c1", "gemini"),
        _claim("c2", "gemini"),
        _claim("c3", "gemini"),
    ]
    cluster_scores = [
        _score("cl1", "SAFE"),
        _score("cl2", "CAUTION"),
        _score("cl3", "REJECT"),
    ]
    clusters = [
        _cluster("cl1", ["c1"]),
        _cluster("cl2", ["c2"]),
        _cluster("cl3", ["c3"]),
    ]

    result = compute_model_metrics(claims, cluster_scores, clusters)

    assert len(result) == 1
    counts = result[0]["claim_counts"]
    assert counts["total"] == 3
    assert counts["supported"] == 1
    assert counts["caution"] == 1
    assert counts["rejected"] == 1


# ---------------------------------------------------------------------------
# test_claims_not_in_cluster
# ---------------------------------------------------------------------------


def test_claims_not_in_cluster():
    """Claims not in any cluster count toward total but not verdict buckets."""
    claims = [
        _claim("c1", "gpt-4"),
        _claim("c2", "gpt-4"),
    ]
    cluster_scores = [
        _score("cl1", "SAFE"),
    ]
    clusters = [
        _cluster("cl1", ["c1"]),
        # c2 is not in any cluster
    ]

    result = compute_model_metrics(claims, cluster_scores, clusters)

    assert len(result) == 1
    counts = result[0]["claim_counts"]
    assert counts["total"] == 2
    assert counts["supported"] == 1
    assert counts["caution"] == 0
    assert counts["rejected"] == 0


# ---------------------------------------------------------------------------
# test_deterministic_ordering
# ---------------------------------------------------------------------------


def test_deterministic_ordering():
    """Output is sorted alphabetically by model_id."""
    claims = [
        _claim("c1", "zebra-model"),
        _claim("c2", "alpha-model"),
        _claim("c3", "mid-model"),
    ]
    cluster_scores = [_score("cl1", "SAFE")]
    clusters = [_cluster("cl1", ["c1", "c2", "c3"])]

    result = compute_model_metrics(claims, cluster_scores, clusters)

    model_ids = [entry["model_id"] for entry in result]
    assert model_ids == ["alpha-model", "mid-model", "zebra-model"]


# ---------------------------------------------------------------------------
# test_large_input_bounded
# ---------------------------------------------------------------------------


def test_large_input_bounded():
    """Verify function handles many claims without error."""
    num_claims = 5000
    claims = []
    claim_ids = []
    for i in range(num_claims):
        cid = "c" + str(i)
        mid = "model_" + str(i % 10)
        claims.append(_claim(cid, mid))
        claim_ids.append(cid)

    cluster_scores = [_score("cl_big", "SAFE")]
    clusters = [_cluster("cl_big", claim_ids)]

    result = compute_model_metrics(claims, cluster_scores, clusters)

    assert len(result) == 10
    total_supported = 0
    total_count = 0
    for i in range(len(result)):
        total_count += result[i]["claim_counts"]["total"]
        total_supported += result[i]["claim_counts"]["supported"]
    assert total_count == num_claims
    assert total_supported == num_claims
