"""Per-model claim-count metrics for the TruthLens pipeline."""

_MAX_CLAIMS = 50_000
_MAX_SCORES = 10_000
_MAX_CLUSTERS = 10_000

_VERDICT_KEY = {
    "SAFE": "supported",
    "CAUTION": "caution",
    "REJECT": "rejected",
}


def _build_claim_to_model(claims):
    """Map claim_id to model_id from the claims list.

    Args:
        claims: list of dicts with "claim_id" and "model_id" keys.

    Returns:
        dict mapping claim_id (str) to model_id (str).
    """
    result = {}
    bound = min(len(claims), _MAX_CLAIMS)
    for i in range(bound):
        claim = claims[i]
        result[claim["claim_id"]] = claim["model_id"]
    return result


def _build_cluster_verdict(cluster_scores):
    """Map cluster_id to verdict from the cluster_scores list.

    Args:
        cluster_scores: list of dicts with "cluster_id" and "verdict" keys.

    Returns:
        dict mapping cluster_id (str) to verdict (str).
    """
    result = {}
    bound = min(len(cluster_scores), _MAX_SCORES)
    for i in range(bound):
        score = cluster_scores[i]
        result[score["cluster_id"]] = score["verdict"]
    return result


def _build_claim_to_verdict(clusters, cluster_verdict):
    """Map each claim_id to its cluster's verdict.

    Args:
        clusters: list of dicts with "cluster_id" and "claim_ids" keys.
        cluster_verdict: dict mapping cluster_id to verdict.

    Returns:
        dict mapping claim_id (str) to verdict (str).
    """
    result = {}
    outer_bound = min(len(clusters), _MAX_CLUSTERS)
    for i in range(outer_bound):
        cluster = clusters[i]
        cid = cluster["cluster_id"]
        verdict = cluster_verdict.get(cid)
        if verdict is None:
            continue
        claim_ids = cluster["claim_ids"]
        inner_bound = min(len(claim_ids), _MAX_CLAIMS)
        for j in range(inner_bound):
            result[claim_ids[j]] = verdict
    return result


def _empty_counts():
    """Return a fresh counter dict for a single model."""
    return {"total": 0, "supported": 0, "caution": 0, "rejected": 0}


def compute_model_metrics(claims, cluster_scores, clusters):
    """Compute per-model claim counts (total, supported, caution, rejected).

    Args:
        claims: list of dicts, each with "claim_id", "model_id", "claim_text".
        cluster_scores: list of dicts, each with "cluster_id", "verdict"
                        (SAFE/CAUTION/REJECT).
        clusters: list of dicts, each with "cluster_id", "claim_ids"
                  (list of str).

    Returns:
        list of dicts sorted by model_id:
        [{"model_id": str,
          "claim_counts": {"total": int, "supported": int,
                           "caution": int, "rejected": int}}]
    """
    if not claims:
        return []

    claim_to_model = _build_claim_to_model(claims)
    cluster_verdict = _build_cluster_verdict(cluster_scores)
    claim_to_verdict = _build_claim_to_verdict(clusters, cluster_verdict)

    model_counts = {}
    bound = min(len(claims), _MAX_CLAIMS)
    for i in range(bound):
        claim = claims[i]
        mid = claim["model_id"]
        if mid not in model_counts:
            model_counts[mid] = _empty_counts()
        model_counts[mid]["total"] += 1

        verdict = claim_to_verdict.get(claim["claim_id"])
        if verdict is None:
            continue
        count_key = _VERDICT_KEY.get(verdict)
        if count_key is None:
            continue
        model_counts[mid][count_key] += 1

    model_ids = sorted(model_counts.keys())
    result = []
    for i in range(len(model_ids)):
        mid = model_ids[i]
        result.append({"model_id": mid, "claim_counts": model_counts[mid]})
    return result
