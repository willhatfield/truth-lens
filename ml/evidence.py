"""Evidence retrieval stub for the TruthLens pipeline.

Returns empty evidence lists per claim. This module provides a clean
interface for future web-search-based evidence retrieval.
"""

_MAX_CLAIMS = 50_000


def retrieve_evidence(claims, analysis_id):
    """Retrieve evidence passages for a list of claims.

    Currently returns empty evidence lists. Future versions will
    perform web searches to find supporting/contradicting passages.

    Args:
        claims: list of dicts, each with "claim_id" and "claim_text"
        analysis_id: str, the analysis identifier

    Returns:
        dict mapping claim_id (str) to list of passage dicts.
        Each passage dict has: {"passage_id": str, "text": str, "source": str}
        Currently returns empty lists for all claims.
    """
    result = {}
    for idx in range(_MAX_CLAIMS):
        if idx >= len(claims):
            break
        claim = claims[idx]
        claim_id = claim.get("claim_id", "")
        if claim_id:
            result[claim_id] = []
    return result
