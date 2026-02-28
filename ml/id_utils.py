"""Deterministic SHA1 ID generation for TruthLens entities."""

import hashlib


def make_claim_id(analysis_id: str, model_id: str, claim_text: str) -> str:
    """Generate deterministic claim ID: 'c_' + SHA1 hex digest."""
    raw = f"{analysis_id}:{model_id}:{claim_text}"
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()
    return f"c_{digest}"


def make_cluster_id(claim_ids: list) -> str:
    """Generate deterministic cluster ID: 'cl_' + SHA1 of sorted claim IDs."""
    sorted_ids = sorted(claim_ids)
    raw = "|".join(sorted_ids)
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()
    return f"cl_{digest}"


def make_pair_id(claim_id: str, passage_id: str) -> str:
    """Generate deterministic NLI pair ID: 'nli_' + SHA1."""
    raw = f"{claim_id}:{passage_id}"
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()
    return f"nli_{digest}"
