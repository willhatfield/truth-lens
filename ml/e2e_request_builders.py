"""Request builders for E2E pipeline chaining.

Each builder accepts a ``responses`` dict (prior phase responses keyed by
phase name) and returns a request payload.  If upstream dependencies are
missing the builder falls back to mock data so that phases can still run
independently.

Design rules (per CLAUDE.md):
- No function pointers: ``build_request`` dispatches via if/elif.
- All loops bounded by explicit MAX constants.
- Single-level pointer dereference; no recursion.
"""

from mock_data import (
    ANALYSIS_ID,
    EVIDENCE_PASSAGES,
    build_extract_claims_request,
    build_embed_claims_request,
    build_cluster_claims_request,
    build_rerank_request,
    build_nli_request,
    build_umap_request,
    build_score_request,
)

# Upper-bound constants for every loop
MAX_CLAIMS = 200
MAX_VECTORS = 200
MAX_RANKINGS = 200
MAX_PAIRS = 200
MAX_CLUSTERS = 50
MAX_NLI = 200

# Valid builder keys (no function pointers — dispatch via if/elif)
VALID_BUILDER_KEYS = frozenset([
    "extract",
    "embed",
    "cluster",
    "rerank",
    "nli",
    "umap",
    "score",
])

# Upstream dependencies per builder key
CHAIN_DEPS = {
    "extract": [],
    "embed": ["extract_claims"],
    "cluster": ["embed_claims", "extract_claims"],
    "rerank": ["extract_claims"],
    "nli": ["extract_claims", "rerank_evidence_batch"],
    "umap": ["embed_claims"],
    "score": ["cluster_claims", "extract_claims", "nli_verify_batch"],
}


def _has_deps(builder_key: str, responses: dict) -> bool:
    """Return True if all upstream dependencies exist in *responses*."""
    deps = CHAIN_DEPS.get(builder_key, [])
    for i in range(len(deps)):
        if deps[i] not in responses:
            return False
    return True


def _analysis_id_from(responses: dict, dep_name: str) -> str:
    """Extract analysis_id from a prior response, or fall back to mock."""
    resp = responses.get(dep_name, {})
    return resp.get("analysis_id", ANALYSIS_ID)


# ── Individual builders ──────────────────────────────────────────────

def _build_extract(responses: dict) -> dict:
    """Build ExtractClaimsRequest.  No upstream deps — always mock."""
    return build_extract_claims_request()


def _build_embed(responses: dict) -> dict:
    """Build EmbedClaimsRequest from extract_claims response."""
    if not _has_deps("embed", responses):
        return build_embed_claims_request()

    extract_resp = responses["extract_claims"]
    claims_list = extract_resp.get("claims", [])
    analysis_id = extract_resp.get("analysis_id", ANALYSIS_ID)

    claim_inputs = []
    for i in range(len(claims_list)):
        if i >= MAX_CLAIMS:
            break
        c = claims_list[i]
        claim_inputs.append({
            "claim_id": c["claim_id"],
            "claim_text": c["claim_text"],
        })
    return {
        "analysis_id": analysis_id,
        "claims": claim_inputs,
    }


def _build_cluster(responses: dict) -> dict:
    """Build ClusterClaimsRequest from embed + extract responses."""
    if not _has_deps("cluster", responses):
        return build_cluster_claims_request()

    embed_resp = responses["embed_claims"]
    extract_resp = responses["extract_claims"]
    analysis_id = embed_resp.get("analysis_id", ANALYSIS_ID)

    vectors = embed_resp.get("vectors", {})
    claims_list = extract_resp.get("claims", [])

    claims_meta = {}
    for i in range(len(claims_list)):
        if i >= MAX_CLAIMS:
            break
        c = claims_list[i]
        claims_meta[c["claim_id"]] = {
            "model_id": c["model_id"],
            "claim_text": c["claim_text"],
        }
    return {
        "analysis_id": analysis_id,
        "vectors": vectors,
        "claims": claims_meta,
    }


def _build_rerank(responses: dict) -> dict:
    """Build RerankEvidenceBatchRequest from extract response."""
    if not _has_deps("rerank", responses):
        return build_rerank_request()

    extract_resp = responses["extract_claims"]
    claims_list = extract_resp.get("claims", [])
    analysis_id = extract_resp.get("analysis_id", ANALYSIS_ID)

    items = []
    for i in range(len(claims_list)):
        if i >= MAX_CLAIMS:
            break
        c = claims_list[i]
        items.append({
            "claim_id": c["claim_id"],
            "claim_text": c["claim_text"],
            "passages": EVIDENCE_PASSAGES,
        })
    return {
        "analysis_id": analysis_id,
        "items": items,
    }


def _build_nli(responses: dict) -> dict:
    """Build NliVerifyBatchRequest from extract + rerank responses."""
    if not _has_deps("nli", responses):
        return build_nli_request()

    extract_resp = responses["extract_claims"]
    rerank_resp = responses["rerank_evidence_batch"]
    claims_list = extract_resp.get("claims", [])
    rankings = rerank_resp.get("rankings", [])
    analysis_id = extract_resp.get("analysis_id", ANALYSIS_ID)

    # Build lookup: claim_id -> top passage_id
    top_passage_for = {}
    for i in range(len(rankings)):
        if i >= MAX_RANKINGS:
            break
        r = rankings[i]
        ordered = r.get("ordered_passage_ids", [])
        if len(ordered) > 0:
            top_passage_for[r["claim_id"]] = ordered[0]

    # Build lookup: passage_id -> passage_text
    passage_text_for = {}
    for i in range(len(EVIDENCE_PASSAGES)):
        p = EVIDENCE_PASSAGES[i]
        passage_text_for[p["passage_id"]] = p["text"]

    from id_utils import make_pair_id

    pairs = []
    for i in range(len(claims_list)):
        if i >= MAX_CLAIMS:
            break
        c = claims_list[i]
        cid = c["claim_id"]
        pid = top_passage_for.get(cid, "")
        if len(pid) == 0:
            continue
        p_text = passage_text_for.get(pid, "")
        pair_id = make_pair_id(cid, pid)
        pairs.append({
            "pair_id": pair_id,
            "claim_id": cid,
            "passage_id": pid,
            "claim_text": c["claim_text"],
            "passage_text": p_text,
        })
    return {
        "analysis_id": analysis_id,
        "pairs": pairs,
    }


def _build_umap(responses: dict) -> dict:
    """Build ComputeUmapRequest from embed response."""
    if not _has_deps("umap", responses):
        return build_umap_request()

    embed_resp = responses["embed_claims"]
    vectors = embed_resp.get("vectors", {})
    analysis_id = embed_resp.get("analysis_id", ANALYSIS_ID)

    n_vectors = len(vectors)
    n_neighbors = min(n_vectors - 1, 15) if n_vectors > 1 else 1

    return {
        "analysis_id": analysis_id,
        "vectors": vectors,
        "n_neighbors": n_neighbors,
    }


def _build_score(responses: dict) -> dict:
    """Build ScoreClustersRequest from cluster + extract + nli responses."""
    if not _has_deps("score", responses):
        return build_score_request()

    cluster_resp = responses["cluster_claims"]
    extract_resp = responses["extract_claims"]
    nli_resp = responses["nli_verify_batch"]
    analysis_id = cluster_resp.get("analysis_id", ANALYSIS_ID)

    clusters = cluster_resp.get("clusters", [])
    claims_list = extract_resp.get("claims", [])
    nli_results = nli_resp.get("results", [])

    claims_meta = {}
    for i in range(len(claims_list)):
        if i >= MAX_CLAIMS:
            break
        c = claims_list[i]
        claims_meta[c["claim_id"]] = {
            "model_id": c["model_id"],
            "claim_text": c["claim_text"],
        }
    return {
        "analysis_id": analysis_id,
        "clusters": clusters,
        "claims": claims_meta,
        "nli_results": nli_results,
    }


# ── Dispatch (no function pointers) ─────────────────────────────────

def build_request(phase_key: str, responses: dict) -> dict:
    """Dispatch to the correct builder by string key.

    Uses if/elif chain (no function pointers) per CLAUDE.md rules.
    """
    if phase_key == "extract":
        return _build_extract(responses)
    elif phase_key == "embed":
        return _build_embed(responses)
    elif phase_key == "cluster":
        return _build_cluster(responses)
    elif phase_key == "rerank":
        return _build_rerank(responses)
    elif phase_key == "nli":
        return _build_nli(responses)
    elif phase_key == "umap":
        return _build_umap(responses)
    elif phase_key == "score":
        return _build_score(responses)
    else:
        raise ValueError(f"Unknown builder key: {phase_key}")
