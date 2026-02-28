import asyncio
import os
from typing import Any

import httpx

from .schemas import EventEnvelope

PublishFn = Callable[[str, dict], Awaitable[None]]


def _ml_service_url() -> str:
    url = os.getenv("ML_SERVICE_URL", "").strip().rstrip("/")
    if not url:
        raise RuntimeError("ML_SERVICE_URL is not set")
    return url


def _ml_auth_headers() -> dict[str, str]:
    key = os.getenv("ML_SERVICE_API_KEY", "").strip()
    if not key:
        raise RuntimeError("ML_SERVICE_API_KEY is not set")
    return {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }


async def run_ml_pipeline(
    analysis_id: str,
    model_outputs: list[dict[str, str]],
    passages_by_claim: dict[str, list[dict[str, str]]] | None = None,
) -> dict[str, Any]:
    filtered_responses = []
    for item in model_outputs:
        text = str(item.get("response_text", "")).strip()
        if not text:
            continue
        filtered_responses.append(
            {
                "model_id": str(item.get("model_id", "")),
                "response_text": text,
            }
        )

    if not filtered_responses:
        return {"warnings": ["ml pipeline skipped: no non-empty model responses"]}

    extract_payload = {
        "schema_version": SCHEMA_VERSION,
        "analysis_id": analysis_id,
        "responses": filtered_responses,
    }
    extract = await _post("extract_claims", extract_payload)
    claims = extract.get("claims", []) or []
    if not claims:
        return {"extract_claims": extract, "warnings": ["no claims extracted"]}

    embed_payload = {
        "schema_version": SCHEMA_VERSION,
        "analysis_id": analysis_id,
        "claims": [
            {
                "claim_id": str(c.get("claim_id", "")),
                "claim_text": str(c.get("claim_text", "")),
            }
            for c in claims
            if c.get("claim_id")
        ],
    }
    embed = await _post("embed_claims", embed_payload)
    vectors = embed.get("vectors", {}) or {}
    if not vectors:
        return {
            "extract_claims": extract,
            "embed_claims": embed,
            "warnings": ["embedding produced no vectors"],
        }

    claims_map = _claim_metadata_map(claims)

    cluster_payload = {
        "schema_version": SCHEMA_VERSION,
        "analysis_id": analysis_id,
        "vectors": vectors,
        "claims": claims_map,
        "sim_threshold": 0.85,
    }
    umap_payload = {
        "schema_version": SCHEMA_VERSION,
        "analysis_id": analysis_id,
        "vectors": vectors,
    }

    cluster_task = _post("cluster_claims", cluster_payload)
    umap_task = _post("compute_umap", umap_payload)
    cluster, umap = await asyncio.gather(cluster_task, umap_task)

    # If retrieval is not wired yet, run with empty evidence lists (valid but less useful).
    passages_by_claim = passages_by_claim or {}
    rerank_items = []
    for c in claims:
        cid = str(c.get("claim_id", ""))
        if not cid:
            continue
        passages = _normalize_passages(passages_by_claim.get(cid, []), cid)
        rerank_items.append(
            {
                "claim_id": cid,
                "claim_text": str(c.get("claim_text", "")),
                "passages": passages,
            }
        )

    rerank_payload = {
        "schema_version": SCHEMA_VERSION,
        "analysis_id": analysis_id,
        "items": rerank_items,
        "top_k": 10,
    }
    rerank = await _post("rerank_evidence_batch", rerank_payload)

    nli_pairs = _build_pairs(
        claims,
        rerank.get("rankings", []) or [],
        rerank_items,
    )
    nli = {"results": [], "warnings": ["nli skipped: no claim/passage pairs"]}
    if nli_pairs:
        nli_payload = {
            "schema_version": SCHEMA_VERSION,
            "analysis_id": analysis_id,
            "pairs": nli_pairs,
            "batch_size": 16,
        }
        nli = await _post("nli_verify_batch", nli_payload)

    score_payload = {
        "schema_version": SCHEMA_VERSION,
        "analysis_id": analysis_id,
        "clusters": cluster.get("clusters", []) or [],
        "claims": claims_map,
        "nli_results": nli.get("results", []) or [],
    }
    score = await _post("score_clusters", score_payload)

    warnings = []
    for block in (extract, embed, cluster, rerank, nli, umap, score):
        warnings.extend(block.get("warnings", []) or [])

    return {
        "extract_claims": extract,
        "embed_claims": embed,
        "cluster_claims": cluster,
        "rerank_evidence_batch": rerank,
        "nli_verify_batch": nli,
        "compute_umap": umap,
        "score_clusters": score,
        "warnings": warnings,
    }
