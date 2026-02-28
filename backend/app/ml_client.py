import asyncio
import os
from typing import Any, Callable

import httpx


SCHEMA_VERSION = "1.0"

ENDPOINT_SUFFIXES = {
    "extract_claims": "http-extract-claims",
    "embed_claims": "http-embed-claims",
    "cluster_claims": "http-cluster-claims",
    "rerank_evidence_batch": "http-rerank-evidence-batch",
    "nli_verify_batch": "http-nli-verify-batch",
    "compute_umap": "http-compute-umap",
    "score_clusters": "http-score-clusters",
}


def _endpoint_url(suffix: str) -> str:
    # Example prefix: https://vicxiya24--truthlens-ml
    prefix = os.getenv("ML_MODAL_ENDPOINT_PREFIX", "").strip().rstrip("/")
    if not prefix:
        raise RuntimeError("ML_MODAL_ENDPOINT_PREFIX is not set")
    return f"{prefix}-{suffix}.modal.run"


def _auth_headers() -> dict[str, str]:
    token = os.getenv("ML_MODAL_API_KEY", "").strip()
    if not token:
        raise RuntimeError("ML_MODAL_API_KEY is not set")
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }


def _claim_metadata_map(claims: list[dict[str, Any]]) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for claim in claims:
        cid = str(claim.get("claim_id", ""))
        if not cid:
            continue
        out[cid] = {
            "model_id": str(claim.get("model_id", "")),
            "claim_text": str(claim.get("claim_text", "")),
        }
    return out


def _normalize_passages(
    passages: list[dict[str, Any]] | None,
    claim_id: str,
) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    for idx, p in enumerate(passages or []):
        pid = str(p.get("passage_id", "")).strip() or f"{claim_id}::p{idx}"
        text = str(p.get("text", ""))
        if not text.strip():
            text = "[no evidence provided]"
        normalized.append({"passage_id": pid, "text": text})
    if not normalized:
        normalized = [
            {"passage_id": f"{claim_id}::p0", "text": "[no evidence provided]"}
        ]
    return normalized


def _build_pairs(
    claims: list[dict[str, Any]],
    rankings: list[dict[str, Any]],
    rerank_items: list[dict[str, Any]],
) -> list[dict[str, str]]:
    claim_text_by_id: dict[str, str] = {}
    for claim in claims:
        cid = str(claim.get("claim_id", ""))
        if not cid:
            continue
        claim_text_by_id[cid] = str(claim.get("claim_text", ""))

    # passage text lookup by claim_id/passage_id from rerank request items
    passage_text_by_claim: dict[str, dict[str, str]] = {}
    for item in rerank_items:
        cid = str(item.get("claim_id", ""))
        if not cid:
            continue
        claim_passages = passage_text_by_claim.setdefault(cid, {})
        for passage in item.get("passages", []) or []:
            pid = str(passage.get("passage_id", ""))
            if not pid:
                continue
            txt = str(passage.get("text", ""))
            claim_passages[pid] = txt if txt else " "

    pairs: list[dict[str, str]] = []
    for r in rankings:
        cid = str(r.get("claim_id", ""))
        if not cid:
            continue
        ordered = r.get("ordered_passage_ids", []) or []
        for pid_raw in ordered:
            pid = str(pid_raw)
            pairs.append(
                {
                    "pair_id": f"{cid}::{pid}",
                    "claim_id": cid,
                    "passage_id": pid,
                    "claim_text": claim_text_by_id.get(cid, ""),
                    "passage_text": passage_text_by_claim.get(cid, {}).get(pid, " "),
                }
            )
    return pairs


async def _post(name: str, payload: dict[str, Any]) -> dict[str, Any]:
    suffix = ENDPOINT_SUFFIXES[name]
    url = _endpoint_url(suffix)
    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(url, headers=_auth_headers(), json=payload)
        resp.raise_for_status()
        return resp.json()


async def run_ml_pipeline(
    analysis_id: str,
    model_outputs: list[dict[str, str]],
    passages_by_claim: dict[str, list[dict[str, str]]] | None = None,
    retrieve_fn: Callable | None = None,
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
    if retrieve_fn is not None:
        passages_by_claim = await retrieve_fn(claims)
    else:
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
