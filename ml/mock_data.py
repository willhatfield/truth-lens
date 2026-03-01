"""Mock data for TruthLens ML pipeline E2E testing.

Simulates 5 LLMs answering "Is the Earth round or flat?"
- 4 models give correct answers (Earth is round/spherical)
- 1 model gives incorrect answer (flat Earth)

Each build_*_request/response pair returns a schema-valid dict.
build_full_pipeline_data() orchestrates all phases.
"""

from id_utils import make_claim_id, make_cluster_id, make_pair_id

ANALYSIS_ID = "mock-earth-shape-001"
QUESTION = "Is the Earth round or flat?"

# 5 model IDs
MODEL_IDS = ["gpt-4o", "claude-3", "gemini-pro", "llama-3", "flat-bot"]

# Responses from each model
MODEL_RESPONSES = {
    "gpt-4o": (
        "The Earth is an oblate spheroid. Scientific evidence from "
        "satellite imagery and gravitational measurements confirms "
        "the Earth is roughly spherical."
    ),
    "claude-3": (
        "Earth is round. Multiple lines of evidence support this, "
        "including ship disappearance over the horizon and lunar "
        "eclipse shadows."
    ),
    "gemini-pro": (
        "The Earth is spherical in shape. This has been confirmed "
        "by space exploration and centuries of scientific observation."
    ),
    "llama-3": (
        "Earth is approximately spherical. Gravity pulls matter into "
        "a sphere, and photos from space clearly show a round planet."
    ),
    "flat-bot": (
        "The Earth is flat. The horizon always appears flat, and "
        "water always finds its level. Globe theory is unproven."
    ),
}

# Pre-defined claims per model (simulating extraction output)
CLAIM_TEXTS = {
    "gpt-4o": [
        "The Earth is an oblate spheroid",
        "Satellite imagery confirms the Earth is spherical",
    ],
    "claude-3": [
        "Earth is round",
        "Ships disappear over the horizon bottom-first",
    ],
    "gemini-pro": [
        "The Earth is spherical in shape",
        "Space exploration confirmed Earth is spherical",
    ],
    "llama-3": [
        "Earth is approximately spherical",
        "Photos from space show a round planet",
    ],
    "flat-bot": [
        "The Earth is flat",
        "Water always finds its level",
    ],
}

# Evidence passages for reranking
EVIDENCE_PASSAGES = [
    {
        "passage_id": "wiki-earth-shape-001",
        "text": (
            "Earth is the third planet from the Sun. It is an oblate "
            "spheroid with a mean radius of 6,371 km."
        ),
    },
    {
        "passage_id": "nasa-blue-marble-002",
        "text": (
            "NASA's Blue Marble photographs taken from space clearly "
            "show Earth as a sphere."
        ),
    },
    {
        "passage_id": "flat-earth-faq-003",
        "text": (
            "Flat Earth proponents argue the horizon appears flat, "
            "but this is explained by Earth's large radius."
        ),
    },
]

EMBED_DIM = 8  # Small dimension for mock vectors


def _make_mock_vector(seed_val: int) -> list:
    """Generate a deterministic mock embedding vector.

    Uses simple math to create vectors where similar claims
    have similar vectors (correct models cluster together).
    """
    base = []
    for i in range(EMBED_DIM):
        val = ((seed_val * 17 + i * 31) % 100) / 100.0
        base.append(round(val, 4))
    return base


def build_extract_claims_request() -> dict:
    """Build ExtractClaimsRequest dict with 5 model responses."""
    responses = []
    for i in range(len(MODEL_IDS)):
        mid = MODEL_IDS[i]
        responses.append({
            "model_id": mid,
            "response_text": MODEL_RESPONSES[mid],
        })
    return {
        "analysis_id": ANALYSIS_ID,
        "responses": responses,
    }


def build_extract_claims_response() -> dict:
    """Build ExtractClaimsResponse dict with pre-defined claims."""
    claims = []
    for i in range(len(MODEL_IDS)):
        mid = MODEL_IDS[i]
        texts = CLAIM_TEXTS[mid]
        for j in range(len(texts)):
            cid = make_claim_id(ANALYSIS_ID, mid, texts[j])
            claims.append({
                "claim_id": cid,
                "model_id": mid,
                "claim_text": texts[j],
            })
    return {
        "analysis_id": ANALYSIS_ID,
        "claims": claims,
        "warnings": [],
    }


def _get_all_claims() -> list:
    """Return flat list of (claim_id, model_id, claim_text) tuples."""
    result = []
    for i in range(len(MODEL_IDS)):
        mid = MODEL_IDS[i]
        texts = CLAIM_TEXTS[mid]
        for j in range(len(texts)):
            cid = make_claim_id(ANALYSIS_ID, mid, texts[j])
            result.append((cid, mid, texts[j]))
    return result


def build_embed_claims_request() -> dict:
    """Build EmbedClaimsRequest from extracted claims."""
    all_claims = _get_all_claims()
    claim_inputs = []
    for i in range(len(all_claims)):
        cid, mid, text = all_claims[i]
        claim_inputs.append({
            "claim_id": cid,
            "claim_text": text,
        })
    return {
        "analysis_id": ANALYSIS_ID,
        "claims": claim_inputs,
    }


def build_embed_claims_response() -> dict:
    """Build EmbedClaimsResponse with mock vectors.

    Correct-model claims get similar vectors; flat-bot gets different ones.
    """
    all_claims = _get_all_claims()
    vectors = {}
    for i in range(len(all_claims)):
        cid, mid, text = all_claims[i]
        if mid == "flat-bot":
            seed = 900 + i
        else:
            seed = 100 + i
        vectors[cid] = _make_mock_vector(seed)
    return {
        "analysis_id": ANALYSIS_ID,
        "vectors": vectors,
        "dim": EMBED_DIM,
        "warnings": [],
    }


def build_cluster_claims_request() -> dict:
    """Build ClusterClaimsRequest from embeddings."""
    embed_resp = build_embed_claims_response()
    all_claims = _get_all_claims()
    claims_meta = {}
    for i in range(len(all_claims)):
        cid, mid, text = all_claims[i]
        claims_meta[cid] = {
            "model_id": mid,
            "claim_text": text,
        }
    return {
        "analysis_id": ANALYSIS_ID,
        "vectors": embed_resp["vectors"],
        "claims": claims_meta,
    }


def build_cluster_claims_response() -> dict:
    """Build ClusterClaimsResponse with 2 clusters.

    Cluster 1: 8 claims from correct models (round Earth)
    Cluster 2: 2 claims from flat-bot (flat Earth)
    """
    all_claims = _get_all_claims()
    correct_ids = []
    flat_ids = []
    for i in range(len(all_claims)):
        cid, mid, text = all_claims[i]
        if mid == "flat-bot":
            flat_ids.append(cid)
        else:
            correct_ids.append(cid)

    correct_rep = correct_ids[0] if len(correct_ids) > 0 else ""
    flat_rep = flat_ids[0] if len(flat_ids) > 0 else ""

    correct_text = ""
    flat_text = ""
    for i in range(len(all_claims)):
        cid, mid, text = all_claims[i]
        if cid == correct_rep:
            correct_text = text
        if cid == flat_rep:
            flat_text = text

    clusters = [
        {
            "cluster_id": make_cluster_id(correct_ids),
            "claim_ids": correct_ids,
            "representative_claim_id": correct_rep,
            "representative_text": correct_text,
        },
        {
            "cluster_id": make_cluster_id(flat_ids),
            "claim_ids": flat_ids,
            "representative_claim_id": flat_rep,
            "representative_text": flat_text,
        },
    ]
    return {
        "analysis_id": ANALYSIS_ID,
        "clusters": clusters,
        "warnings": [],
    }


def build_rerank_request() -> dict:
    """Build RerankEvidenceBatchRequest for all claims."""
    all_claims = _get_all_claims()
    items = []
    for i in range(len(all_claims)):
        cid, mid, text = all_claims[i]
        items.append({
            "claim_id": cid,
            "claim_text": text,
            "passages": EVIDENCE_PASSAGES,
        })
    return {
        "analysis_id": ANALYSIS_ID,
        "items": items,
    }


def build_rerank_response() -> dict:
    """Build RerankEvidenceBatchResponse with mock rankings."""
    all_claims = _get_all_claims()
    rankings = []
    for i in range(len(all_claims)):
        cid, mid, text = all_claims[i]
        ordered_ids = []
        scores = {}
        for j in range(len(EVIDENCE_PASSAGES)):
            pid = EVIDENCE_PASSAGES[j]["passage_id"]
            ordered_ids.append(pid)
            score = round(0.9 - j * 0.2, 4)
            scores[pid] = score
        rankings.append({
            "claim_id": cid,
            "ordered_passage_ids": ordered_ids,
            "scores": scores,
        })
    return {
        "analysis_id": ANALYSIS_ID,
        "rankings": rankings,
        "warnings": [],
    }


def build_nli_request() -> dict:
    """Build NliVerifyBatchRequest for top-ranked passages."""
    all_claims = _get_all_claims()
    pairs = []
    for i in range(len(all_claims)):
        cid, mid, text = all_claims[i]
        top_passage = EVIDENCE_PASSAGES[0]
        pid = top_passage["passage_id"]
        pair_id = make_pair_id(cid, pid)
        pairs.append({
            "pair_id": pair_id,
            "claim_id": cid,
            "passage_id": pid,
            "claim_text": text,
            "passage_text": top_passage["text"],
        })
    return {
        "analysis_id": ANALYSIS_ID,
        "pairs": pairs,
    }


def build_nli_response() -> dict:
    """Build NliVerifyBatchResponse with realistic NLI verdicts.

    Correct claims get high entailment; flat-bot claims get contradiction.
    """
    all_claims = _get_all_claims()
    results = []
    for i in range(len(all_claims)):
        cid, mid, text = all_claims[i]
        top_passage = EVIDENCE_PASSAGES[0]
        pid = top_passage["passage_id"]
        pair_id = make_pair_id(cid, pid)

        if mid == "flat-bot":
            probs = {
                "entailment": 0.05,
                "neutral": 0.15,
                "contradiction": 0.80,
            }
            label = "contradiction"
        else:
            probs = {
                "entailment": 0.85,
                "neutral": 0.10,
                "contradiction": 0.05,
            }
            label = "entailment"

        results.append({
            "pair_id": pair_id,
            "claim_id": cid,
            "passage_id": pid,
            "label": label,
            "probs": probs,
        })
    return {
        "analysis_id": ANALYSIS_ID,
        "results": results,
        "warnings": [],
    }


def build_umap_request() -> dict:
    """Build ComputeUmapRequest from embedding vectors."""
    embed_resp = build_embed_claims_response()
    return {
        "analysis_id": ANALYSIS_ID,
        "vectors": embed_resp["vectors"],
        "n_neighbors": 3,
    }


def build_umap_response() -> dict:
    """Build ComputeUmapResponse with mock 3D coordinates."""
    embed_resp = build_embed_claims_response()
    coords3d = {}
    idx = 0
    for cid in embed_resp["vectors"]:
        coords3d[cid] = [
            round(float(idx) * 0.1, 2),
            round(float(idx) * 0.2, 2),
            round(float(idx) * 0.05, 2),
        ]
        idx += 1
    return {
        "analysis_id": ANALYSIS_ID,
        "coords3d": coords3d,
        "warnings": [],
    }


def build_score_request() -> dict:
    """Build ScoreClustersRequest from clusters and NLI results."""
    cluster_resp = build_cluster_claims_response()
    nli_resp = build_nli_response()
    all_claims = _get_all_claims()

    claims_meta = {}
    for i in range(len(all_claims)):
        cid, mid, text = all_claims[i]
        claims_meta[cid] = {
            "model_id": mid,
            "claim_text": text,
        }

    return {
        "analysis_id": ANALYSIS_ID,
        "clusters": cluster_resp["clusters"],
        "claims": claims_meta,
        "nli_results": nli_resp["results"],
    }


def build_score_response() -> dict:
    """Build ScoreClustersResponse with expected verdicts.

    Cluster 1 (correct, 4 models): SAFE
    Cluster 2 (flat-bot, 1 model): REJECT
    """
    from scoring import (
        compute_agreement_score,
        compute_consistency_score,
        compute_independence_score,
        compute_verification_score,
        compute_trust_score,
        determine_verdict,
    )

    cluster_resp = build_cluster_claims_response()
    clusters = cluster_resp["clusters"]

    # Cluster 1: correct models (4 of 5 unique models)
    agreement_1 = compute_agreement_score(4, 5)
    verification_1 = compute_verification_score(0.85, 0.05)
    independence_1 = compute_independence_score(4, 5)
    consistency_1 = compute_consistency_score(0.05)
    trust_1 = compute_trust_score(agreement_1, verification_1, independence_1, consistency_1, has_evidence=True)
    verdict_1 = determine_verdict(trust_1, 0.05, 75, 45)

    # Cluster 2: flat-bot only (1 of 5 unique models)
    agreement_2 = compute_agreement_score(1, 5)
    verification_2 = compute_verification_score(0.05, 0.80)
    independence_2 = compute_independence_score(1, 5)
    consistency_2 = compute_consistency_score(0.80)
    trust_2 = compute_trust_score(agreement_2, verification_2, independence_2, consistency_2, has_evidence=True)
    verdict_2 = determine_verdict(trust_2, 0.80, 75, 45)

    scores = [
        {
            "cluster_id": clusters[0]["cluster_id"],
            "trust_score": trust_1,
            "verdict": verdict_1,
            "agreement": {
                "models_supporting": ["gpt-4o", "claude-3",
                                      "gemini-pro", "llama-3"],
                "count": 4,
            },
            "verification": {
                "best_entailment_prob": 0.85,
                "best_contradiction_prob": 0.05,
                "evidence_passage_id": "wiki-earth-shape-001",
            },
        },
        {
            "cluster_id": clusters[1]["cluster_id"],
            "trust_score": trust_2,
            "verdict": verdict_2,
            "agreement": {
                "models_supporting": ["flat-bot"],
                "count": 1,
            },
            "verification": {
                "best_entailment_prob": 0.05,
                "best_contradiction_prob": 0.80,
                "evidence_passage_id": "wiki-earth-shape-001",
            },
        },
    ]
    return {
        "analysis_id": ANALYSIS_ID,
        "scores": scores,
        "warnings": [],
    }


def build_full_pipeline_data() -> dict:
    """Build all 14 request/response payloads for the full pipeline.

    Returns a dict with 14 keys:
      extract_request, extract_response,
      embed_request, embed_response,
      cluster_request, cluster_response,
      rerank_request, rerank_response,
      nli_request, nli_response,
      umap_request, umap_response,
      score_request, score_response
    """
    return {
        "extract_request": build_extract_claims_request(),
        "extract_response": build_extract_claims_response(),
        "embed_request": build_embed_claims_request(),
        "embed_response": build_embed_claims_response(),
        "cluster_request": build_cluster_claims_request(),
        "cluster_response": build_cluster_claims_response(),
        "rerank_request": build_rerank_request(),
        "rerank_response": build_rerank_response(),
        "nli_request": build_nli_request(),
        "nli_response": build_nli_response(),
        "umap_request": build_umap_request(),
        "umap_response": build_umap_response(),
        "score_request": build_score_request(),
        "score_response": build_score_response(),
    }
