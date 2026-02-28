"""TruthLens ML pipeline -- Modal serverless functions.

Provides seven functions callable via `fn.remote(dict)`:
  extract_claims, embed_claims, cluster_claims,
  rerank_evidence_batch, nli_verify_batch, compute_umap, score_clusters.

Each function accepts a plain dict, validates via Pydantic inside the
boundary, and returns a plain dict (always including a ``warnings`` field).
"""

import modal

from schemas import (
    ExtractClaimsRequest,
    ExtractClaimsResponse,
    Claim,
    ClaimInput,
    EmbedClaimsRequest,
    EmbedClaimsResponse,
    ClaimMetadata,
    Cluster,
    ClusterClaimsRequest,
    ClusterClaimsResponse,
    RerankItem,
    ClaimRanking,
    RerankEvidenceBatchRequest,
    RerankEvidenceBatchResponse,
    NliPairInput,
    NliResultOutput,
    NliVerifyBatchRequest,
    NliVerifyBatchResponse,
    ComputeUmapRequest,
    ComputeUmapResponse,
    ScoreClustersRequest,
    ScoreClustersResponse,
    ClusterScore,
    AgreementDetail,
    VerificationDetail,
)
from batch_utils import chunk_list
from fallback_utils import build_warning_response

# -- Modal App + shared volume -------------------------------------------

app = modal.App("truthlens-ml")

model_volume = modal.Volume.from_name(
    "truthlens-model-cache", create_if_missing=True
)

VOLUME_MOUNT = "/models"
SHARED_ENV = {
    "HF_HOME": "/models/hf",
    "TRANSFORMERS_CACHE": "/models/hf",
    "SENTENCE_TRANSFORMERS_HOME": "/models/st",
}

# -- Container images -----------------------------------------------------

gpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.5.1",
        "transformers==4.47.1",
        "sentence-transformers==3.3.1",
        "pydantic==2.10.6",
        "numpy==1.26.4",
    )
    .env(SHARED_ENV)
    .add_local_python_source(
        "schemas", "batch_utils", "fallback_utils",
        "id_utils", "claim_extraction", "scoring",
    )
)

cpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "scikit-learn==1.6.1",
        "umap-learn==0.5.7",
        "pydantic==2.10.6",
        "numpy==1.26.4",
    )
    .env(SHARED_ENV)
    .add_local_python_source(
        "schemas", "batch_utils", "fallback_utils",
        "id_utils", "claim_extraction", "scoring",
    )
)

# -- Model name constants -------------------------------------------------

EMBED_MODEL_NAME = "BAAI/bge-large-en-v1.5"
RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
NLI_MODEL_NAME = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli"
LLAMA_MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"

MAX_RESPONSES = 10
MAX_ITEMS = 100
MAX_PAIRS = 5000
MAX_VECTORS = 10000
MAX_CLUSTERS = 1000

# -- Helpers (module-level, no recursion) ----------------------------------


def _softmax(logits: list) -> list:
    """Numerically-stable softmax over a flat list of floats."""
    max_val = logits[0]
    for i in range(1, len(logits)):
        if logits[i] > max_val:
            max_val = logits[i]

    exp_vals: list = []
    for i in range(len(logits)):
        exp_vals.append(0.0)

    total = 0.0
    for i in range(len(logits)):
        val = logits[i] - max_val
        # Clamp to avoid overflow
        if val < -500.0:
            val = -500.0
        import math
        e = math.exp(val)
        exp_vals[i] = e
        total += e

    result: list = []
    for i in range(len(exp_vals)):
        result.append(exp_vals[i] / total)

    return result


# -- Extraction helper (keeps extract_claims under 60 lines) ---------------


def _texts_to_claims(
    texts: list, analysis_id: str, model_id: str,
) -> list:
    """Convert a list of claim text strings into Claim objects."""
    from id_utils import make_claim_id

    claims: list = []
    for idx in range(len(texts)):
        if idx >= MAX_RESPONSES * 50:
            break
        text = str(texts[idx]).strip()
        if len(text) == 0:
            continue
        cid = make_claim_id(analysis_id, model_id, text)
        claims.append(Claim(
            claim_id=cid, model_id=model_id, claim_text=text,
        ))
    return claims


def _extract_from_single_response(
    model_response,
    analysis_id: str,
    tokenizer,
    model,
    device: str,
) -> tuple:
    """Extract claims from one ModelResponse using Llama.

    Returns (claims_list, warning_or_none).
    """
    from claim_extraction import sentence_split_claims
    import json

    system_prompt = (
        "You are a claim extractor. Given a text, extract all atomic "
        "factual claims as a JSON array of strings. Each claim should be "
        "a single, verifiable statement. Return ONLY the JSON array, "
        "no other text."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": model_response.response_text},
    ]

    try:
        input_ids = tokenizer.apply_chat_template(
            messages, return_tensors="pt",
        )
        input_ids = input_ids.to(device)
        output_ids = model.generate(
            input_ids, max_new_tokens=1024, do_sample=False,
        )
        generated = output_ids[0][input_ids.shape[1]:]
        raw_text = tokenizer.decode(generated, skip_special_tokens=True)
        claim_texts = json.loads(raw_text)
        claims = _texts_to_claims(
            claim_texts, analysis_id, model_response.model_id,
        )
        return (claims, None)
    except Exception as exc:
        warning = (
            f"Llama extraction failed for {model_response.model_id}, "
            f"using sentence-split fallback: {exc}"
        )
        sentences = sentence_split_claims(model_response.response_text)
        claims = _texts_to_claims(
            sentences, analysis_id, model_response.model_id,
        )
        return (claims, warning)


# -- 1. extract_claims (GPU) -----------------------------------------------

@app.function(
    image=gpu_image,
    gpu="A10G",
    memory=16384,
    timeout=600,
    volumes={VOLUME_MOUNT: model_volume},
)
def extract_claims(payload: dict) -> dict:
    """Decompose model responses into atomic claims using Llama."""
    try:
        req = ExtractClaimsRequest(**payload)
    except Exception as exc:
        resp = build_warning_response(
            ExtractClaimsResponse,
            payload.get("analysis_id", "unknown"),
            str(exc),
        )
        return resp.model_dump()

    all_claims: list = []
    warnings: list = []

    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch

        tokenizer = AutoTokenizer.from_pretrained(LLAMA_MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(LLAMA_MODEL_NAME)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval()

        for i in range(len(req.responses)):
            if i >= MAX_RESPONSES:
                break
            claims, warning = _extract_from_single_response(
                req.responses[i], req.analysis_id,
                tokenizer, model, device,
            )
            all_claims.extend(claims)
            if warning is not None:
                warnings.append(warning)
    except Exception as exc:
        warnings.append(f"extract_claims model load failed: {exc}")
        all_claims = _fallback_extract_all(req)

    resp = ExtractClaimsResponse(
        analysis_id=req.analysis_id,
        claims=all_claims,
        warnings=warnings,
    )
    return resp.model_dump()


def _fallback_extract_all(req) -> list:
    """Sentence-split fallback for all responses when model fails."""
    from id_utils import make_claim_id
    from claim_extraction import sentence_split_claims

    fallback_claims: list = []
    for i in range(len(req.responses)):
        if i >= MAX_RESPONSES:
            break
        mr = req.responses[i]
        sentences = sentence_split_claims(mr.response_text)
        for j in range(len(sentences)):
            if j >= MAX_RESPONSES * 50:
                break
            cid = make_claim_id(req.analysis_id, mr.model_id, sentences[j])
            fallback_claims.append(Claim(
                claim_id=cid,
                model_id=mr.model_id,
                claim_text=sentences[j],
            ))
    return fallback_claims


# -- 2. embed_claims (GPU) -------------------------------------------------

@app.function(
    image=gpu_image,
    gpu="A10G",
    memory=16384,
    timeout=600,
    volumes={VOLUME_MOUNT: model_volume},
)
def embed_claims(payload: dict) -> dict:
    """Encode claim texts into dense vectors using BGE-large-en-v1.5."""
    try:
        req = EmbedClaimsRequest(**payload)
    except Exception as exc:
        resp = build_warning_response(
            EmbedClaimsResponse,
            payload.get("analysis_id", "unknown"),
            str(exc),
        )
        return resp.model_dump()

    try:
        from sentence_transformers import SentenceTransformer

        claim_ids: list = []
        claim_texts: list = []
        for i in range(len(req.claims)):
            if i >= MAX_VECTORS:
                break
            claim_ids.append(req.claims[i].claim_id)
            claim_texts.append(req.claims[i].claim_text)

        model = SentenceTransformer(req.model_name)
        embeddings = model.encode(
            claim_texts, batch_size=64, normalize_embeddings=True,
        )

        vectors: dict = {}
        dim = 0
        for i in range(len(claim_ids)):
            vec = embeddings[i].tolist()
            vectors[claim_ids[i]] = vec
            if i == 0:
                dim = len(vec)

        resp = EmbedClaimsResponse(
            analysis_id=req.analysis_id,
            vectors=vectors,
            dim=dim,
        )
        return resp.model_dump()

    except Exception as exc:
        resp = build_warning_response(
            EmbedClaimsResponse,
            req.analysis_id,
            f"embed_claims failed: {exc}",
        )
        return resp.model_dump()


# -- 3. cluster_claims (CPU) ------------------------------------------------

@app.function(
    image=cpu_image,
    cpu=4,
    memory=8192,
    timeout=300,
)
def cluster_claims(payload: dict) -> dict:
    """Cluster embedding vectors using Agglomerative Clustering."""
    try:
        req = ClusterClaimsRequest(**payload)
    except Exception as exc:
        resp = build_warning_response(
            ClusterClaimsResponse,
            payload.get("analysis_id", "unknown"),
            str(exc),
        )
        return resp.model_dump()

    try:
        import numpy as np
        from sklearn.cluster import AgglomerativeClustering

        ordered_ids: list = list(req.vectors.keys())
        ordered_vecs: list = []
        for i in range(len(ordered_ids)):
            ordered_vecs.append(req.vectors[ordered_ids[i]])

        # Single vector cannot be clustered
        if len(ordered_ids) == 1:
            return _single_vector_response(ordered_ids[0], req)

        vectors_np = np.array(ordered_vecs, dtype=np.float32)
        distance_threshold = 1.0 - req.sim_threshold

        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=distance_threshold,
            metric="cosine",
            linkage="average",
        )
        labels = clustering.fit_predict(vectors_np)

        clusters = _build_clusters(labels, ordered_ids, req)
        resp = ClusterClaimsResponse(
            analysis_id=req.analysis_id, clusters=clusters,
        )
        return resp.model_dump()

    except Exception as exc:
        clusters = _fallback_single_cluster(req)
        resp = ClusterClaimsResponse(
            analysis_id=req.analysis_id,
            clusters=clusters,
            warnings=[
                f"cluster_claims failed, single cluster fallback: {exc}"
            ],
        )
        return resp.model_dump()


def _single_vector_response(cid: str, req) -> dict:
    """Build response for a single-vector input."""
    from id_utils import make_cluster_id

    rep_text = ""
    if cid in req.claims:
        rep_text = req.claims[cid].claim_text
    cluster_obj = Cluster(
        cluster_id=make_cluster_id([cid]),
        claim_ids=[cid],
        representative_claim_id=cid,
        representative_text=rep_text,
    )
    resp = ClusterClaimsResponse(
        analysis_id=req.analysis_id, clusters=[cluster_obj],
    )
    return resp.model_dump()


def _build_clusters(labels, ordered_ids: list, req) -> list:
    """Group labels into Cluster objects with metadata."""
    from id_utils import make_cluster_id

    cluster_map: dict = {}
    for i in range(len(labels)):
        label = int(labels[i])
        if label not in cluster_map:
            cluster_map[label] = []
        cluster_map[label].append(ordered_ids[i])

    clusters: list = []
    sorted_keys = sorted(cluster_map.keys())
    for key in sorted_keys:
        cids = cluster_map[key]
        rep_id = cids[0]
        rep_text = ""
        if rep_id in req.claims:
            rep_text = req.claims[rep_id].claim_text
        clusters.append(Cluster(
            cluster_id=make_cluster_id(cids),
            claim_ids=cids,
            representative_claim_id=rep_id,
            representative_text=rep_text,
        ))
    return clusters


def _fallback_single_cluster(req) -> list:
    """Put all claims into one cluster as fallback."""
    from id_utils import make_cluster_id

    all_ids = list(req.vectors.keys())
    rep_id = all_ids[0] if len(all_ids) > 0 else ""
    rep_text = ""
    if rep_id in req.claims:
        rep_text = req.claims[rep_id].claim_text
    return [Cluster(
        cluster_id=make_cluster_id(all_ids),
        claim_ids=all_ids,
        representative_claim_id=rep_id,
        representative_text=rep_text,
    )]


# -- 4. rerank_evidence_batch (GPU) -----------------------------------------

@app.function(
    image=gpu_image,
    gpu="A10G",
    memory=16384,
    timeout=600,
    volumes={VOLUME_MOUNT: model_volume},
)
def rerank_evidence_batch(payload: dict) -> dict:
    """Rerank passages against claims using a cross-encoder (batch)."""
    try:
        req = RerankEvidenceBatchRequest(**payload)
    except Exception as exc:
        resp = build_warning_response(
            RerankEvidenceBatchResponse,
            payload.get("analysis_id", "unknown"),
            str(exc),
        )
        return resp.model_dump()

    try:
        from sentence_transformers import CrossEncoder

        model = CrossEncoder(req.reranker_model)
        rankings: list = []
        warnings: list = []

        for i in range(len(req.items)):
            if i >= MAX_ITEMS:
                break
            ranking, warning = _rerank_single_item(
                req.items[i], model, req.top_k,
            )
            rankings.append(ranking)
            if warning is not None:
                warnings.append(warning)

        resp = RerankEvidenceBatchResponse(
            analysis_id=req.analysis_id,
            rankings=rankings,
            warnings=warnings,
        )
        return resp.model_dump()

    except Exception as exc:
        rankings = _fallback_rerank_all(req)
        resp = RerankEvidenceBatchResponse(
            analysis_id=req.analysis_id,
            rankings=rankings,
            warnings=[
                f"rerank_evidence_batch failed, original order: {exc}"
            ],
        )
        return resp.model_dump()


def _rerank_single_item(item, model, top_k: int) -> tuple:
    """Rerank passages for a single RerankItem.

    Returns (ClaimRanking, warning_or_none).
    """
    warning = None
    pairs: list = []
    for j in range(len(item.passages)):
        pairs.append([item.claim_text, item.passages[j].text])

    try:
        scores = model.predict(pairs)
    except Exception as exc:
        warning = f"Rerank failed for {item.claim_id}: {exc}"
        return (_fallback_ranking_for_item(item, top_k), warning)

    scored: list = []
    for j in range(len(scores)):
        scored.append((float(scores[j]), j))

    # Sort descending by score (insertion sort, bounded)
    for j in range(1, len(scored)):
        key = scored[j]
        k = j - 1
        while k >= 0 and scored[k][0] < key[0]:
            scored[k + 1] = scored[k]
            k -= 1
        scored[k + 1] = key

    limit = top_k
    if limit > len(scored):
        limit = len(scored)

    ordered_ids: list = []
    scores_dict: dict = {}
    for j in range(limit):
        score_val, orig_idx = scored[j]
        pid = item.passages[orig_idx].passage_id
        ordered_ids.append(pid)
        scores_dict[pid] = round(score_val, 6)

    ranking = ClaimRanking(
        claim_id=item.claim_id,
        ordered_passage_ids=ordered_ids,
        scores=scores_dict,
    )
    return (ranking, warning)


def _fallback_ranking_for_item(item, top_k: int) -> ClaimRanking:
    """Return original passage order with zero scores."""
    ordered_ids: list = []
    scores_dict: dict = {}
    limit = top_k
    if limit > len(item.passages):
        limit = len(item.passages)
    for j in range(limit):
        pid = item.passages[j].passage_id
        ordered_ids.append(pid)
        scores_dict[pid] = 0.0
    return ClaimRanking(
        claim_id=item.claim_id,
        ordered_passage_ids=ordered_ids,
        scores=scores_dict,
    )


def _fallback_rerank_all(req) -> list:
    """Return original passage order for all items."""
    rankings: list = []
    for i in range(len(req.items)):
        if i >= MAX_ITEMS:
            break
        rankings.append(_fallback_ranking_for_item(
            req.items[i], req.top_k,
        ))
    return rankings


# -- 5. nli_verify_batch (GPU) ----------------------------------------------

@app.function(
    image=gpu_image,
    gpu="A10G",
    memory=24576,
    timeout=900,
    volumes={VOLUME_MOUNT: model_volume},
)
def nli_verify_batch(payload: dict) -> dict:
    """Run NLI on claim-passage pairs using DeBERTa-v3-large."""
    try:
        req = NliVerifyBatchRequest(**payload)
    except Exception as exc:
        resp = build_warning_response(
            NliVerifyBatchResponse,
            payload.get("analysis_id", "unknown"),
            str(exc),
        )
        return resp.model_dump()

    try:
        from transformers import AutoTokenizer
        from transformers import AutoModelForSequenceClassification
        import torch

        tokenizer = AutoTokenizer.from_pretrained(req.nli_model)
        model = AutoModelForSequenceClassification.from_pretrained(
            req.nli_model,
        )
        model.eval()

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)

        id2label = model.config.id2label

        all_results: list = []
        batches = chunk_list(list(range(len(req.pairs))), req.batch_size)

        for batch_idx in range(len(batches)):
            indices = batches[batch_idx]
            batch_results = _run_nli_batch(
                indices, req.pairs, tokenizer, model, device, id2label,
            )
            all_results.extend(batch_results)

        resp = NliVerifyBatchResponse(
            analysis_id=req.analysis_id, results=all_results,
        )
        return resp.model_dump()

    except Exception as exc:
        results = _fallback_nli_all(req)
        resp = NliVerifyBatchResponse(
            analysis_id=req.analysis_id,
            results=results,
            warnings=[f"nli_verify_batch failed, returning neutral: {exc}"],
        )
        return resp.model_dump()


def _run_nli_batch(
    indices: list,
    pairs: list,
    tokenizer,
    model,
    device: str,
    id2label: dict,
) -> list:
    """Run NLI inference on a batch of pair indices."""
    import torch

    premises: list = []
    hypotheses: list = []
    for idx in indices:
        premises.append(pairs[idx].passage_text)
        hypotheses.append(pairs[idx].claim_text)

    inputs = tokenizer(
        premises, hypotheses,
        return_tensors="pt", padding=True,
        truncation=True, max_length=512,
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits.cpu().tolist()

    results: list = []
    for row_idx in range(len(logits)):
        probs = _softmax(logits[row_idx])
        probs_dict: dict = {}
        best_label = id2label[0]
        best_score = probs[0]
        for label_idx in range(len(probs)):
            label_name = id2label[label_idx]
            probs_dict[label_name] = round(probs[label_idx], 6)
            if probs[label_idx] > best_score:
                best_score = probs[label_idx]
                best_label = label_name

        pair = pairs[indices[row_idx]]
        results.append(NliResultOutput(
            pair_id=pair.pair_id,
            claim_id=pair.claim_id,
            passage_id=pair.passage_id,
            label=best_label,
            probs=probs_dict,
        ))
    return results


def _fallback_nli_all(req) -> list:
    """Return neutral with near-uniform probs for all pairs."""
    uniform = {
        "entailment": 0.34,
        "contradiction": 0.33,
        "neutral": 0.33,
    }
    results: list = []
    for i in range(len(req.pairs)):
        if i >= MAX_PAIRS:
            break
        pair = req.pairs[i]
        results.append(NliResultOutput(
            pair_id=pair.pair_id,
            claim_id=pair.claim_id,
            passage_id=pair.passage_id,
            label="neutral",
            probs=uniform,
        ))
    return results


# -- 6. compute_umap (CPU) --------------------------------------------------

@app.function(
    image=cpu_image,
    cpu=8,
    memory=16384,
    timeout=600,
)
def compute_umap(payload: dict) -> dict:
    """Project high-dimensional vectors into 3-D via UMAP."""
    try:
        req = ComputeUmapRequest(**payload)
    except Exception as exc:
        resp = build_warning_response(
            ComputeUmapResponse,
            payload.get("analysis_id", "unknown"),
            str(exc),
        )
        return resp.model_dump()

    try:
        import numpy as np
        import umap

        ordered_ids: list = list(req.vectors.keys())
        ordered_vecs: list = []
        for i in range(len(ordered_ids)):
            ordered_vecs.append(req.vectors[ordered_ids[i]])

        vectors_np = np.array(ordered_vecs, dtype=np.float32)

        reducer = umap.UMAP(
            n_components=3,
            n_neighbors=req.n_neighbors,
            min_dist=req.min_dist,
            metric="cosine",
            random_state=req.random_state,
        )
        coords = reducer.fit_transform(vectors_np)

        coords3d: dict = {}
        for i in range(len(ordered_ids)):
            coords3d[ordered_ids[i]] = [
                float(coords[i][0]),
                float(coords[i][1]),
                float(coords[i][2]),
            ]

        resp = ComputeUmapResponse(
            analysis_id=req.analysis_id, coords3d=coords3d,
        )
        return resp.model_dump()

    except Exception as exc:
        ordered_ids = list(req.vectors.keys())
        zero_coords: dict = {}
        for i in range(len(ordered_ids)):
            zero_coords[ordered_ids[i]] = [0.0, 0.0, 0.0]

        resp = ComputeUmapResponse(
            analysis_id=req.analysis_id,
            coords3d=zero_coords,
            warnings=[f"compute_umap failed, returning zeros: {exc}"],
        )
        return resp.model_dump()


# -- 7. score_clusters (CPU) ------------------------------------------------

@app.function(
    image=cpu_image,
    cpu=4,
    memory=8192,
    timeout=300,
)
def score_clusters(payload: dict) -> dict:
    """Compute trust scores and verdicts for each cluster."""
    try:
        req = ScoreClustersRequest(**payload)
    except Exception as exc:
        resp = build_warning_response(
            ScoreClustersResponse,
            payload.get("analysis_id", "unknown"),
            str(exc),
        )
        return resp.model_dump()

    try:
        from scoring import (
            find_supporting_models,
            compute_agreement_score,
            find_best_nli_for_cluster,
            compute_verification_score,
            compute_trust_score,
            determine_verdict,
        )

        scores: list = []
        for i in range(len(req.clusters)):
            if i >= MAX_CLUSTERS:
                break
            cluster_score = _score_single_cluster(
                req.clusters[i], req,
                find_supporting_models,
                compute_agreement_score,
                find_best_nli_for_cluster,
                compute_verification_score,
                compute_trust_score,
                determine_verdict,
            )
            scores.append(cluster_score)

        resp = ScoreClustersResponse(
            analysis_id=req.analysis_id, scores=scores,
        )
        return resp.model_dump()

    except Exception as exc:
        resp = build_warning_response(
            ScoreClustersResponse,
            req.analysis_id,
            f"score_clusters failed: {exc}",
        )
        return resp.model_dump()


def _score_single_cluster(
    cluster, req,
    find_supporting_models_fn,
    compute_agreement_score_fn,
    find_best_nli_for_cluster_fn,
    compute_verification_score_fn,
    compute_trust_score_fn,
    determine_verdict_fn,
) -> ClusterScore:
    """Compute trust score and verdict for a single cluster."""
    supporting = find_supporting_models_fn(
        cluster.claim_ids, req.claims,
    )
    agreement = compute_agreement_score_fn(len(supporting), 5)

    best_ent, best_contra, evidence_pid = find_best_nli_for_cluster_fn(
        cluster.claim_ids, req.nli_results,
    )
    verification = compute_verification_score_fn(best_ent, best_contra)

    trust = compute_trust_score_fn(
        agreement, verification,
        req.weights.agreement_weight,
        req.weights.verification_weight,
    )

    verdict = determine_verdict_fn(
        trust, best_contra,
        req.verdict_thresholds.safe_min,
        req.verdict_thresholds.caution_min,
    )

    return ClusterScore(
        cluster_id=cluster.cluster_id,
        trust_score=trust,
        verdict=verdict,
        agreement=AgreementDetail(
            models_supporting=supporting,
            count=len(supporting),
        ),
        verification=VerificationDetail(
            best_entailment_prob=best_ent,
            best_contradiction_prob=best_contra,
            evidence_passage_id=evidence_pid,
        ),
    )
