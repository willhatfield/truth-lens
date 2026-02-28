"""TruthLens ML pipeline — Modal serverless functions.

Provides five functions callable via `fn.remote(dict)`:
  embed_claims, rerank_evidence, nli_verify, cluster_claims, compute_umap.

Each function accepts a plain dict, validates via Pydantic inside the
boundary, and returns a plain dict (always including an ``error`` field).
"""

import modal

from schemas import (
    EmbedClaimsRequest,
    EmbedClaimsResponse,
    ClusterClaimsRequest,
    ClusterClaimsResponse,
    RerankEvidenceRequest,
    RerankEvidenceResponse,
    RankedPassage,
    NliVerifyRequest,
    NliVerifyResponse,
    NliResult,
    ComputeUmapRequest,
    ComputeUmapResponse,
    UmapPoint,
)
from batch_utils import chunk_list, flatten_batch_results
from fallback_utils import build_error_response

# ── Modal App + shared volume ────────────────────────────────────────────────

app = modal.App("truthlens-ml")

model_volume = modal.Volume.from_name(
    "truthlens-model-cache", create_if_missing=True
)

VOLUME_MOUNT = "/models"
SHARED_ENV = {
    "HF_HOME": "/models/hf",
    "TRANSFORMERS_CACHE": "/models/transformers",
    "SENTENCE_TRANSFORMERS_HOME": "/models/sentence_transformers",
}

# ── Container images ─────────────────────────────────────────────────────────

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
    .add_local_python_source("schemas", "batch_utils", "fallback_utils")
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
    .add_local_python_source("schemas", "batch_utils", "fallback_utils")
)

# ── Model name constants ─────────────────────────────────────────────────────

EMBED_MODEL_NAME = "BAAI/bge-large-en-v1.5"
RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
NLI_MODEL_NAME = "microsoft/deberta-large-mnli"

# ── Helpers (module-level, no recursion) ──────────────────────────────────────


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


# ── 1. embed_claims (GPU) ────────────────────────────────────────────────────

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
        resp = build_error_response(EmbedClaimsResponse, str(exc))
        return resp.model_dump()

    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(EMBED_MODEL_NAME)

        all_vectors: list = []
        batches = chunk_list(req.claim_texts, req.batch_size)
        for batch_idx in range(len(batches)):
            batch = batches[batch_idx]
            embeddings = model.encode(batch, normalize_embeddings=True)
            for row_idx in range(len(embeddings)):
                all_vectors.append(embeddings[row_idx].tolist())

        dimension = 0
        if len(all_vectors) > 0:
            dimension = len(all_vectors[0])

        resp = EmbedClaimsResponse(
            vectors=all_vectors,
            dimension=dimension,
            model_name=EMBED_MODEL_NAME,
        )
        return resp.model_dump()

    except Exception as exc:
        resp = build_error_response(
            EmbedClaimsResponse, f"embed_claims failed: {exc}"
        )
        return resp.model_dump()


# ── 2. rerank_evidence (GPU) ─────────────────────────────────────────────────

@app.function(
    image=gpu_image,
    gpu="A10G",
    memory=16384,
    timeout=600,
    volumes={VOLUME_MOUNT: model_volume},
)
def rerank_evidence(payload: dict) -> dict:
    """Rerank passages against a claim using a cross-encoder."""
    try:
        req = RerankEvidenceRequest(**payload)
    except Exception as exc:
        resp = build_error_response(RerankEvidenceResponse, str(exc))
        return resp.model_dump()

    try:
        from sentence_transformers import CrossEncoder

        model = CrossEncoder(RERANK_MODEL_NAME)

        pairs: list = []
        for i in range(len(req.passages)):
            pairs.append([req.claim, req.passages[i]])

        scores = model.predict(pairs)

        scored: list = []
        for i in range(len(scores)):
            scored.append((float(scores[i]), i))

        # Sort descending by score (simple insertion sort, bounded)
        for i in range(1, len(scored)):
            key = scored[i]
            j = i - 1
            while j >= 0 and scored[j][0] < key[0]:
                scored[j + 1] = scored[j]
                j -= 1
            scored[j + 1] = key

        top_k = req.top_k
        if top_k > len(scored):
            top_k = len(scored)

        ranked: list = []
        for i in range(top_k):
            score_val, orig_idx = scored[i]
            ranked.append(
                RankedPassage(
                    index=orig_idx,
                    text=req.passages[orig_idx],
                    score=score_val,
                )
            )

        resp = RerankEvidenceResponse(ranked_passages=ranked)
        return resp.model_dump()

    except Exception as exc:
        # Fallback: return passages in original order
        fallback: list = []
        for i in range(len(req.passages)):
            if i >= req.top_k:
                break
            fallback.append(
                RankedPassage(index=i, text=req.passages[i], score=0.0)
            )
        resp = RerankEvidenceResponse(
            ranked_passages=fallback,
            error=f"rerank_evidence failed, returning original order: {exc}",
        )
        return resp.model_dump()


# ── 3. nli_verify (GPU) ─────────────────────────────────────────────────────

NLI_LABELS = ["contradiction", "neutral", "entailment"]

@app.function(
    image=gpu_image,
    gpu="A10G",
    memory=24576,
    timeout=900,
    volumes={VOLUME_MOUNT: model_volume},
)
def nli_verify(payload: dict) -> dict:
    """Run NLI on premise-hypothesis pairs using DeBERTa-large-MNLI."""
    try:
        req = NliVerifyRequest(**payload)
    except Exception as exc:
        resp = build_error_response(NliVerifyResponse, str(exc))
        return resp.model_dump()

    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch

        tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL_NAME)
        model.eval()

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)

        all_results: list = []
        batches = chunk_list(list(range(len(req.pairs))), req.batch_size)

        for batch_idx in range(len(batches)):
            indices = batches[batch_idx]
            premises: list = []
            hypotheses: list = []
            for idx in indices:
                premises.append(req.pairs[idx].premise)
                hypotheses.append(req.pairs[idx].hypothesis)

            inputs = tokenizer(
                premises,
                hypotheses,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(device)

            with torch.no_grad():
                outputs = model(**inputs)

            logits = outputs.logits.cpu().tolist()

            for row_idx in range(len(logits)):
                probs = _softmax(logits[row_idx])
                scores_dict: dict = {}
                best_label = NLI_LABELS[0]
                best_score = probs[0]
                for label_idx in range(len(NLI_LABELS)):
                    scores_dict[NLI_LABELS[label_idx]] = round(probs[label_idx], 6)
                    if probs[label_idx] > best_score:
                        best_score = probs[label_idx]
                        best_label = NLI_LABELS[label_idx]

                all_results.append(
                    NliResult(label=best_label, scores=scores_dict)
                )

        resp = NliVerifyResponse(results=all_results)
        return resp.model_dump()

    except Exception as exc:
        # Fallback: all neutral with uniform scores
        fallback_results: list = []
        uniform = {label: round(1.0 / 3.0, 6) for label in NLI_LABELS}
        for _ in range(len(req.pairs)):
            fallback_results.append(
                NliResult(label="neutral", scores=uniform)
            )
        resp = NliVerifyResponse(
            results=fallback_results,
            error=f"nli_verify failed, returning neutral: {exc}",
        )
        return resp.model_dump()


# ── 4. cluster_claims (CPU) ──────────────────────────────────────────────────

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
        resp = build_error_response(ClusterClaimsResponse, str(exc))
        return resp.model_dump()

    try:
        import numpy as np
        from sklearn.cluster import AgglomerativeClustering

        # Single vector cannot be clustered; return it as its own cluster
        if len(req.vectors) == 1:
            resp = ClusterClaimsResponse(clusters=[[0]], num_clusters=1)
            return resp.model_dump()

        vectors = np.array(req.vectors, dtype=np.float32)

        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=req.threshold,
            metric="cosine",
            linkage="average",
        )
        labels = clustering.fit_predict(vectors)

        cluster_map: dict = {}
        for i in range(len(labels)):
            label = int(labels[i])
            if label not in cluster_map:
                cluster_map[label] = []
            cluster_map[label].append(i)

        clusters: list = []
        sorted_keys = sorted(cluster_map.keys())
        for key in sorted_keys:
            clusters.append(cluster_map[key])

        resp = ClusterClaimsResponse(
            clusters=clusters,
            num_clusters=len(clusters),
        )
        return resp.model_dump()

    except Exception as exc:
        # Fallback: single cluster with all indices
        all_indices = list(range(len(req.vectors)))
        resp = ClusterClaimsResponse(
            clusters=[all_indices],
            num_clusters=1,
            error=f"cluster_claims failed, single cluster fallback: {exc}",
        )
        return resp.model_dump()


# ── 5. compute_umap (CPU) ────────────────────────────────────────────────────

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
        resp = build_error_response(ComputeUmapResponse, str(exc))
        return resp.model_dump()

    try:
        import numpy as np
        import umap

        vectors = np.array(req.vectors, dtype=np.float32)

        reducer = umap.UMAP(
            n_components=3,
            n_neighbors=req.n_neighbors,
            min_dist=req.min_dist,
            metric="cosine",
        )
        coords = reducer.fit_transform(vectors)

        points: list = []
        for i in range(len(coords)):
            points.append(
                UmapPoint(
                    x=float(coords[i][0]),
                    y=float(coords[i][1]),
                    z=float(coords[i][2]),
                )
            )

        resp = ComputeUmapResponse(coords_3d=points)
        return resp.model_dump()

    except Exception as exc:
        # Fallback: zero coordinates
        zero_points: list = []
        for _ in range(len(req.vectors)):
            zero_points.append(UmapPoint())
        resp = ComputeUmapResponse(
            coords_3d=zero_points,
            error=f"compute_umap failed, returning zeros: {exc}",
        )
        return resp.model_dump()
