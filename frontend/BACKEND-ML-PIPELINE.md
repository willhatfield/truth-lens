# TruthLens Integration Spec (v1)
Backend ↔ ML (Modal) and Backend ↔ Frontend contracts

This file is the single source of truth for how services communicate so the system is correctly hooked up end-to-end.

---

## 0) Goals and Principles

1. **Backend is the orchestrator**
   - Calls 5 LLM providers, retrieves evidence, calls ML on Modal, streams progress to UI.
   - Backend must not load torch models.

2. **ML is compute-only**
   - Runs embedding/rerank/NLI/UMAP on Modal with explicit resources.
   - Accepts and returns JSON-serializable inputs/outputs only.
   - Never crashes the pipeline; returns warnings and neutral fallbacks instead.

3. **Frontend is renderer-only**
   - Consumes backend events + final payload.
   - Does not implement ML logic.

4. **Stable contracts**
   - Every message includes `schema_version: "1.0"`.
   - Additive changes within v1.x only; breaking changes require v2.

---

## 1) System Overview

### 1.1 High-level data flow
1) User submits `prompt` to backend
2) Backend calls 5 LLMs (“arena”) and streams tokens to frontend
3) Backend sends outputs to ML (Modal) for:
   - claim extraction
   - embeddings
   - clustering
   - rerank evidence
   - NLI verification
   - trust scoring
   - UMAP 3D projection
4) Backend builds:
   - Safe Answer (from verified clusters)
   - Visualization payload (constellation + deck)
5) Backend streams stage events to frontend and ends with `DONE` containing the final payload

### 1.2 Shared identifiers
- `analysis_id`: created by backend for a single run
- `model_id`: identifier for each LLM (e.g., `openai_gpt4o`, `claude_3_5`, `gemini_1_5`, `llama_3`, `mistral`)
- `claim_id`: unique per atomic claim
- `cluster_id`: unique per cluster of semantically similar claims
- `passage_id`: unique per evidence passage
- `pair_id`: unique per (claim, passage) pair for NLI

Recommended deterministic IDs:
- `claim_id = sha1(f"{analysis_id}:{model_id}:{claim_text}")`
- `cluster_id = sha1("|".join(sorted(claim_ids)))`
- `pair_id = sha1(f"{claim_id}:{passage_id}")`

---

## 2) Backend ↔ Frontend Communication

### 2.1 Transport
Choose one:
- **WebSocket** (recommended: token streaming + many event types)
- **SSE** (acceptable if no bidirectional needs)

### 2.2 Endpoints (suggested)
- `POST /analyze` → starts a run, returns `{ analysis_id }`
- `WS /ws/analysis/{analysis_id}` → streams events to the client
- Optional: `GET /analysis/{analysis_id}` → fetch final payload by ID

### 2.3 Event envelope (all streamed messages)
Every event MUST follow this envelope:

{
  "schema_version": "1.0",
  "analysis_id": "a_...",
  "type": "EVENT_TYPE",
  "ts": "2026-02-27T18:00:00Z",
  "payload": {}
}
2.4 Core event types
A) LLM Arena streaming

MODEL_STARTED

MODEL_TOKEN

MODEL_DONE

MODEL_STARTED

{
  "schema_version": "1.0",
  "analysis_id": "a_...",
  "type": "MODEL_STARTED",
  "ts": "2026-02-27T18:00:01Z",
  "payload": { "model_id": "openai_gpt4o" }
}

MODEL_TOKEN

{
  "schema_version": "1.0",
  "analysis_id": "a_...",
  "type": "MODEL_TOKEN",
  "ts": "2026-02-27T18:00:02Z",
  "payload": {
    "model_id": "openai_gpt4o",
    "delta": "Next token chunk..."
  }
}

MODEL_DONE

{
  "schema_version": "1.0",
  "analysis_id": "a_...",
  "type": "MODEL_DONE",
  "ts": "2026-02-27T18:00:06Z",
  "payload": {
    "model_id": "openai_gpt4o",
    "response_text": "Full model response (optional if frontend already assembled)."
  }
}
B) Pipeline stage events (backend emits when stage outputs are available)

CLAIMS_READY

EMBEDDINGS_READY (optional to send; typically summary only)

CLUSTERS_READY

EVIDENCE_CANDIDATES_READY (backend retrieval complete)

EVIDENCE_RERANKED

NLI_READY

SCORES_READY

UMAP_READY

SAFE_ANSWER_READY

Each stage event should include either:

summary counts only, and frontend fetches full payload later; or

the full stage output if small.

Example summary

{
  "schema_version": "1.0",
  "analysis_id": "a_...",
  "type": "CLUSTERS_READY",
  "ts": "2026-02-27T18:00:10Z",
  "payload": { "cluster_count": 12 }
}
C) Errors

STAGE_FAILED (non-fatal; degrade gracefully)

FATAL_ERROR (stop run)

STAGE_FAILED

{
  "schema_version": "1.0",
  "analysis_id": "a_...",
  "type": "STAGE_FAILED",
  "ts": "2026-02-27T18:00:12Z",
  "payload": {
    "stage": "nli_verify_batch",
    "message": "NLI unavailable; using neutral fallback."
  }
}
D) Finalization

DONE

DONE MUST contain the final payload (or a URL pointer to fetch it):

{
  "schema_version": "1.0",
  "analysis_id": "a_...",
  "type": "DONE",
  "ts": "2026-02-27T18:00:30Z",
  "payload": {
    "result": { /* Final AnalysisResult payload (section 6) */ }
  }
}
3) Backend ↔ ML (Modal) Communication
3.1 Modal execution pattern

ML exposes Modal functions callable from backend via .remote(). Heavy functions declare explicit GPU/CPU, memory, and timeout.

Backend import/call pattern (Python):

from ml.modal_app import embed_claims, cluster_claims

resp = embed_claims.remote(request_dict)
3.2 Resource-heavy work routing rule

If a step:

loads torch models,

needs >2GB RAM, or

is slower than ~1s per batch,

then it runs on Modal, not on backend.

3.3 Caching requirement

ML must mount a Modal Volume at /models and set caches:

HF_HOME=/models/hf

TRANSFORMERS_CACHE=/models/hf

SENTENCE_TRANSFORMERS_HOME=/models/st

This prevents repeated downloads on cold starts.

3.4 Batching requirements (backend MUST comply)

Backend should:

send all claims in one embedding request (or chunks of 64–256)

rerank top-K (default K=10) passages per claim

run NLI only on top-k after rerank (default k=3–5)

prefer fewer Modal calls with larger payloads

4) Canonical JSON Schemas (Shared)

All request/response objects MUST include:

schema_version: "1.0"

analysis_id

4.1 ModelResponse
{
  "model_id": "openai_gpt4o",
  "response_text": "..."
}
4.2 Claim
{
  "claim_id": "c_...",
  "model_id": "openai_gpt4o",
  "claim_text": "Atomic factual statement.",
  "span": { "start": 0, "end": 42 }
}

Notes:

span is optional; if present, offsets refer to the original response_text.

4.3 Passage (Evidence candidate)
{
  "passage_id": "p_...",
  "source": {
    "type": "web",
    "title": "…",
    "url": "…",
    "retrieved_at": "2026-02-27T18:00:00Z"
  },
  "text": "Evidence text…"
}
4.4 NLIPair
{
  "pair_id": "nli_...",
  "claim_id": "c_...",
  "passage_id": "p_...",
  "claim_text": "…",
  "passage_text": "…"
}
4.5 NLIResult
{
  "pair_id": "nli_...",
  "claim_id": "c_...",
  "passage_id": "p_...",
  "label": "entailment",
  "probs": {
    "entailment": 0.92,
    "contradiction": 0.03,
    "neutral": 0.05
  }
}

Allowed labels:

entailment

contradiction

neutral

4.6 Cluster
{
  "cluster_id": "cl_...",
  "claim_ids": ["c_1", "c_2"],
  "representative_claim_id": "c_1",
  "representative_text": "..."
}
4.7 ClusterScore
{
  "cluster_id": "cl_...",
  "trust_score": 84,
  "verdict": "SAFE",
  "agreement": {
    "models_supporting": ["openai_gpt4o", "claude_3_5"],
    "count": 2
  },
  "verification": {
    "best_entailment_prob": 0.92,
    "best_contradiction_prob": 0.08,
    "evidence_passage_id": "p_..."
  }
}

Allowed verdicts:

SAFE (strongly verified)

CAUTION (uncertain/partially supported)

REJECT (contradicted or unsupported)

5) ML (Modal) Functions — Required Interfaces

All ML functions take a single request object and return a single response object.

5.1 extract_claims

Purpose: Convert each model response into atomic claims.

Request:

{
  "schema_version": "1.0",
  "analysis_id": "a_...",
  "responses": [
    { "model_id": "openai_gpt4o", "response_text": "..." }
  ]
}

Response:

{
  "schema_version": "1.0",
  "analysis_id": "a_...",
  "claims": [
    { "claim_id": "c_...", "model_id": "openai_gpt4o", "claim_text": "...", "span": null }
  ],
  "warnings": []
}

Fallback:

If LLM-based extraction fails, do sentence-split fallback.

5.2 embed_claims (GPU)

Purpose: Embed claim texts into vectors (default dim=768).

Request:

{
  "schema_version": "1.0",
  "analysis_id": "a_...",
  "claims": [
    { "claim_id": "c_...", "claim_text": "..." }
  ],
  "model_name": "BAAI/bge-large-en-v1.5"
}

Response:

{
  "schema_version": "1.0",
  "analysis_id": "a_...",
  "vectors": {
    "c_...": [0.12, -0.03, "..."]
  },
  "dim": 768,
  "warnings": []
}

Batching:

Backend sends all claims or chunks of 64–256.

5.3 cluster_claims (CPU)

Purpose: Group semantically similar claims.

Request:

{
  "schema_version": "1.0",
  "analysis_id": "a_...",
  "vectors": {
    "c_1": [ ... ],
    "c_2": [ ... ]
  },
  "claims": {
    "c_1": { "model_id": "openai_gpt4o", "claim_text": "..." },
    "c_2": { "model_id": "claude_3_5", "claim_text": "..." }
  },
  "sim_threshold": 0.85
}

Response:

{
  "schema_version": "1.0",
  "analysis_id": "a_...",
  "clusters": [
    {
      "cluster_id": "cl_...",
      "claim_ids": ["c_1", "c_2"],
      "representative_claim_id": "c_1",
      "representative_text": "..."
    }
  ],
  "warnings": []
}
5.4 rerank_evidence_batch (GPU)

Purpose: Cross-encoder rerank candidate passages for each claim.

Request:

{
  "schema_version": "1.0",
  "analysis_id": "a_...",
  "items": [
    {
      "claim_id": "c_...",
      "claim_text": "...",
      "passages": [
        { "passage_id": "p_1", "text": "..." },
        { "passage_id": "p_2", "text": "..." }
      ]
    }
  ],
  "reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
  "top_k": 10
}

Response:

{
  "schema_version": "1.0",
  "analysis_id": "a_...",
  "rankings": [
    {
      "claim_id": "c_...",
      "ordered_passage_ids": ["p_2", "p_1"],
      "scores": { "p_2": 12.3, "p_1": 10.8 }
    }
  ],
  "warnings": []
}

Fallback:

If rerank fails, return original ordering and a warning.

5.5 nli_verify_batch (GPU)

Purpose: NLI classification for (claim, passage) pairs.

Request:

{
  "schema_version": "1.0",
  "analysis_id": "a_...",
  "pairs": [
    {
      "pair_id": "nli_1",
      "claim_id": "c_...",
      "passage_id": "p_...",
      "claim_text": "...",
      "passage_text": "..."
    }
  ],
  "nli_model": "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli"
}

Response:

{
  "schema_version": "1.0",
  "analysis_id": "a_...",
  "results": [
    {
      "pair_id": "nli_1",
      "claim_id": "c_...",
      "passage_id": "p_...",
      "label": "entailment",
      "probs": { "entailment": 0.92, "contradiction": 0.03, "neutral": 0.05 }
    }
  ],
  "warnings": []
}

Fallback:

If NLI fails, return neutral with near-uniform probs (0.34/0.33/0.33) and a warning.

Batching:

Backend only sends top passages per claim (k=3–5).

5.6 compute_umap (CPU)

Purpose: Reduce vectors to 3D coords for visualization.

Request:

{
  "schema_version": "1.0",
  "analysis_id": "a_...",
  "vectors": { "c_1": [ ... ], "c_2": [ ... ] },
  "random_state": 42,
  "n_neighbors": 15,
  "min_dist": 0.1
}

Response:

{
  "schema_version": "1.0",
  "analysis_id": "a_...",
  "coords3d": {
    "c_1": [0.12, 1.03, -0.44],
    "c_2": [-0.50, 0.22, 0.90]
  },
  "warnings": []
}

Fallback:

If UMAP fails, return PCA-3D or zeros and a warning.

5.7 score_clusters (CPU)

Purpose: Compute trust score and verdict per cluster using agreement + best NLI.

Request:

{
  "schema_version": "1.0",
  "analysis_id": "a_...",
  "clusters": [
    { "cluster_id": "cl_...", "claim_ids": ["c_1","c_2"], "representative_claim_id": "c_1", "representative_text": "..." }
  ],
  "claims": {
    "c_1": { "model_id": "openai_gpt4o", "claim_text": "..." },
    "c_2": { "model_id": "claude_3_5", "claim_text": "..." }
  },
  "nli_results": [
    { "pair_id": "nli_1", "claim_id": "c_1", "passage_id": "p_1", "label": "entailment", "probs": { "entailment": 0.92, "contradiction": 0.03, "neutral": 0.05 } }
  ],
  "weights": { "agreement_weight": 0.4, "verification_weight": 0.6 },
  "verdict_thresholds": { "safe_min": 75, "caution_min": 45 }
}

Response:

{
  "schema_version": "1.0",
  "analysis_id": "a_...",
  "scores": [
    {
      "cluster_id": "cl_...",
      "trust_score": 84,
      "verdict": "SAFE",
      "agreement": { "models_supporting": ["openai_gpt4o","claude_3_5"], "count": 2 },
      "verification": { "best_entailment_prob": 0.92, "best_contradiction_prob": 0.08, "evidence_passage_id": "p_1" }
    }
  ],
  "warnings": []
}

Suggested scoring (simple + explainable):

agreement_score = 100 * (models_supporting_count / 5)

verification_score = 100 * max_entailment_prob - 100 * max_contradiction_prob

trust_score = round(agreement_weight * agreement_score + verification_weight * clamp(verification_score, 0, 100))

verdict:

SAFE if trust_score >= safe_min and best_contradiction_prob <= 0.2

CAUTION if trust_score >= caution_min

else REJECT

6) Backend Orchestration Sequence (Recommended)

Backend should run and stream events in this order:

LLM arena: stream MODEL_* events as tokens arrive

Call ML extract_claims → emit CLAIMS_READY

Call ML embed_claims → emit EMBEDDINGS_READY (summary is fine)

Call ML cluster_claims → emit CLUSTERS_READY

Backend evidence retrieval → emit EVIDENCE_CANDIDATES_READY

Call ML rerank_evidence_batch → emit EVIDENCE_RERANKED

Call ML nli_verify_batch → emit NLI_READY

Call ML score_clusters → emit SCORES_READY

Call ML compute_umap → emit UMAP_READY

Backend builds safe answer + final payload → emit SAFE_ANSWER_READY, then DONE

If any ML stage fails:

backend emits STAGE_FAILED and continues with fallbacks if possible.

7) Final Payload Schema (Backend → Frontend)

The backend must deliver a final object AnalysisResult in the DONE.payload.result.

{
  "schema_version": "1.0",
  "analysis_id": "a_...",
  "prompt": "...",
  "models": [
    {
      "model_id": "openai_gpt4o",
      "response_text": "..."
    }
  ],
  "claims": [
    {
      "claim_id": "c_...",
      "model_id": "openai_gpt4o",
      "claim_text": "...",
      "span": null
    }
  ],
  "clusters": [
    {
      "cluster_id": "cl_...",
      "claim_ids": ["c_1","c_2"],
      "representative_claim_id": "c_1",
      "representative_text": "..."
    }
  ],
  "evidence": [
    {
      "passage_id": "p_...",
      "source": { "type": "web", "title": "...", "url": "...", "retrieved_at": "..." },
      "text": "..."
    }
  ],
  "nli_results": [
    {
      "pair_id": "nli_...",
      "claim_id": "c_...",
      "passage_id": "p_...",
      "label": "entailment",
      "probs": { "entailment": 0.92, "contradiction": 0.03, "neutral": 0.05 }
    }
  ],
  "cluster_scores": [
    {
      "cluster_id": "cl_...",
      "trust_score": 84,
      "verdict": "SAFE",
      "agreement": { "models_supporting": ["openai_gpt4o","claude_3_5"], "count": 2 },
      "verification": { "best_entailment_prob": 0.92, "best_contradiction_prob": 0.08, "evidence_passage_id": "p_..." }
    }
  ],
  "coords3d": {
    "c_1": [0.12, 1.03, -0.44],
    "c_2": [-0.50, 0.22, 0.90]
  },
  "safe_answer": {
    "text": "...",
    "supported_cluster_ids": ["cl_..."],
    "rejected_cluster_ids": ["cl_..."]
  },
  "model_metrics": [
    {
      "model_id": "openai_gpt4o",
      "claim_counts": { "total": 12, "supported": 9, "caution": 2, "rejected": 1 }
    }
  ],
  "warnings": []
}

Frontend rendering expectations:

Constellation mode uses coords3d for points and cluster_scores for size/color/verdict.

Deck mode uses safe_answer, clusters, and evidence + NLI details.

Frontend should not compute trust logic; it displays what backend provides.

8) Modal Resource Settings (ML Implementation Requirements)

ML functions must declare resources appropriate to their workload.

Suggested v1 defaults:

embed_claims: GPU A10G or L4, memory 16384MB, timeout 600s

rerank_evidence_batch: GPU A10G or L4, memory 16384MB, timeout 600s

nli_verify_batch: GPU A10G or L4, memory 24576MB (or 16384 for smaller model), timeout 900s

cluster_claims: CPU 2–4, memory 4096–8192, timeout 300s

compute_umap: CPU 4–8, memory 8192–16384, timeout 600s

score_clusters: CPU 2–4, memory 4096–8192, timeout 300s

Caching:

Mount Modal Volume at /models and set cache env vars:

HF_HOME=/models/hf

TRANSFORMERS_CACHE=/models/hf

SENTENCE_TRANSFORMERS_HOME=/models/st

Batching:

embed: 64–256 claims per batch

rerank: N*K pairs per batch (N claims, K passages)

NLI: top 3–5 passages per claim

9) Non-fatal Degradation Rules

ML must never fail hard for expected issues. It should return:

partial results where possible

warnings list describing degradation

Required fallbacks:

rerank failure → original passage ordering

NLI failure → neutral with probs ~uniform

UMAP failure → PCA-3D or zeros

claim extraction failure → sentence split fallback

Backend:

emits STAGE_FAILED for any degraded stage

still emits DONE if it can produce a usable payload

10) Versioning Rules

Every request/response/event includes schema_version.

Backend accepts:

1.x versions

2.x is breaking and requires coordinated update across backend/ML/frontend.
