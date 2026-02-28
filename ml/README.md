# TruthLens ML Pipeline - Backend Integration Guide

## Overview

The TruthLens ML layer is a 7-function pipeline deployed on [Modal](https://modal.com/) (serverless GPU/CPU). It extracts claims from LLM responses, embeds and clusters them, retrieves and reranks evidence, runs NLI verification, projects to 3D, and scores trust.

---

## Authentication

All endpoints require a bearer token in the `Authorization` header:

```
Authorization: Bearer <token>
```

- The token is stored as a Modal Secret named `truthlens-api-key`, exposed via env var `MODAL_API_KEY`.
- **401** is returned on invalid or missing token.
- **500** is returned if the server-side secret is not configured.

---

## URL Pattern

Modal uses subdomain-based routing. Each function is exposed at:

```
https://{workspace}--truthlens-ml-{function-name}.modal.run
```

Underscores in function names become hyphens in the URL suffix. For example:

| Python function        | URL suffix                      |
|------------------------|---------------------------------|
| `http_extract_claims`  | `http-extract-claims`           |

Full URL example:

```
https://myworkspace--truthlens-ml-http-extract-claims.modal.run
```

---

## Endpoints

| # | Function                 | URL Suffix                    | Method | Resource       | Timeout |
|---|--------------------------|-------------------------------|--------|----------------|---------|
| 1 | `extract_claims`         | `http-extract-claims`         | POST   | GPU A10G, 16GB | 600s    |
| 2 | `embed_claims`           | `http-embed-claims`           | POST   | GPU A10G, 16GB | 600s    |
| 3 | `cluster_claims`         | `http-cluster-claims`         | POST   | CPU 4c, 8GB    | 300s    |
| 4 | `rerank_evidence_batch`  | `http-rerank-evidence-batch`  | POST   | GPU A10G, 16GB | 600s    |
| 5 | `nli_verify_batch`       | `http-nli-verify-batch`       | POST   | GPU A10G, 24GB | 900s    |
| 6 | `compute_umap`           | `http-compute-umap`           | POST   | CPU 8c, 16GB   | 600s    |
| 7 | `score_clusters`         | `http-score-clusters`         | POST   | CPU 4c, 8GB    | 300s    |

---

## Call Order and Dependencies

```
extract_claims
    |
    v
embed_claims
    |          \
    v           v
cluster_claims  compute_umap  (parallel, independent)
    |
    v
rerank_evidence_batch  (needs claims from extract)
    |
    v
nli_verify_batch
    |
    v
score_clusters  (needs clusters, claims, nli_results)
```

**Main chain:** `extract` -> `embed` -> `cluster` -> `rerank` -> `nli` -> `score`

**Parallel branch:** `embed` -> `compute_umap` (independent of the main chain)

---

## Per-Endpoint Request/Response Schemas

### 1. extract_claims

**Request:**

| Field            | Type                    | Required | Default |
|------------------|-------------------------|----------|---------|
| `schema_version` | string                  | no       | `"1.0"` |
| `analysis_id`    | string (min_length=1)   | yes      | --      |
| `responses`      | array of `ModelResponse` (min 1) | yes | --   |

`ModelResponse`:

| Field           | Type   | Required |
|-----------------|--------|----------|
| `model_id`      | string | yes      |
| `response_text` | string | yes      |

**Response:**

| Field            | Type                  |
|------------------|-----------------------|
| `schema_version` | string                |
| `analysis_id`    | string                |
| `warnings`       | array of strings      |
| `claims`         | array of `Claim`      |

`Claim`:

| Field        | Type                                   |
|--------------|----------------------------------------|
| `claim_id`   | string                                 |
| `model_id`   | string                                 |
| `claim_text` | string                                 |
| `span`       | optional `{start: int, end: int}`      |

**Fallback:** On model load failure, uses sentence-split extraction.

---

### 2. embed_claims

**Request:**

| Field            | Type                              | Required | Default                    |
|------------------|-----------------------------------|----------|----------------------------|
| `schema_version` | string                            | no       | `"1.0"`                    |
| `analysis_id`    | string                            | yes      | --                         |
| `claims`         | array of `ClaimInput` (min 1)     | yes      | --                         |
| `model_name`     | string                            | no       | `"BAAI/bge-large-en-v1.5"` |

`ClaimInput`:

| Field        | Type   | Required |
|--------------|--------|----------|
| `claim_id`   | string | yes      |
| `claim_text` | string | yes      |

**Response:**

| Field            | Type                          |
|------------------|-------------------------------|
| `schema_version` | string                        |
| `analysis_id`    | string                        |
| `warnings`       | array of strings              |
| `vectors`        | dict: claim_id -> float array |
| `dim`            | int (embedding dimension)     |

**Fallback:** Returns empty vectors with warning.

---

### 3. cluster_claims

**Request:**

| Field            | Type                                    | Required | Default |
|------------------|-----------------------------------------|----------|---------|
| `schema_version` | string                                  | no       | `"1.0"` |
| `analysis_id`    | string                                  | yes      | --      |
| `vectors`        | dict: claim_id -> float array (min 1)   | yes      | --      |
| `claims`         | dict: claim_id -> `ClaimMetadata` (min 1) | yes    | --      |
| `sim_threshold`  | float, range (0.0, 1.0]                 | no       | `0.85`  |

`ClaimMetadata`:

| Field        | Type   | Required |
|--------------|--------|----------|
| `model_id`   | string | yes      |
| `claim_text` | string | yes      |

**Response:**

| Field            | Type                 |
|------------------|----------------------|
| `schema_version` | string               |
| `analysis_id`    | string               |
| `warnings`       | array of strings     |
| `clusters`       | array of `Cluster`   |

`Cluster`:

| Field                    | Type             |
|--------------------------|------------------|
| `cluster_id`             | string           |
| `representative_claim_id`| string           |
| `representative_text`    | string           |
| `claim_ids`              | array of strings |

**Fallback:** All claims placed in a single cluster.

---

### 4. rerank_evidence_batch

**Request:**

| Field            | Type                              | Required | Default                                    |
|------------------|-----------------------------------|----------|--------------------------------------------|
| `schema_version` | string                            | no       | `"1.0"`                                    |
| `analysis_id`    | string                            | yes      | --                                         |
| `items`          | array of `RerankItem` (min 1)     | yes      | --                                         |
| `reranker_model` | string                            | no       | `"cross-encoder/ms-marco-MiniLM-L-6-v2"`  |
| `top_k`          | int, range [1, 100]               | no       | `10`                                       |

`RerankItem`:

| Field        | Type                               | Required |
|--------------|------------------------------------|----------|
| `claim_id`   | string                             | yes      |
| `claim_text` | string                             | yes      |
| `passages`   | array of `PassageInput` (min 1)    | yes      |

`PassageInput`:

| Field        | Type   | Required |
|--------------|--------|----------|
| `passage_id` | string | yes      |
| `text`       | string | yes      |

**Response:**

| Field            | Type                       |
|------------------|----------------------------|
| `schema_version` | string                     |
| `analysis_id`    | string                     |
| `warnings`       | array of strings           |
| `rankings`       | array of `ClaimRanking`    |

`ClaimRanking`:

| Field                | Type                       |
|----------------------|----------------------------|
| `claim_id`           | string                     |
| `ordered_passage_ids`| array of strings           |
| `scores`             | dict: passage_id -> float  |

**Fallback:** Returns original passage order with zero scores.

---

### 5. nli_verify_batch

**Request:**

| Field            | Type                              | Required | Default                                         |
|------------------|-----------------------------------|----------|-------------------------------------------------|
| `schema_version` | string                            | no       | `"1.0"`                                         |
| `analysis_id`    | string                            | yes      | --                                              |
| `pairs`          | array of `NliPairInput` (min 1)   | yes      | --                                              |
| `nli_model`      | string                            | no       | `"MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli"` |
| `batch_size`     | int, range [1, 256]               | no       | `16`                                            |

`NliPairInput`:

| Field          | Type   | Required |
|----------------|--------|----------|
| `pair_id`      | string | yes      |
| `claim_id`     | string | yes      |
| `passage_id`   | string | yes      |
| `claim_text`   | string | yes      |
| `passage_text` | string | yes      |

**Response:**

| Field            | Type                        |
|------------------|-----------------------------|
| `schema_version` | string                      |
| `analysis_id`    | string                      |
| `warnings`       | array of strings            |
| `results`        | array of `NliResultOutput`  |

`NliResultOutput`:

| Field        | Type                                                    |
|--------------|---------------------------------------------------------|
| `pair_id`    | string                                                  |
| `claim_id`   | string                                                  |
| `passage_id` | string                                                  |
| `label`      | string                                                  |
| `probs`      | dict: `"entailment"` / `"contradiction"` / `"neutral"` -> float |

**Fallback:** Returns `label="neutral"`, probs near-uniform (`neutral=0.34`, others `0.33`).

---

### 6. compute_umap

**Request:**

| Field            | Type                                  | Required | Default |
|------------------|---------------------------------------|----------|---------|
| `schema_version` | string                                | no       | `"1.0"` |
| `analysis_id`    | string                                | yes      | --      |
| `vectors`        | dict: claim_id -> float array (min 1) | yes      | --      |
| `random_state`   | int                                   | no       | `42`    |
| `n_neighbors`    | int, range [2, 200]                   | no       | `15`    |
| `min_dist`       | float, range (0.0, 1.0]              | no       | `0.1`   |

**Response:**

| Field            | Type                                 |
|------------------|--------------------------------------|
| `schema_version` | string                               |
| `analysis_id`    | string                               |
| `warnings`       | array of strings                     |
| `coords3d`       | dict: claim_id -> `[x, y, z]` floats |

**Fallback:** Returns `[0.0, 0.0, 0.0]` for all vectors.

---

### 7. score_clusters

**Request:**

| Field                | Type                               | Required | Default  |
|----------------------|------------------------------------|----------|----------|
| `schema_version`     | string                             | no       | `"1.0"`  |
| `analysis_id`        | string                             | yes      | --       |
| `clusters`           | array of `Cluster` (min 1)         | yes      | --       |
| `claims`             | dict: claim_id -> `ClaimMetadata` (min 1) | yes | --       |
| `nli_results`        | array of `NliResultOutput`         | no       | `[]`     |
| `weights`            | `ScoringWeights`                   | no       | see below|
| `verdict_thresholds` | `VerdictThresholds`                | no       | see below|

`ScoringWeights`:

| Field                | Type                  | Default |
|----------------------|-----------------------|---------|
| `agreement_weight`   | float, range [0.0, 1.0] | `0.4`  |
| `verification_weight`| float, range [0.0, 1.0] | `0.6`  |

`VerdictThresholds`:

| Field        | Type                  | Default |
|--------------|-----------------------|---------|
| `safe_min`   | int, range [0, 100]   | `75`    |
| `caution_min`| int, range [0, 100]   | `45`    |

**Response:**

| Field            | Type                       |
|------------------|----------------------------|
| `schema_version` | string                     |
| `analysis_id`    | string                     |
| `warnings`       | array of strings           |
| `scores`         | array of `ClusterScore`    |

`ClusterScore`:

| Field          | Type                                                        |
|----------------|-------------------------------------------------------------|
| `cluster_id`   | string                                                      |
| `trust_score`  | int (0-100)                                                 |
| `verdict`      | `"SAFE"` / `"CAUTION"` / `"REJECT"`                        |
| `agreement`    | `{models_supporting: string[], count: int}`                 |
| `verification` | `{best_entailment_prob: float, best_contradiction_prob: float, evidence_passage_id: string}` |

**Fallback:** Returns warning response with empty scores.

---

## Deterministic ID Generation

All IDs are SHA1 hex digests with prefixes:

| ID Type      | Prefix  | Input                                        |
|--------------|---------|----------------------------------------------|
| `claim_id`   | `c_`    | `SHA1("{analysis_id}:{model_id}:{claim_text}")` |
| `cluster_id` | `cl_`   | `SHA1(sorted claim_ids joined by "\|")`       |
| `pair_id`    | `nli_`  | `SHA1("{claim_id}:{passage_id}")`             |

---

## Scoring Algorithm

```
agreement_score  = 100 * (supporting_model_count / total_models)
verification_score = 100 * best_entailment - 100 * best_contradiction
trust_score = round(clamp(0.4 * agreement + 0.6 * clamp(verification, 0, 100), 0, 100))
```

**Verdict thresholds (defaults):**

| Verdict     | Condition                                              |
|-------------|--------------------------------------------------------|
| **SAFE**    | `trust_score >= 75` AND `best_contradiction_prob <= 0.2` |
| **CAUTION** | `trust_score >= 45`                                    |
| **REJECT**  | otherwise                                              |

---

## Error Handling

- All endpoints return a `warnings` array (may be empty).
- On validation failure: returns response with empty data and a warning message.
- On model/inference failure: uses fallback logic (documented per-endpoint above) and adds a warning.
- **HTTP 401**: Invalid or missing bearer token.
- **HTTP 500**: Server-side secret not configured.

---

## Limits

| Constant       | Value  | Scope                      |
|----------------|--------|----------------------------|
| MAX_RESPONSES  | 10     | `extract_claims`           |
| MAX_ITEMS      | 100    | `rerank_evidence_batch`    |
| MAX_PAIRS      | 5000   | `nli_verify_batch`         |
| MAX_VECTORS    | 10000  | `embed_claims`, `compute_umap` |
| MAX_CLUSTERS   | 1000   | `score_clusters`           |

---

## Deployment

```bash
modal deploy modal_app.py
```
