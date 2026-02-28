# 002 — Spec-Alignment Architecture

**Date:** 2026-02-27
**Branch for this doc:** `feat/spec-alignment-docs`
**Branch for implementation:** `feat/spec-alignment-v1`
**Spec reference:** `/BACKEND-ML-PIPELINE.md` (TruthLens Integration Spec v1)

---

## Section 1: What Has Been Done (Current State)

The initial scaffold (see `001_initial_scaffold.md`) established:

- **5 Modal functions:** `embed_claims`, `rerank_evidence`, `nli_verify`, `cluster_claims`, `compute_umap`
- **12 Pydantic models** in `schemas.py`: `EmbedClaimsRequest`, `EmbedClaimsResponse`, `ClusterClaimsRequest`, `ClusterClaimsResponse`, `RankedPassage`, `RerankEvidenceRequest`, `RerankEvidenceResponse`, `NliPair`, `NliResult`, `NliVerifyRequest`, `NliVerifyResponse`, `UmapPoint`, `ComputeUmapRequest`, `ComputeUmapResponse`
- **86 tests** across 4 test files (all passing):
  - `tests/test_schemas.py` — 35 tests
  - `tests/test_batch_utils.py` — 17 tests
  - `tests/test_fallback_utils.py` — 7 tests
  - `tests/test_modal_functions.py` — 27 tests
- **Reusable utilities:**
  - `batch_utils.py` — `chunk_list()`, `flatten_batch_results()` with bounded loops
  - `fallback_utils.py` — `build_error_response()` generic error constructor
- **Architecture patterns:**
  - Dict-in/dict-out at Modal boundary; Pydantic validation inside functions
  - Two container images: `gpu_image` (torch/transformers/sentence-transformers), `cpu_image` (sklearn/umap)
  - Shared Modal Volume at `/models` with HF cache env vars
  - Custom `_softmax()` helper (bounded loops, no recursion)
- **Test harness:** `test_harness.py` — local integration test with fake model outputs

---

## Section 2: Full Gap Analysis (Current vs Spec)

### 2.1 Missing Functions (2)

| Function | Spec Section | Purpose | Compute |
|----------|-------------|---------|---------|
| `extract_claims` | 5.1 | Decompose model responses into atomic claims using locally-run Llama on Modal GPU, with sentence-split fallback | GPU A10G |
| `score_clusters` | 5.7 | Trust scoring + verdict per cluster using agreement + best NLI | CPU |

### 2.2 Renamed Functions (2)

| Current Name | Spec Name | Reason |
|-------------|-----------|--------|
| `rerank_evidence` | `rerank_evidence_batch` | Spec accepts batch of items (multiple claims), not single claim |
| `nli_verify` | `nli_verify_batch` | Spec name matches batch semantics |

### 2.3 Cross-Cutting Changes (Apply to ALL Functions)

1. **`schema_version` and `analysis_id` on every request/response:**
   - Current: No `schema_version` or `analysis_id` on any model
   - Spec: Every request has `schema_version: "1.0"` and `analysis_id: str`; every response has both plus `warnings: List[str]`

2. **`error: Optional[str]` replaced by `warnings: List[str]`:**
   - Current: All responses have `error: Optional[str] = None`
   - Spec: All responses have `warnings: List[str] = []` (never a single error; collect warnings)

3. **Vectors change from positional lists to keyed dicts:**
   - Current: `vectors: List[List[float]]` (position = claim index)
   - Spec: `vectors: Dict[str, List[float]]` (key = `claim_id`, value = vector)
   - Affects: `embed_claims`, `cluster_claims`, `compute_umap`

4. **First-class IDs throughout:**
   - `claim_id` (prefix `c_`), `passage_id` (prefix `p_`), `pair_id` (prefix `nli_`), `cluster_id` (prefix `cl_`)
   - Generated deterministically via SHA1 (see Section 6: `id_utils.py`)

### 2.4 Per-Function Schema Diffs

#### `embed_claims` (spec 5.2)

| Field | Current | Spec |
|-------|---------|------|
| Request `claim_texts: List[str]` | Present | **Removed** |
| Request `claims: List[ClaimInput]` | Missing | **Added** — each has `claim_id` and `claim_text` |
| Request `model_name: str` | Missing | **Added** — optional override, default `"BAAI/bge-large-en-v1.5"` |
| Response `vectors: List[List[float]]` | Present | **Changed** to `Dict[str, List[float]]` keyed by `claim_id` |
| Response `dimension: int` | Present | **Renamed** to `dim: int` |
| Response `model_name: str` | Present | **Removed** from response (not in spec response) |
| Response `error: Optional[str]` | Present | **Replaced** by `warnings: List[str]` |

#### `cluster_claims` (spec 5.3)

| Field | Current | Spec |
|-------|---------|------|
| Request `vectors: List[List[float]]` | Present | **Changed** to `Dict[str, List[float]]` keyed by `claim_id` |
| Request `claims: Dict[str, ClaimMetadata]` | Missing | **Added** — maps `claim_id` to `{model_id, claim_text}` |
| Request `threshold: float` (default 0.5) | Present | **Renamed** to `sim_threshold: float` (default 0.85) |
| Response `clusters: List[List[int]]` | Present | **Changed** to `List[Cluster]` — structured objects with `cluster_id`, `claim_ids`, `representative_claim_id`, `representative_text` |
| Response `num_clusters: int` | Present | **Removed** (derivable from `len(clusters)`) |
| Response `error: Optional[str]` | Present | **Replaced** by `warnings: List[str]` |

**CRITICAL:** `sim_threshold` is a similarity threshold (0.85 = "85% similar"). AgglomerativeClustering uses distance, so: `distance_threshold = 1.0 - sim_threshold`. Default: `1.0 - 0.85 = 0.15`.

#### `rerank_evidence` → `rerank_evidence_batch` (spec 5.4)

| Field | Current | Spec |
|-------|---------|------|
| Request `claim: str` | Present | **Removed** (single claim) |
| Request `passages: List[str]` | Present | **Removed** (plain strings) |
| Request `items: List[RerankItem]` | Missing | **Added** — batch of `{claim_id, claim_text, passages: List[PassageInput]}` |
| Request `reranker_model: str` | Missing | **Added** — default `"cross-encoder/ms-marco-MiniLM-L-6-v2"` |
| Response `ranked_passages: List[RankedPassage]` | Present | **Removed** |
| Response `rankings: List[ClaimRanking]` | Missing | **Added** — `{claim_id, ordered_passage_ids: List[str], scores: Dict[str, float]}` |
| Response `error: Optional[str]` | Present | **Replaced** by `warnings: List[str]` |

#### `nli_verify` → `nli_verify_batch` (spec 5.5)

| Field | Current | Spec |
|-------|---------|------|
| Request `pairs: List[NliPair]` | Present | **Changed** type from `NliPair(premise, hypothesis)` to `NliPairInput(pair_id, claim_id, passage_id, claim_text, passage_text)` |
| Request `nli_model: str` | Missing | **Added** — default `"MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli"` |
| Request `batch_size: int` | Present | **Keep** (internal batching param) |
| Response `results: List[NliResult]` | Present | **Changed** type from `NliResult(label, scores)` to `NliResultOutput(pair_id, claim_id, passage_id, label, probs)` |
| Response `error: Optional[str]` | Present | **Replaced** by `warnings: List[str]` |

**NLI Model Change:** `microsoft/deberta-large-mnli` → `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli`. The new model may have a different label order in `model.config.id2label`. **Do NOT hardcode label order.** Read from `model.config.id2label` at runtime.

#### `compute_umap` (spec 5.6)

| Field | Current | Spec |
|-------|---------|------|
| Request `vectors: List[List[float]]` | Present | **Changed** to `Dict[str, List[float]]` keyed by `claim_id` |
| Request `random_state: int` | Missing | **Added** — default `42` (reproducibility) |
| Response `coords_3d: List[UmapPoint]` | Present | **Changed** to `coords3d: Dict[str, List[float]]` (key = `claim_id`, value = `[x, y, z]`) |
| Response `error: Optional[str]` | Present | **Replaced** by `warnings: List[str]` |

### 2.5 Cache Env Var Diff

| Var | Current | Spec |
|-----|---------|------|
| `HF_HOME` | `/models/hf` | `/models/hf` (match) |
| `TRANSFORMERS_CACHE` | `/models/transformers` | `/models/hf` (**mismatch — change to `/models/hf`**) |
| `SENTENCE_TRANSFORMERS_HOME` | `/models/sentence_transformers` | `/models/st` (**mismatch — change to `/models/st`**) |

---

## Section 3: Target File Structure

```
ml/
  schemas.py                    -- REWRITE: ~30 Pydantic models (replacing 12)
  modal_app.py                  -- REWRITE: 7 functions (was 5), rename 2
  batch_utils.py                -- KEEP AS-IS (17 tests passing, no spec changes)
  fallback_utils.py             -- REWRITE: build_warning_response() replaces build_error_response()
  claim_extraction.py           -- NEW: sentence_split_claims() helper
  scoring.py                    -- NEW: trust score algorithm (6 pure functions)
  id_utils.py                   -- NEW: deterministic SHA1 ID generation
  test_harness.py               -- REWRITE: 7-function pipeline
  requirements.txt              -- UPDATE: add transformers version pin (already present)
  tests/
    __init__.py                 -- KEEP
    test_schemas.py             -- REWRITE: ~60 tests for ~30 models
    test_batch_utils.py         -- KEEP AS-IS: 17 tests
    test_fallback_utils.py      -- REWRITE: ~8 tests for new warning pattern
    test_modal_functions.py     -- REWRITE: ~40 tests for 7 functions
    test_claim_extraction.py    -- NEW: ~10 tests
    test_scoring.py             -- NEW: ~15 tests
    test_id_utils.py            -- NEW: ~8 tests
  ClaudeProgress/
    001_initial_scaffold.md     -- existing (unchanged)
    002_spec_alignment_architecture.md -- THIS DOC
```

---

## Section 4: Complete Schema Specifications

### 4.1 Base Classes (2 models)

```python
class BaseRequest(BaseModel):
    schema_version: str = Field(default="1.0")
    analysis_id: str = Field(..., min_length=1)

class BaseResponse(BaseModel):
    schema_version: str = Field(default="1.0")
    analysis_id: str = Field(default="")
    warnings: List[str] = Field(default_factory=list)
```

### 4.2 extract_claims Models (5 models)

```python
class ModelResponse(BaseModel):
    """A single LLM response to decompose into claims."""
    model_id: str = Field(..., min_length=1)
    response_text: str = Field(..., min_length=1)

class ClaimSpan(BaseModel):
    """Character offsets into the original response_text."""
    start: int = Field(..., ge=0)
    end: int = Field(..., ge=0)

class Claim(BaseModel):
    """An atomic factual claim extracted from a model response."""
    claim_id: str = Field(..., min_length=1)
    model_id: str = Field(..., min_length=1)
    claim_text: str = Field(..., min_length=1)
    span: Optional[ClaimSpan] = Field(default=None)

class ExtractClaimsRequest(BaseRequest):
    responses: List[ModelResponse] = Field(..., min_length=1)

class ExtractClaimsResponse(BaseResponse):
    claims: List[Claim] = Field(default_factory=list)
```

### 4.3 embed_claims Models (3 models)

```python
class ClaimInput(BaseModel):
    """Claim ID + text pair for embedding."""
    claim_id: str = Field(..., min_length=1)
    claim_text: str = Field(..., min_length=1)

class EmbedClaimsRequest(BaseRequest):
    claims: List[ClaimInput] = Field(..., min_length=1)
    model_name: str = Field(default="BAAI/bge-large-en-v1.5")

class EmbedClaimsResponse(BaseResponse):
    vectors: Dict[str, List[float]] = Field(default_factory=dict)
    dim: int = Field(default=0)
```

### 4.4 cluster_claims Models (4 models)

```python
class ClaimMetadata(BaseModel):
    """Metadata for a claim used during clustering."""
    model_id: str = Field(..., min_length=1)
    claim_text: str = Field(..., min_length=1)

class Cluster(BaseModel):
    """A group of semantically similar claims."""
    cluster_id: str = Field(..., min_length=1)
    claim_ids: List[str] = Field(..., min_length=1)
    representative_claim_id: str = Field(..., min_length=1)
    representative_text: str = Field(..., min_length=1)

class ClusterClaimsRequest(BaseRequest):
    vectors: Dict[str, List[float]] = Field(..., min_length=1)
    claims: Dict[str, ClaimMetadata] = Field(..., min_length=1)
    sim_threshold: float = Field(default=0.85, gt=0.0, le=1.0)

class ClusterClaimsResponse(BaseResponse):
    clusters: List[Cluster] = Field(default_factory=list)
```

### 4.5 rerank_evidence_batch Models (5 models)

```python
class PassageInput(BaseModel):
    """A candidate evidence passage."""
    passage_id: str = Field(..., min_length=1)
    text: str = Field(..., min_length=1)

class RerankItem(BaseModel):
    """A single claim with its candidate passages for reranking."""
    claim_id: str = Field(..., min_length=1)
    claim_text: str = Field(..., min_length=1)
    passages: List[PassageInput] = Field(..., min_length=1)

class ClaimRanking(BaseModel):
    """Reranked result for a single claim."""
    claim_id: str = Field(..., min_length=1)
    ordered_passage_ids: List[str] = Field(default_factory=list)
    scores: Dict[str, float] = Field(default_factory=dict)

class RerankEvidenceBatchRequest(BaseRequest):
    items: List[RerankItem] = Field(..., min_length=1)
    reranker_model: str = Field(default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    top_k: int = Field(default=10, ge=1, le=100)

class RerankEvidenceBatchResponse(BaseResponse):
    rankings: List[ClaimRanking] = Field(default_factory=list)
```

### 4.6 nli_verify_batch Models (4 models)

```python
class NliPairInput(BaseModel):
    """A (claim, passage) pair for NLI classification."""
    pair_id: str = Field(..., min_length=1)
    claim_id: str = Field(..., min_length=1)
    passage_id: str = Field(..., min_length=1)
    claim_text: str = Field(..., min_length=1)
    passage_text: str = Field(..., min_length=1)

class NliResultOutput(BaseModel):
    """NLI prediction for a single pair."""
    pair_id: str = Field(..., min_length=1)
    claim_id: str = Field(..., min_length=1)
    passage_id: str = Field(..., min_length=1)
    label: str = Field(default="neutral")
    probs: Dict[str, float] = Field(default_factory=dict)

class NliVerifyBatchRequest(BaseRequest):
    pairs: List[NliPairInput] = Field(..., min_length=1)
    nli_model: str = Field(default="MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli")
    batch_size: int = Field(default=16, ge=1, le=256)

class NliVerifyBatchResponse(BaseResponse):
    results: List[NliResultOutput] = Field(default_factory=list)
```

### 4.7 compute_umap Models (2 models)

```python
class ComputeUmapRequest(BaseRequest):
    vectors: Dict[str, List[float]] = Field(..., min_length=1)
    random_state: int = Field(default=42)
    n_neighbors: int = Field(default=15, ge=2, le=200)
    min_dist: float = Field(default=0.1, gt=0.0, le=1.0)

class ComputeUmapResponse(BaseResponse):
    coords3d: Dict[str, List[float]] = Field(default_factory=dict)
```

### 4.8 score_clusters Models (7 models)

```python
class ScoringWeights(BaseModel):
    """Weights for agreement vs verification in trust score."""
    agreement_weight: float = Field(default=0.4, ge=0.0, le=1.0)
    verification_weight: float = Field(default=0.6, ge=0.0, le=1.0)

class VerdictThresholds(BaseModel):
    """Thresholds for verdict classification."""
    safe_min: int = Field(default=75, ge=0, le=100)
    caution_min: int = Field(default=45, ge=0, le=100)

class AgreementDetail(BaseModel):
    """Agreement info for a cluster score."""
    models_supporting: List[str] = Field(default_factory=list)
    count: int = Field(default=0, ge=0)

class VerificationDetail(BaseModel):
    """NLI verification info for a cluster score."""
    best_entailment_prob: float = Field(default=0.0, ge=0.0, le=1.0)
    best_contradiction_prob: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence_passage_id: str = Field(default="")

class ClusterScore(BaseModel):
    """Trust score + verdict for a single cluster."""
    cluster_id: str = Field(..., min_length=1)
    trust_score: int = Field(default=0, ge=0, le=100)
    verdict: str = Field(default="REJECT")
    agreement: AgreementDetail = Field(default_factory=AgreementDetail)
    verification: VerificationDetail = Field(default_factory=VerificationDetail)

class ScoreClustersRequest(BaseRequest):
    clusters: List[Cluster] = Field(..., min_length=1)
    claims: Dict[str, ClaimMetadata] = Field(..., min_length=1)
    nli_results: List[NliResultOutput] = Field(default_factory=list)
    weights: ScoringWeights = Field(default_factory=ScoringWeights)
    verdict_thresholds: VerdictThresholds = Field(default_factory=VerdictThresholds)

class ScoreClustersResponse(BaseResponse):
    scores: List[ClusterScore] = Field(default_factory=list)
```

### 4.9 Model Count Summary

| Category | Models | Total |
|----------|--------|-------|
| Base classes | `BaseRequest`, `BaseResponse` | 2 |
| extract_claims | `ModelResponse`, `ClaimSpan`, `Claim`, `ExtractClaimsRequest`, `ExtractClaimsResponse` | 5 |
| embed_claims | `ClaimInput`, `EmbedClaimsRequest`, `EmbedClaimsResponse` | 3 |
| cluster_claims | `ClaimMetadata`, `Cluster`, `ClusterClaimsRequest`, `ClusterClaimsResponse` | 4 |
| rerank_evidence_batch | `PassageInput`, `RerankItem`, `ClaimRanking`, `RerankEvidenceBatchRequest`, `RerankEvidenceBatchResponse` | 5 |
| nli_verify_batch | `NliPairInput`, `NliResultOutput`, `NliVerifyBatchRequest`, `NliVerifyBatchResponse` | 4 |
| compute_umap | `ComputeUmapRequest`, `ComputeUmapResponse` | 2 |
| score_clusters | `ScoringWeights`, `VerdictThresholds`, `AgreementDetail`, `VerificationDetail`, `ClusterScore`, `ScoreClustersRequest`, `ScoreClustersResponse` | 7 |
| **TOTAL** | | **32** |

**Models removed (no longer needed):** `RankedPassage`, `NliPair`, `NliResult`, `UmapPoint`

---

## Section 5: Function Implementation Specs

### 5.1 `extract_claims` (NEW — spec 5.1)

**Decorator:**
```python
@app.function(
    image=gpu_image,       # reuse existing gpu_image (already has transformers)
    gpu="A10G",
    memory=16384,
    timeout=600,
    volumes={VOLUME_MOUNT: model_volume},
)
```

**Algorithm:**
1. Validate `ExtractClaimsRequest` from payload dict
2. Load Llama model (`meta-llama/Llama-3.1-8B-Instruct`) via `transformers.AutoModelForCausalLM` + `AutoTokenizer`
3. For each `ModelResponse` in `req.responses` (bounded loop, max 10 responses):
   - Build prompt instructing Llama to return a JSON array of atomic claim strings
   - Parse JSON response into list of claim strings
   - For each claim string, generate `claim_id` via `id_utils.make_claim_id(analysis_id, model_id, claim_text)`
   - Create `Claim` objects with `claim_id`, `model_id`, `claim_text`, `span=None`
4. Return `ExtractClaimsResponse` with all claims

**Fallback:** If Llama fails (load error, parse error, timeout), call `sentence_split_claims(response_text)` from `claim_extraction.py`. Append warning: `"Llama extraction failed for {model_id}, using sentence-split fallback: {error}"`

**System prompt for Llama (suggested):**
```
You are a claim extractor. Given a text, extract all atomic factual claims as a JSON array of strings. Each claim should be a single, verifiable statement. Return ONLY the JSON array, no other text.
```

**Model caching:** Model is cached on Modal Volume at `/models/hf`. The `HF_HOME` env var handles this.

**Container:** Reuses `gpu_image` — it already has `transformers` and `torch`. No new image needed.

### 5.2 `embed_claims` (REWRITE — spec 5.2)

**Decorator:** Same as current (GPU A10G, 16384MB, 600s timeout).

**Changes from current:**
1. Accept `EmbedClaimsRequest` (has `claims: List[ClaimInput]` instead of `claim_texts: List[str]`)
2. Optionally use `req.model_name` override (default `"BAAI/bge-large-en-v1.5"`)
3. Extract `claim_texts` from `ClaimInput` objects for encoding
4. Build response `vectors` as `Dict[str, List[float]]` keyed by `claim_id`
5. Return `dim` instead of `dimension`
6. Return `warnings` instead of `error`

**Core logic** (simplified):
```
claim_ids = [c.claim_id for c in req.claims]
claim_texts = [c.claim_text for c in req.claims]
model = SentenceTransformer(req.model_name)
embeddings = model.encode(claim_texts, batch_size=64, normalize_embeddings=True)
vectors = {claim_ids[i]: embeddings[i].tolist() for i in range(len(claim_ids))}
```

### 5.3 `cluster_claims` (REWRITE — spec 5.3)

**Decorator:** Same as current (CPU 4, 8192MB, 300s timeout).

**Changes from current:**
1. Accept `ClusterClaimsRequest` with `vectors: Dict[str, List[float]]`, `claims: Dict[str, ClaimMetadata]`, `sim_threshold: float`
2. Convert `sim_threshold` to `distance_threshold`: `distance_threshold = 1.0 - req.sim_threshold`
3. Extract ordered `claim_ids` and corresponding vectors for sklearn
4. Run `AgglomerativeClustering(n_clusters=None, distance_threshold=distance_threshold, metric="cosine", linkage="average")`
5. Build `Cluster` objects with:
   - `cluster_id` via `id_utils.make_cluster_id(claim_ids_in_cluster)`
   - `claim_ids` list
   - `representative_claim_id` = first claim_id in cluster (or pick by frequency)
   - `representative_text` = `claims[representative_claim_id].claim_text`
6. Return `ClusterClaimsResponse` with structured clusters

**Fallback:** If clustering fails, put all claims in one cluster. Append warning.

### 5.4 `rerank_evidence_batch` (REWRITE + RENAME — spec 5.4)

**Decorator:** Same GPU config as current `rerank_evidence`.

**Changes from current:**
1. Function renamed from `rerank_evidence` to `rerank_evidence_batch`
2. Accept `RerankEvidenceBatchRequest` with `items: List[RerankItem]` (batch of claims)
3. For each `RerankItem` (bounded loop):
   - Build `(claim_text, passage.text)` pairs for cross-encoder
   - Score all pairs, sort descending
   - Take top_k
   - Build `ClaimRanking` with `ordered_passage_ids` and `scores` dict
4. Return `RerankEvidenceBatchResponse` with `rankings: List[ClaimRanking]`

**Fallback:** If rerank fails for an item, return original passage ordering with scores all 0.0. Append warning.

### 5.5 `nli_verify_batch` (REWRITE + RENAME — spec 5.5)

**Decorator:** GPU A10G, 24576MB memory, 900s timeout (same as current).

**Changes from current:**
1. Function renamed from `nli_verify` to `nli_verify_batch`
2. **Model change:** `microsoft/deberta-large-mnli` → `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli`
3. Accept `NliVerifyBatchRequest` with `pairs: List[NliPairInput]`
4. **Read label order from model config at runtime:**
   ```python
   id2label = model.config.id2label  # e.g. {0: "entailment", 1: "neutral", 2: "contradiction"}
   ```
   Do NOT hardcode `NLI_LABELS = ["contradiction", "neutral", "entailment"]`.
5. For each pair, build `NliResultOutput` with `pair_id`, `claim_id`, `passage_id`, `label`, `probs` (dict keyed by label name)
6. Return `NliVerifyBatchResponse`

**Fallback:** Return neutral with probs `{"entailment": 0.34, "contradiction": 0.33, "neutral": 0.33}` for all pairs. Append warning.

### 5.6 `compute_umap` (REWRITE — spec 5.6)

**Decorator:** Same as current (CPU 8, 16384MB, 600s timeout).

**Changes from current:**
1. Accept `ComputeUmapRequest` with `vectors: Dict[str, List[float]]` and `random_state: int = 42`
2. Extract ordered `claim_ids` and vectors
3. Pass `random_state=req.random_state` to `umap.UMAP()`
4. Build response `coords3d: Dict[str, List[float]]` keyed by `claim_id` (value = `[x, y, z]`)
5. Return `ComputeUmapResponse`

**Fallback:** Return zeros `{claim_id: [0.0, 0.0, 0.0]}` for all claims. Append warning.

### 5.7 `score_clusters` (NEW — spec 5.7)

**Decorator:**
```python
@app.function(
    image=cpu_image,
    cpu=4,
    memory=8192,
    timeout=300,
)
```

**No ML models loaded.** Pure CPU math. Delegates to `scoring.py` helpers.

**Algorithm:**
For each cluster in `req.clusters` (bounded loop):
1. `supporting_models = find_supporting_models(cluster.claim_ids, req.claims)` — unique model_ids that contributed claims to this cluster
2. `agreement_score = compute_agreement_score(len(supporting_models), total_models=5)`
3. `best_entailment, best_contradiction, evidence_passage_id = find_best_nli_for_cluster(cluster.claim_ids, req.nli_results)`
4. `verification_score = compute_verification_score(best_entailment, best_contradiction)`
5. `trust_score = compute_trust_score(agreement_score, verification_score, weights)`
6. `verdict = determine_verdict(trust_score, best_contradiction, thresholds)`
7. Build `ClusterScore` object

Return `ScoreClustersResponse` with all scores.

---

## Section 6: New Utility Files Specs

### 6.1 `id_utils.py` — Deterministic ID Generation

Three public functions. All use SHA1 hashing for deterministic, reproducible IDs.

```python
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
```

**No loops, no recursion, no pointers.** Each function is a pure hash computation.

### 6.2 `claim_extraction.py` — Sentence Split Fallback

```python
MAX_SENTENCES = 500

def sentence_split_claims(text: str) -> list:
    """Split text into sentence-level claims using period/question/exclamation delimiters.

    Returns a list of stripped, non-empty sentence strings.
    Bounded to MAX_SENTENCES iterations.
    """
```

**Algorithm:**
1. Replace `? ` and `! ` with `.\n` to normalize delimiters
2. Split on `.`
3. Iterate through splits (bounded loop, max `MAX_SENTENCES`):
   - Strip whitespace
   - Skip empty strings
   - Append to results
4. Return list of sentence strings

**No recursion. Single bounded loop. No function pointers.**

### 6.3 `scoring.py` — Trust Score Algorithm (6 Pure Functions)

```python
TOTAL_MODELS = 5  # Number of LLM providers in the arena

def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp value to [min_val, max_val] range."""
    if value < min_val:
        return min_val
    if value > max_val:
        return max_val
    return value

def compute_agreement_score(supporting_count: int, total_models: int) -> float:
    """agreement_score = 100 * (supporting_count / total_models)"""
    if total_models <= 0:
        return 0.0
    return 100.0 * (supporting_count / total_models)

def compute_verification_score(best_entailment: float, best_contradiction: float) -> float:
    """verification_score = 100 * best_entailment - 100 * best_contradiction"""
    return 100.0 * best_entailment - 100.0 * best_contradiction

def compute_trust_score(
    agreement_score: float,
    verification_score: float,
    agreement_weight: float,
    verification_weight: float,
) -> int:
    """trust_score = round(aw * agreement + vw * clamp(verification, 0, 100))"""
    clamped_verification = clamp(verification_score, 0.0, 100.0)
    raw = agreement_weight * agreement_score + verification_weight * clamped_verification
    return round(raw)

def determine_verdict(
    trust_score: int,
    best_contradiction_prob: float,
    safe_min: int,
    caution_min: int,
) -> str:
    """
    SAFE:    trust_score >= safe_min AND best_contradiction_prob <= 0.2
    CAUTION: trust_score >= caution_min
    REJECT:  otherwise
    """
    if trust_score >= safe_min and best_contradiction_prob <= 0.2:
        return "SAFE"
    if trust_score >= caution_min:
        return "CAUTION"
    return "REJECT"

def find_supporting_models(claim_ids: list, claims: dict) -> list:
    """Return list of unique model_ids that contributed claims to this cluster.

    Args:
        claim_ids: List of claim IDs in the cluster
        claims: Dict mapping claim_id -> ClaimMetadata (has model_id field)

    Bounded loop over claim_ids (max 500).
    """
    seen: dict = {}
    models: list = []
    max_iter = 500
    count = 0
    for i in range(len(claim_ids)):
        if count >= max_iter:
            break
        count += 1
        cid = claim_ids[i]
        if cid in claims:
            mid = claims[cid].model_id
            if mid not in seen:
                seen[mid] = True
                models.append(mid)
    return models

def find_best_nli_for_cluster(
    claim_ids: list,
    nli_results: list,
) -> tuple:
    """Find the best entailment and worst contradiction across NLI results for a cluster.

    Returns: (best_entailment_prob, best_contradiction_prob, evidence_passage_id)

    Bounded loops over nli_results (max 5000) and claim_ids.
    """
    best_ent = 0.0
    best_contra = 0.0
    evidence_pid = ""
    claim_set: dict = {}
    for i in range(len(claim_ids)):
        claim_set[claim_ids[i]] = True

    max_iter = 5000
    count = 0
    for i in range(len(nli_results)):
        if count >= max_iter:
            break
        count += 1
        result = nli_results[i]
        if result.claim_id not in claim_set:
            continue
        ent = result.probs.get("entailment", 0.0)
        contra = result.probs.get("contradiction", 0.0)
        if ent > best_ent:
            best_ent = ent
            evidence_pid = result.passage_id
        if contra > best_contra:
            best_contra = contra

    return (best_ent, best_contra, evidence_pid)
```

### 6.4 `fallback_utils.py` — Rewrite

**Replace** `build_error_response(response_class, error_msg)` with:

```python
def build_warning_response(
    response_class: Type[T],
    analysis_id: str,
    warning_msg: str,
) -> T:
    """Construct a default instance of response_class with a warning.

    The response class must accept analysis_id and warnings as keyword arguments.
    All other fields use their declared defaults.
    """
    return response_class(
        analysis_id=analysis_id,
        warnings=[warning_msg],
    )
```

**Key difference:** Takes `analysis_id` (required by all responses now) and sets `warnings` list instead of `error` string.

---

## Section 7: Critical Implementation Notes

### 7.1 Similarity-to-Distance Conversion

The spec uses `sim_threshold` (similarity, 0-1 where 1 = identical). AgglomerativeClustering uses `distance_threshold` (distance, 0-2 for cosine where 0 = identical).

**Conversion:** `distance_threshold = 1.0 - sim_threshold`

Example: `sim_threshold=0.85` → `distance_threshold=0.15`

### 7.2 NLI Label Order — Do NOT Hardcode

The current code hardcodes: `NLI_LABELS = ["contradiction", "neutral", "entailment"]`

The new model (`MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli`) may have a different order. **Read from model config:**

```python
id2label = model.config.id2label
# Then: probs_dict = {id2label[i]: softmax_probs[i] for i in range(len(softmax_probs))}
```

### 7.3 Function Length Rule

Per CLAUDE.md: "No function should be longer than what can be printed on a single sheet of paper in a standard format with one line per statement and one line per declaration." This is approximately 60 lines.

Functions that may need helper extraction:
- `rerank_evidence_batch` — extract `_rerank_single_item()` helper
- `nli_verify_batch` — extract `_run_nli_batch()` helper
- `extract_claims` — extract `_extract_from_single_response()` helper

### 7.4 Container Images

Three images total (though `extract_claims` reuses `gpu_image`):

| Image | Used By | Packages |
|-------|---------|----------|
| `gpu_image` | `embed_claims`, `rerank_evidence_batch`, `nli_verify_batch`, `extract_claims` | torch, transformers, sentence-transformers, pydantic, numpy |
| `cpu_image` | `cluster_claims`, `compute_umap`, `score_clusters` | scikit-learn, umap-learn, pydantic, numpy |

`extract_claims` reuses `gpu_image` because it already has `transformers` and `torch`, which are needed for Llama inference.

### 7.5 Images Need Updated `add_local_python_source`

Both images must add the new utility modules:

```python
.add_local_python_source(
    "schemas", "batch_utils", "fallback_utils",
    "id_utils", "claim_extraction", "scoring"
)
```

### 7.6 Cache Env Var Fix

Update `SHARED_ENV` to match spec:

```python
SHARED_ENV = {
    "HF_HOME": "/models/hf",
    "TRANSFORMERS_CACHE": "/models/hf",              # was /models/transformers
    "SENTENCE_TRANSFORMERS_HOME": "/models/st",      # was /models/sentence_transformers
}
```

### 7.7 Model Name Constants Update

```python
EMBED_MODEL_NAME = "BAAI/bge-large-en-v1.5"                          # unchanged
RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"          # unchanged
NLI_MODEL_NAME = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli"    # CHANGED from microsoft/deberta-large-mnli
LLAMA_MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"               # NEW
```

### 7.8 `_softmax()` Helper

Keep the existing `_softmax()` helper as-is. It has no recursion, uses bounded loops, and is numerically stable. It's used by `nli_verify_batch`.

---

## Section 8: Test Strategy

### 8.1 Test Count Summary (~158 total)

| File | Count | Status | Notes |
|------|-------|--------|-------|
| `test_schemas.py` | ~60 | REWRITE | ~2 tests per model (valid + invalid) for ~30 models |
| `test_batch_utils.py` | 17 | KEEP | No changes needed |
| `test_fallback_utils.py` | ~8 | REWRITE | Test `build_warning_response` for all response types |
| `test_modal_functions.py` | ~40 | REWRITE | ~5-6 tests per function for 7 functions |
| `test_claim_extraction.py` | ~10 | NEW | Sentence splitting edge cases |
| `test_scoring.py` | ~15 | NEW | All 6+1 scoring functions |
| `test_id_utils.py` | ~8 | NEW | Determinism, uniqueness, prefix checks |
| **TOTAL** | **~158** | | |

### 8.2 `test_schemas.py` (~60 tests) — REWRITE

Test each of the ~30 Pydantic models for:
- Valid construction with required fields
- Default values are correct
- Required field validation (missing fields raise)
- Field constraints (min_length, ge, le, etc.)
- Round-trip serialization (`model_dump()` → reconstruct)

Key tests:
- `BaseRequest` requires `analysis_id`
- `BaseResponse` defaults `warnings` to `[]`
- `EmbedClaimsRequest` has `claims: List[ClaimInput]` not `claim_texts`
- `ClusterClaimsRequest` has `sim_threshold` default `0.85`
- `ComputeUmapRequest` has `random_state` default `42`
- `ScoreClustersRequest` accepts complex nested structures
- All request types inherit `schema_version="1.0"` and `analysis_id`
- All response types inherit `warnings: List[str]`

### 8.3 `test_fallback_utils.py` (~8 tests) — REWRITE

Test `build_warning_response()` for each response type:
- `EmbedClaimsResponse` — warning set, vectors empty, dim 0
- `ClusterClaimsResponse` — warning set, clusters empty
- `RerankEvidenceBatchResponse` — warning set, rankings empty
- `NliVerifyBatchResponse` — warning set, results empty
- `ComputeUmapResponse` — warning set, coords3d empty
- `ExtractClaimsResponse` — warning set, claims empty
- `ScoreClustersResponse` — warning set, scores empty
- `analysis_id` preserved in all warning responses

### 8.4 `test_modal_functions.py` (~40 tests) — REWRITE

For each of the 7 functions, test:

**`extract_claims` (~6 tests):**
- Happy path: mocked Llama returns JSON array of claims
- Fallback: Llama fails, sentence-split used, warning appended
- Invalid payload: warning response returned
- Multiple model responses processed
- Claim IDs are deterministic (same input → same ID)
- Empty response_text handling

**`embed_claims` (~5 tests):**
- Happy path: vectors keyed by claim_id, dim correct
- Multiple claims batched correctly
- Invalid payload: warning response
- Model name override works
- schema_version and analysis_id propagated

**`cluster_claims` (~6 tests):**
- Happy path: clusters have IDs, representative_claim_id set
- Single vector: one cluster returned
- sim_threshold conversion: 0.85 → distance 0.15
- Fallback: sklearn fails, single cluster, warning
- Invalid payload: warning response
- claims metadata used for representative_text

**`rerank_evidence_batch` (~6 tests):**
- Happy path: rankings with ordered_passage_ids and scores dict
- Multiple items in batch
- top_k limiting works
- Fallback: original order, warning
- Invalid payload: warning response
- Scores are descending

**`nli_verify_batch` (~6 tests):**
- Happy path: results with pair_id, claim_id, passage_id, probs
- Label order read from model config (not hardcoded)
- Fallback: neutral with near-uniform probs, warning
- Invalid payload: warning response
- Batch size respected
- probs sum to ~1.0

**`compute_umap` (~5 tests):**
- Happy path: coords3d keyed by claim_id, 3 values per key
- random_state passed to UMAP
- Fallback: zeros, warning
- Invalid payload: warning response
- n_neighbors and min_dist respected

**`score_clusters` (~6 tests):**
- Happy path: trust_score, verdict, agreement, verification computed
- SAFE verdict: score >= 75, contradiction <= 0.2
- CAUTION verdict: score >= 45 but not SAFE
- REJECT verdict: score < 45
- No NLI results: verification score is 0
- Multiple clusters scored independently

### 8.5 `test_claim_extraction.py` (~10 tests) — NEW

- Simple sentences split correctly
- Multiple delimiters (period, question mark, exclamation)
- Empty string returns empty list
- Single sentence returns one-element list
- Whitespace-only segments skipped
- MAX_SENTENCES bound respected
- No trailing empty strings from trailing periods
- Unicode text handled
- Very long text (stress test within bound)
- Newlines in text handled

### 8.6 `test_scoring.py` (~15 tests) — NEW

- `clamp` — value below min, above max, within range
- `compute_agreement_score` — 0/5 models, 3/5 models, 5/5 models, 0 total_models edge case
- `compute_verification_score` — high entailment low contradiction, low entailment high contradiction, both zero
- `compute_trust_score` — typical values, edge cases with 0 weights
- `determine_verdict` — SAFE (score 80, contra 0.1), CAUTION (score 50, contra 0.5), REJECT (score 30)
- `determine_verdict` — CAUTION even with high score if contradiction > 0.2
- `find_supporting_models` — multiple models, single model, unknown claim_id skipped
- `find_best_nli_for_cluster` — multiple results, no results, claim not in cluster skipped

### 8.7 `test_id_utils.py` (~8 tests) — NEW

- `make_claim_id` returns string starting with `"c_"`
- `make_claim_id` is deterministic (same inputs → same output)
- `make_claim_id` different inputs → different output
- `make_cluster_id` returns string starting with `"cl_"`
- `make_cluster_id` is order-independent (sorted internally)
- `make_pair_id` returns string starting with `"nli_"`
- `make_pair_id` is deterministic
- All IDs are valid hex after prefix

---

## Section 9: Implementation Order

Steps are dependency-sequenced. Steps marked parallel can be done simultaneously.

```
Step 1:  Create branch feat/spec-alignment-v1 from main
         └─ git checkout -b feat/spec-alignment-v1 main

Step 2:  id_utils.py + tests/test_id_utils.py          ← NO dependencies
Step 3:  claim_extraction.py + tests/test_claim_extraction.py  ← NO dependencies
Step 4:  scoring.py + tests/test_scoring.py             ← NO dependencies

         ↑ Steps 2-4 can run IN PARALLEL ↑

Step 5:  schemas.py FULL REWRITE (~32 models)
         └─ depends on: nothing (but models reference types from id_utils conceptually)

Step 6:  tests/test_schemas.py REWRITE (~60 tests)
         └─ depends on: Step 5

Step 7:  fallback_utils.py REWRITE
         └─ depends on: Step 5 (needs new response classes)

Step 8:  tests/test_fallback_utils.py REWRITE
         └─ depends on: Steps 5, 7

         ↑ Steps 6 and 7-8 can run IN PARALLEL after Step 5 ↑

Step 9:  modal_app.py FULL REWRITE (7 functions)
         └─ depends on: Steps 2, 3, 4, 5, 7

Step 10: tests/test_modal_functions.py REWRITE (~40 tests)
         └─ depends on: Step 9

Step 11: test_harness.py REWRITE (7-function pipeline)
         └─ depends on: Steps 5, 9

Step 12: requirements.txt UPDATE (if any new deps needed)
         └─ depends on: nothing (can do anytime)

Step 13: Run pytest tests/ -v — target ~158 passing
         └─ depends on: ALL previous steps
```

**Dependency graph (simplified):**
```
[2,3,4] ──┐
           ├──→ [5] ──→ [6]
           │         ├──→ [7] ──→ [8]
           │         └──→ [9] ──→ [10]
           │                  └──→ [11]
           └──→ [12]
                          ALL ──→ [13]
```

---

## Section 10: Next Steps After Implementation

1. **Modal deployment:** `modal run modal_app.py` to verify containers build and functions execute on remote GPU/CPU. Requires valid Modal token.

2. **Llama model access:** `meta-llama/Llama-3.1-8B-Instruct` requires a Hugging Face token with Llama access approval. Set `HF_TOKEN` as a Modal secret.

3. **Backend integration:** Wire `.remote()` calls for all 7 functions in the backend orchestration sequence (spec section 6):
   ```
   extract_claims.remote(payload) → embed_claims.remote(payload) → cluster_claims.remote(payload)
   → rerank_evidence_batch.remote(payload) → nli_verify_batch.remote(payload)
   → score_clusters.remote(payload) → compute_umap.remote(payload)
   ```

4. **Model pre-warming:** Consider upgrading to `@app.cls` with `@modal.enter()` for Llama, embedding, rerank, and NLI models to reduce cold-start latency.

5. **Volume seeding:** Add a `download_models` function to pre-cache Llama + NLI + embedding + rerank model weights on the Modal Volume.

6. **Monitoring:** Add structured logging and timing metrics inside each function for observability.

7. **Backend ID generation:** Backend should use the same `id_utils` functions (or equivalent) to generate `claim_id`, `cluster_id`, `pair_id`, and `passage_id` values that ML will receive. Consider publishing `id_utils.py` as a shared utility.

---

## Verification Checklist

- [x] All 7 spec functions (5.1-5.7) are covered in this document
- [x] Every request/response field matches the spec JSON examples exactly
- [x] Implementation order has no circular dependencies
- [x] Test strategy covers all functions, edge cases, and fallbacks
- [x] `extract_claims` uses locally-run Llama per user decision
- [x] `score_clusters` algorithm matches spec formula exactly
- [x] NLI model change documented with runtime label-order reading
- [x] `sim_threshold` → `distance_threshold` conversion documented
- [x] Cache env var mismatches identified and corrected
- [x] Function length rule acknowledged with helper extraction guidance
- [x] All CLAUDE.md constraints addressed (no recursion, bounded loops, single dereference, smallest scope, max function length)
