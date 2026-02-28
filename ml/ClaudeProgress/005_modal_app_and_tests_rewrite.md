# 005 -- modal_app.py and test_modal_functions.py Full Rewrite (Steps 9-10)

**Date:** 2026-02-27
**Branch:** `feat/spec-alignment-v1`

## Changes Made

### Modified Files

1. **`modal_app.py`** -- Complete rewrite from 5 functions to 7 functions, aligned with the TruthLens Integration Spec v1.

2. **`tests/test_modal_functions.py`** -- Complete rewrite from 27 tests to 44 tests covering all 7 functions.

### modal_app.py Changes

**New functions (2):**
- `extract_claims` -- GPU A10G, Llama-based claim extraction with sentence-split fallback via `claim_extraction.py`. Helpers `_texts_to_claims`, `_extract_from_single_response`, and `_fallback_extract_all` keep the main function under 60 lines.
- `score_clusters` -- CPU, delegates to `scoring.py` helpers for trust score computation. Helper `_score_single_cluster` extracted for function length compliance.

**Renamed functions (2):**
- `rerank_evidence` renamed to `rerank_evidence_batch` -- now accepts `RerankEvidenceBatchRequest` with `items: List[RerankItem]` (batch of claims). Helpers `_rerank_single_item`, `_fallback_ranking_for_item`, `_fallback_rerank_all` extracted.
- `nli_verify` renamed to `nli_verify_batch` -- now accepts `NliVerifyBatchRequest` with `pairs: List[NliPairInput]`. Helper `_run_nli_batch` and `_fallback_nli_all` extracted.

**Rewritten functions (3):**
- `embed_claims` -- now uses `ClaimInput` objects, returns `Dict[str, List[float]]` vectors keyed by `claim_id`, returns `dim` instead of `dimension`
- `cluster_claims` -- now uses `Dict[str, List[float]]` vectors, `sim_threshold` with `distance_threshold = 1.0 - sim_threshold` conversion, returns `List[Cluster]` objects with `cluster_id`, `claim_ids`, `representative_claim_id`, `representative_text`
- `compute_umap` -- now uses `Dict[str, List[float]]` vectors, `random_state` parameter, returns `coords3d: Dict[str, List[float]]`

**Cross-cutting changes:**
- Imports updated to use new schemas (32-model set from spec)
- `build_error_response` replaced with `build_warning_response` from updated `fallback_utils.py`
- `error: Optional[str]` replaced with `warnings: List[str]` on all responses
- `SHARED_ENV` updated: `TRANSFORMERS_CACHE` now `/models/hf`, `SENTENCE_TRANSFORMERS_HOME` now `/models/st`
- `NLI_MODEL_NAME` changed from `microsoft/deberta-large-mnli` to `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli`
- Added `LLAMA_MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"`
- NLI labels read from `model.config.id2label` at runtime instead of hardcoded `NLI_LABELS` list
- `.add_local_python_source` updated to include `id_utils`, `claim_extraction`, `scoring`
- `_softmax()` helper kept as-is (bounded loops, numerically stable)
- All functions bounded with `MAX_RESPONSES`, `MAX_ITEMS`, `MAX_PAIRS`, `MAX_VECTORS`, `MAX_CLUSTERS` constants

### test_modal_functions.py Changes (44 tests)

**TestSoftmax (4 tests):** Unchanged -- uniform, dominant, single, negative values.

**TestExtractClaims (6 tests):**
- Happy path: mocked Llama returns JSON array of claims
- Fallback: Llama load fails, sentence-split used, warning appended
- Invalid payload: warning response returned
- Multiple model responses processed
- Claim IDs are deterministic (same input produces same ID)
- schema_version propagation

**TestEmbedClaims (5 tests):**
- Happy path: vectors keyed by claim_id, dim correct
- Multiple claims batched correctly
- Invalid payload: empty claims
- Missing analysis_id
- Model load failure with warning

**TestClusterClaims (6 tests):**
- Happy path: clusters with mocked sklearn returning 2 clusters
- Single vector: one cluster returned
- sim_threshold conversion verified: 0.85 yields distance_threshold=0.15 (verified via mock call args)
- Fallback: sklearn fails, single cluster, warning
- Invalid payload: missing required fields
- Claims metadata used for representative_text

**TestRerankEvidenceBatch (6 tests):**
- Happy path: rankings with ordered_passage_ids and scores dict
- Multiple items in batch
- top_k limiting works
- Fallback: original order on model failure
- Invalid payload
- Scores are descending

**TestNliVerifyBatch (6 tests):**
- Happy path: results with pair_id, claim_id, passage_id, probs
- Label order read from model.config.id2label (not hardcoded)
- Fallback: neutral with near-uniform probs on failure
- Invalid payload: empty pairs
- Missing field
- Probs sum to ~1.0

**TestComputeUmap (5 tests):**
- Happy path: coords3d keyed by claim_id
- random_state passed to UMAP (verified via mock call args)
- Fallback: zeros on failure
- Invalid payload: empty vectors
- Missing vectors

**TestScoreClusters (6 tests):**
- Happy path: trust_score, verdict, agreement, verification computed
- SAFE verdict
- CAUTION verdict
- REJECT verdict
- No NLI results: verification based on zeros
- Multiple clusters scored independently

### Mocking Pattern

Tests use `patch.dict("sys.modules", {...})` to mock lazily-imported packages (sentence_transformers, transformers, torch, sklearn.cluster, umap, id_utils, claim_extraction, scoring). This is the same pattern from the initial scaffold, extended to cover the new modules.

## Test Results
- `pytest tests/test_modal_functions.py -v` -- **44 passed in 1.18s**

## Bugs Encountered
1. **scipy import failure in local env:** The `test_sim_threshold_conversion` and `test_happy_path_with_cluster_objects` tests originally called real sklearn, which triggered a scipy import error (`No module named 'scipy.optimize._highspy._core.simplex_constants'`). Fixed by mocking `sklearn.cluster` in those tests and verifying the distance_threshold conversion via mock call args instead.
2. **Function length violations:** `_extract_from_single_response` was 77 lines and `cluster_claims` was 66 lines, exceeding the 60-line CLAUDE.md rule. Fixed by extracting `_texts_to_claims` helper from the extraction function, and `_single_vector_response` helper from the clustering function.

## Next Steps
1. **Step 11:** Rewrite `test_harness.py` for the 7-function pipeline
2. **Step 12:** Update `requirements.txt` if any new dependencies are needed
3. **Step 13:** Run `pytest tests/ -v` to verify all ~158 tests pass across all test files
4. Note: Some tests in `test_schemas.py` or `test_fallback_utils.py` may need the new schemas and `build_warning_response` -- verify these steps (5-8) have been completed
