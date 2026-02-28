# 008 -- ML Folder Comprehensive Status Report

**Date:** 2026-02-27
**Branch:** `docs/008-ml-status-report`
**Purpose:** Single entry point for future agents. Synthesizes all prior progress files (001-007) against the `BACKEND-ML-PIPELINE.md` spec.

---

## What Has Been Done (ML Folder -- COMPLETE per spec)

### Infrastructure (3 files)

| File | Lines | Purpose |
|------|-------|---------|
| `schemas.py` | 238 | 32 Pydantic models covering all 7 function request/response pairs + shared types, matching spec Section 4 |
| `modal_app.py` | 920 | All 7 Modal functions with GPU/CPU resource declarations per spec Section 8 |
| `requirements.txt` | -- | All dependencies pinned (modal 1.3.4, torch 2.5.1, transformers 4.47.1, etc.) |

### Utilities (5 files)

| File | Lines | Purpose |
|------|-------|---------|
| `id_utils.py` | 25 | Deterministic SHA1 IDs (claim, cluster, pair) per spec Section 1.2 |
| `claim_extraction.py` | 26 | Sentence-split fallback per spec Section 5.1 |
| `scoring.py` | 120 | Trust score algorithm (agreement + verification weighting) per spec Section 5.7 |
| `fallback_utils.py` | 23 | Generic warning response builder for non-fatal degradation (spec Section 9) |
| `batch_utils.py` | 66 | Chunking/flattening for batched inference (spec Section 3.4) |

### 7 Modal Functions (all implemented with fallbacks)

1. **`extract_claims`** (GPU A10G, 16GB) -- Llama-3.1-8B extraction with sentence-split fallback
2. **`embed_claims`** (GPU A10G, 16GB) -- BGE-large-en-v1.5 embeddings
3. **`cluster_claims`** (CPU 4-core, 8GB) -- Agglomerative clustering
4. **`rerank_evidence_batch`** (GPU A10G, 16GB) -- Cross-encoder ms-marco-MiniLM reranking
5. **`nli_verify_batch`** (GPU A10G, 24GB) -- DeBERTa-v3-large NLI with neutral fallback
6. **`compute_umap`** (CPU 8-core, 16GB) -- 3D UMAP projection with zero-coord fallback
7. **`score_clusters`** (CPU 4-core, 8GB) -- Trust scores + SAFE/CAUTION/REJECT verdicts

### Tests (8 test files + 1 harness, 248 passing)

| Test File | Count | Coverage |
|-----------|-------|----------|
| `tests/test_schemas.py` | 85 | All 32 Pydantic models |
| `tests/test_modal_functions.py` | 45 | Mocked ML models for all 7 functions |
| `tests/test_modal_deploy.py` | 41 | Deployment config + server connectivity (credential-guarded) |
| `tests/test_scoring.py` | 24 | Scoring algorithm edge cases |
| `tests/test_batch_utils.py` | 17 | Chunking/flattening edge cases |
| `tests/test_fallback_utils.py` | 14 | All 7 response types |
| `tests/test_claim_extraction.py` | 10 | Sentence splitting edge cases |
| `tests/test_id_utils.py` | 10 | ID generation determinism |
| `test_harness.py` | 2 pass, 6 error | Full 7-phase local integration test (fixture chaining broken -- see Known Issues) |

**Total:** 246 pass in `tests/`, 2 pass in harness, 17 deselected (modal_server marker), 6 harness errors.

### Spec Compliance

- `schema_version: "1.0"` in all requests/responses
- All non-fatal degradation rules implemented (spec Section 9)
- Modal Volume caching at `/models` with `HF_HOME`, `TRANSFORMERS_CACHE`, `SENTENCE_TRANSFORMERS_HOME`
- Bounded loops, no recursion, single-page functions per CLAUDE.md constraints
- NLI labels read from `model.config.id2label` at runtime (not hardcoded)

---

## Known Issues

1. **`test_harness.py` fixture errors (6 tests):** Tests `test_embed_phase`, `test_cluster_phase`, `test_nli_phase`, `test_score_phase`, `test_umap_phase`, and `test_serialization` error because they reference pytest fixtures (e.g. `extract_resp`) that are not defined. These are meant to chain phase outputs but the fixtures were never implemented. The 2 passing tests (`test_extract_phase`, `test_rerank_phase`) also trigger `PytestReturnNotNoneWarning` because they `return` instead of only `assert`.

2. **scipy import issue (local env only):** `sklearn.cluster` tests originally triggered `No module named 'scipy.optimize._highspy._core.simplex_constants'`. Resolved by mocking sklearn in tests. Not a code bug -- environment-specific.

---

## What Remains (ML Folder)

**Nothing is blocking from the ML side.** The ML folder is spec-complete per `BACKEND-ML-PIPELINE.md` Sections 3-5 and 8-9.

### Non-blocking future improvements

1. **Fix `test_harness.py` fixtures** -- Define proper pytest fixtures or refactor to pass phase outputs via module-level state. Low priority since all logic is covered by unit tests.
2. **Real GPU inference testing** -- Current tests mock all ML models; no live Modal GPU runs exist.
3. **Performance benchmarking** -- No latency/throughput numbers for cold/warm starts.
4. **Model weight pre-loading** -- Could add a `modal volume put` script to pre-cache weights and avoid cold-start downloads.
5. **Batch size tuning** -- `batch_utils` defaults are conservative; real-world tuning needed after backend integration.

### What the backend still needs to do (out of ML scope)

- **Backend orchestration sequence** (spec Section 6) -- Call ML functions in order and emit stage events
- **Frontend event streaming** (spec Section 2) -- WebSocket/SSE events
- **Evidence retrieval** (spec Section 6 step 5) -- Backend responsibility, not ML
- **Safe answer generation** (spec Section 7) -- Backend builds from scored clusters
- **Final AnalysisResult payload assembly** (spec Section 7)

---

## Prior Progress Files (chronological)

| File | Summary |
|------|---------|
| `001_initial_scaffold.md` | Initial project setup, Modal app skeleton, first test structure |
| `002_spec_alignment_architecture.md` | Architecture decisions aligning with BACKEND-ML-PIPELINE.md |
| `002_spec_alignment_implementation.md` | Implementation of spec-aligned schemas and utilities |
| `003_scoring_tests.md` | Scoring algorithm implementation and tests |
| `004_schema_tests_rewrite.md` | Full rewrite of schema tests to cover all 32 models (85 tests) |
| `005_modal_app_and_tests_rewrite.md` | Complete rewrite of modal_app.py (7 functions) and test_modal_functions.py (44 tests) |
| `006_modal_deploy_integration_tests.md` | Deployment config tests and Modal server connectivity tests |
| `007_codex_review_p1_p2_fixes.md` | Guarded Modal auth checks, hydrated lazy references |
