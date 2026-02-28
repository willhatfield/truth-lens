# 002 — Spec-Alignment Implementation

**Date:** 2026-02-28
**Branch:** `feat/spec-alignment-v1`
**Based on:** `002_spec_alignment_architecture.md` (on `feat/spec-alignment-docs` branch)

## Changes Made

### New Files (6)
- `id_utils.py` — 3 deterministic SHA1 ID functions (make_claim_id, make_cluster_id, make_pair_id)
- `claim_extraction.py` — sentence_split_claims() fallback with MAX_SENTENCES=500 bound
- `scoring.py` — 7 pure trust-scoring functions (clamp, compute_agreement_score, compute_verification_score, compute_trust_score, determine_verdict, find_supporting_models, find_best_nli_for_cluster)
- `tests/test_id_utils.py` — 10 tests
- `tests/test_claim_extraction.py` — 10 tests
- `tests/test_scoring.py` — 22 tests

### Rewritten Files (6)
- `schemas.py` — 32 Pydantic models (was 12), BaseRequest/BaseResponse inheritance, warnings instead of error, Dict vectors keyed by claim_id
- `fallback_utils.py` — build_warning_response() replaces build_error_response()
- `modal_app.py` — 7 functions (was 5): extract_claims (NEW), embed_claims, cluster_claims, rerank_evidence_batch (RENAMED), nli_verify_batch (RENAMED), compute_umap, score_clusters (NEW)
- `test_harness.py` — 7-function pipeline with new schema types
- `tests/test_schemas.py` — 85 tests (was 35)
- `tests/test_fallback_utils.py` — 14 tests (was 7)
- `tests/test_modal_functions.py` — 44 tests (was 27)

### Kept As-Is (2)
- `batch_utils.py` — No spec changes needed
- `tests/test_batch_utils.py` — 17 tests still passing

## Test Results
- `pytest tests/ -v` — **202 passed** in 1.62s
- `python test_harness.py` — **All 8 phases passed**

## Key Implementation Decisions
1. NLI label order read from `model.config.id2label` at runtime (not hardcoded)
2. `sim_threshold` → `distance_threshold` via `1.0 - sim_threshold`
3. Cache env vars fixed: `TRANSFORMERS_CACHE=/models/hf`, `SENTENCE_TRANSFORMERS_HOME=/models/st`
4. NLI model changed to `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli`
5. Llama model: `meta-llama/Llama-3.1-8B-Instruct` with sentence-split fallback
6. Helper functions extracted to keep all functions under 60 lines

## Bugs Encountered
- Subagents on `feat/spec-alignment-v1` couldn't find `002_spec_alignment_architecture.md` since it lives on `feat/spec-alignment-docs` branch. Agents used detailed prompts + source code reading instead.
- Modal "automount" deprecation warnings for `tests` module are benign (tests aren't deployed to containers).

## Detailed Sub-Agent Notes
- `003_scoring_tests.md` — Detailed scoring test coverage breakdown
- `004_schema_tests_rewrite.md` — Per-model test coverage for all 32 schemas
- `005_modal_app_and_tests_rewrite.md` — Modal function rewrite details, mocking patterns, bug fixes

## Next Steps
1. Modal deployment: `modal run modal_app.py` to verify containers build
2. Llama model access: needs HF token with Llama access approval
3. Backend integration: wire `.remote()` calls for all 7 functions in sequence:
   `extract_claims → embed_claims → cluster_claims → rerank_evidence_batch → nli_verify_batch → score_clusters → compute_umap`
4. Model pre-warming: consider `@app.cls` with `@modal.enter()`
5. Volume seeding: pre-cache model weights on Modal Volume
6. Backend ID generation: backend should use same `id_utils` functions for `claim_id`, `cluster_id`, `pair_id`, `passage_id`
