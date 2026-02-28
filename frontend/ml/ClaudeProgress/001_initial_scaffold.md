# 001 — Initial ML Pipeline Scaffold

**Date:** 2026-02-27
**Branch:** `feat/modal-ml-pipeline`

## Changes Made

### New Files
- `requirements.txt` — Pinned dependencies for Modal, Pydantic, torch, transformers, sentence-transformers, scikit-learn, umap-learn, numpy, pytest
- `schemas.py` — 12 Pydantic models covering all 5 function request/response pairs plus sub-models (RankedPassage, NliPair, NliResult, UmapPoint)
- `batch_utils.py` — `chunk_list()` and `flatten_batch_results()` with bounded loops and clamped parameters
- `fallback_utils.py` — `build_error_response()` generic error-response constructor
- `modal_app.py` — Modal App with 5 serverless functions:
  - `embed_claims` (GPU A10G, BGE-large-en-v1.5)
  - `rerank_evidence` (GPU A10G, ms-marco-MiniLM-L-6-v2)
  - `nli_verify` (GPU A10G, DeBERTa-large-MNLI)
  - `cluster_claims` (CPU, AgglomerativeClustering)
  - `compute_umap` (CPU, UMAP 3D projection)
- `test_harness.py` — Local integration test with fake model outputs validating the full pipeline
- `tests/test_schemas.py` — 35 tests for validation, defaults, constraints, round-trips
- `tests/test_batch_utils.py` — 17 tests for chunking and flattening edge cases
- `tests/test_fallback_utils.py` — 7 tests for error response construction
- `tests/test_modal_functions.py` — 27 tests with mocked ML models covering happy paths, error paths, and fallbacks

### Architecture Decisions
- Dict-in/dict-out at Modal boundary; Pydantic validation inside functions
- Two container images: `gpu_image` (torch/transformers) and `cpu_image` (sklearn/umap)
- Images include `add_local_python_source()` for Modal 1.0 forward compatibility
- Single-vector guard added to `cluster_claims` (AgglomerativeClustering fails on 1 sample)
- Custom `_softmax()` helper (no recursion, bounded loops) instead of scipy dependency
- All functions return `error` field — never raise to caller

## Bugs Encountered
1. **Mocking GPU imports:** `from sentence_transformers import SentenceTransformer` happens inside function bodies. `patch("modal_app.SentenceTransformer")` doesn't work — must use `patch.dict("sys.modules", {...})` to mock the entire module before the import executes.
2. **Single vector clustering:** `AgglomerativeClustering(n_clusters=None)` raises on single-sample inputs. Added early return guard.
3. **Modal automount deprecation:** Modal warns about implicit Python source mounting. Fixed by adding `.add_local_python_source("schemas", "batch_utils", "fallback_utils")` to both images. The remaining `tests` warning is benign (tests aren't deployed to containers).

## Test Results
- `pytest tests/ -v` — **86 passed**
- `python test_harness.py` — **All integration tests passed**

## Next Steps
1. **Modal deployment:** Run `modal run modal_app.py` with a valid Modal token to verify containers build and functions execute on remote GPU/CPU
2. **FastAPI integration:** Wire the backend to call these functions via `.remote()` — the dict-in/dict-out contract means the backend just does `result = embed_claims.remote(payload_dict)`
3. **Model pre-warming:** Consider upgrading to `@app.cls` with `@modal.enter()` for model preloading if cold-start latency is problematic
4. **Volume seeding:** Optionally add a `download_models` function to pre-cache model weights on the volume so first-run latency is lower
5. **Monitoring:** Add structured logging and timing metrics inside each function for observability
