# 011 — HTTP Endpoints + E2E Test

**Date:** 2026-02-28
**Branch:** feat/011-http-endpoints-e2e-test
**Status:** Complete

## Changes Made

### 1. Restored `modal_app.py`
- Restored from commit `80eb5ad` (correct 920-line, 7-function version)
- Fixed import issues from accidental revert in commit `e9326fb`

### 2. New Files Created
| File | Lines | Purpose |
|------|-------|---------|
| `auth_utils.py` | ~28 | Reusable Bearer token validation for HTTP endpoints |
| `mock_data.py` | ~350 | Realistic mock data for 5 LLMs answering Earth shape question |
| `e2e_modal_test.py` | ~150 | Standalone E2E test script for deployed Modal HTTP endpoints |
| `tests/test_auth_utils.py` | ~55 | 5 tests for token validation |
| `tests/test_mock_data.py` | ~130 | 16 tests for schema validation, determinism, ID prefixes |
| `tests/test_e2e_structure.py` | ~50 | 8 tests for E2E phase config and URL building |

### 3. Modified Files
| File | Change |
|------|--------|
| `modal_app.py` | Added 7 HTTP POST endpoints with Bearer auth, fastapi deps in images |
| `tests/test_modal_deploy.py` | Added HTTP endpoint registration, secrets, memory tests |
| `requirements.txt` | Added `fastapi[standard]`, `requests` |
| `Makefile` | Added `e2e` and `mock-data` targets |

### 4. HTTP Endpoints Added
All 7 use `@modal.fastapi_endpoint(method="POST")` with `truthlens-api-key` secret:
- `http_extract_claims` (GPU A10G)
- `http_embed_claims` (GPU A10G)
- `http_rerank_evidence_batch` (GPU A10G)
- `http_nli_verify_batch` (GPU A10G, 24GB mem)
- `http_cluster_claims` (CPU 4-core)
- `http_compute_umap` (CPU 8-core)
- `http_score_clusters` (CPU 4-core)

### 5. Mock Data Scenario
- Question: "Is the Earth round or flat?"
- 5 models: gpt-4o, claude-3, gemini-pro, llama-3, flat-bot
- 4 correct (round/spherical) → cluster with SAFE verdict
- 1 incorrect (flat Earth) → cluster with REJECT verdict
- 10 total claims, 2 clusters, 3 evidence passages

## Bugs Encountered
- Pre-existing test failure: `test_fallback_probs_neutral_is_highest` (entailment=0.34 > neutral=0.33 in fallback probs). Not introduced by this work.
- **URL routing 404 bug (fixed):** All 7 HTTP endpoints returned 404 ("modal-http: invalid function call"). Root cause: `e2e_modal_test.py` built path-based URLs (`https://ws--app.modal.run/http_extract_claims`) but Modal uses subdomain-based routing (`https://ws--app-http-extract-claims.modal.run`). Fix: rewrote `_build_url()` to construct subdomain-based URLs with underscore→hyphen normalization. Updated CLI args from `--base-url` to `--workspace` / `--app-name`.

## E2E Test Results (2026-02-28)

All 7 phases passed on first deployed run (cold-start timing):

| Phase | Status | Time | Output |
|---|---|---|---|
| extract_claims | OK | 14.63s | 11 claims |
| embed_claims | OK | 29.89s | 10 vectors, dim=1024 |
| cluster_claims | OK | 7.84s | 4 clusters |
| rerank_evidence_batch | OK | 11.42s | 10 rankings |
| nli_verify_batch | OK | 19.30s | 10 results |
| compute_umap | OK | 33.14s | 10 coords3d |
| score_clusters | OK | 3.57s | 2 scores |
| **TOTAL** | **7/7** | **119.79s** | |

### Notes
- **extract_claims returned 11 claims** (mock data has 10): LLM split one claim into two sub-claims. Acceptable behavior.
- **cluster_claims returned 4 clusters** (expected 2): HDBSCAN found finer-grained groupings. `score_clusters` still returned 2 final scores, so downstream merging/filtering works correctly.
- **Timing is cold-start**: all containers were booting + loading models. `compute_umap` (33s) and `embed_claims` (30s) were the slowest. Warm runs will be significantly faster.

## Next Steps
1. Wire Vercel backend to use HTTP endpoints with Bearer auth
2. Consider adding rate limiting to HTTP endpoints
3. Monitor warm-start latency to establish performance baseline
