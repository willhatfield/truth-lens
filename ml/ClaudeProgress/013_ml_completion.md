# 013 - ML Pipeline Completion

## Date: 2026-02-28

## Changes Made

### Bug Fix: `modal_app.py` — `_fallback_nli_all()`
- Fixed near-uniform probability distribution in NLI fallback
- Changed entailment from 0.34 to 0.33, neutral from 0.33 to 0.34
- The fallback label is "neutral", so neutral probability must be highest
- This fixes the pre-existing `test_fallback_probs_neutral_is_highest` failure noted in entries 011 and 012

### New File: `README.md`
- Backend integration documentation for all 7 ML endpoints
- Covers: authentication (Bearer token), URL pattern (subdomain-based), request/response schemas, call ordering, deterministic ID generation, scoring algorithm, error handling, deployment, and rate limits
- Designed for backend developers to integrate without reading source code

## Current State
- 7 Modal functions: extract_claims, embed_claims, cluster_claims, rerank_evidence_batch, nli_verify_batch, compute_umap, score_clusters
- 7 HTTP endpoints with Bearer token authentication
- 32 Pydantic models in schemas.py
- 344+ tests across 5 test files, all passing
- E2E test infrastructure with chained request builders
- Full fallback/graceful degradation for every function

## Bugs Encountered
- Pre-existing: `test_fallback_probs_neutral_is_highest` — entailment=0.34 was highest probability but label was "neutral". Fixed by swapping entailment/neutral values.

## Next Steps
- ML pipeline is spec-complete and ready for backend integration
- Backend team can use README.md as the integration guide
- Future: live E2E test (`MODAL_API_KEY=xxx python e2e_modal_test.py`) to verify chaining against deployed endpoints
- Future: integration-level tests that mock HTTP responses for full chaining flow verification
