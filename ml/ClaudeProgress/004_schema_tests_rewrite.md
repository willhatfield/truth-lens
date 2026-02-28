# 004 -- Schema Tests Rewrite for 32-Model schemas.py

**Date:** 2026-02-27
**Branch:** `feat/spec-alignment-v1`

## Changes Made

### Modified Files
- `tests/test_schemas.py` -- Complete rewrite to cover all 32 Pydantic models in the
  new `schemas.py`. The old file tested the 12-model v0 schema set (EmbedClaimsRequest,
  RankedPassage, NliPair, UmapPoint, etc.) which no longer exists.

### Test Coverage (85 tests across 32 models)

**Base classes (2 models):**
- BaseRequest: valid construction, missing analysis_id, empty analysis_id, custom schema_version
- BaseResponse: valid construction, missing analysis_id, warnings default, warnings populated

**extract_claims group (5 models):**
- ModelResponse: valid, missing model_id, empty response_text
- ClaimSpan: valid, negative start, negative end
- Claim: valid without span, valid with span, missing claim_text
- ExtractClaimsRequest: valid with inherited fields, empty responses rejected
- ExtractClaimsResponse: defaults, round-trip

**embed_claims group (3 models):**
- ClaimInput: valid, empty claim_id rejected
- EmbedClaimsRequest: valid with defaults, empty claims rejected, custom model_name
- EmbedClaimsResponse: defaults, with values

**cluster_claims group (4 models):**
- ClaimMetadata: valid, empty model_id rejected
- Cluster: valid, empty claim_ids rejected, missing representative_text
- ClusterClaimsRequest: valid with defaults, sim_threshold=0 rejected, sim_threshold>1 rejected
- ClusterClaimsResponse: defaults, round-trip

**rerank_evidence_batch group (5 models):**
- PassageInput: valid, empty text rejected
- RerankItem: valid, empty passages rejected
- ClaimRanking: valid with defaults, with values
- RerankEvidenceBatchRequest: valid with defaults, empty items, top_k too small/large
- RerankEvidenceBatchResponse: defaults

**nli_verify_batch group (4 models):**
- NliPairInput: valid, empty passage_text rejected
- NliResultOutput: valid with defaults, custom label and probs
- NliVerifyBatchRequest: valid with defaults, empty pairs, batch_size too small/large
- NliVerifyBatchResponse: defaults, round-trip

**compute_umap group (2 models):**
- ComputeUmapRequest: valid with defaults, n_neighbors too small/large, min_dist zero/above-one
- ComputeUmapResponse: defaults

**score_clusters group (7 models):**
- ScoringWeights: defaults, weight above 1, weight below 0
- VerdictThresholds: defaults, safe_min above 100, caution_min below 0
- AgreementDetail: defaults, negative count
- VerificationDetail: defaults, entailment_prob above 1, contradiction_prob below 0
- ClusterScore: valid with defaults, trust_score above 100/below 0, empty cluster_id
- ScoreClustersRequest: valid with defaults, empty clusters, inherits analysis_id
- ScoreClustersResponse: defaults, round-trip

## Test Results
- `pytest tests/test_schemas.py -v` -- **85 passed in 0.08s**

## Bugs Encountered
None. All 32 models validated cleanly against their field constraints.

## Next Steps
1. Run the full test suite (`pytest tests/ -v`) to verify no regressions in other test files
2. If `test_modal_functions.py` or `test_batch_utils.py` import old schema names, they will
   need updating to match the new 32-model schema set
3. Consider adding cross-model integration tests (e.g., building a full ScoreClustersRequest
   from upstream model outputs)
