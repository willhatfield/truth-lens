# 017 — Trust Score 4-Component Formula

## Branch
`trust-score-4-component-formula`

## Changes Made

### New formula
```
trust_score = 0.35 × Agreement + 0.35 × Verification + 0.15 × Independence + 0.15 × Consistency
```

### scoring.py
- Removed `agreement_weight` and `verification_weight` parameters from `compute_trust_score`; weights are now fixed constants.
- Added `compute_independence_score(supporting_count, total_claim_models) -> float`
  - `100 * (supporting_count / total_claim_models)`
  - Measures fraction of unique request models that supported the cluster
- Added `compute_consistency_score(best_contradiction) -> float`
  - `100 * (1 - best_contradiction)`
  - Inverse of the best NLI contradiction probability

### modal_app.py
- Imported `compute_independence_score` and `compute_consistency_score`
- Added them as injected fn params to `_score_single_cluster`
- Computed `total_unique_models` from `req.claims.values()` (set of model_ids)
- Updated `compute_trust_score_fn` call to pass `(agreement, verification, independence, consistency)`
- Removed `req.weights.agreement_weight / verification_weight` from the call (weights are now fixed)

### tests/test_scoring.py
- Added imports for new functions
- Replaced old `compute_trust_score` tests (weight-param style) with new 4-param style
- Added tests for `compute_independence_score` (3 cases)
- Added tests for `compute_consistency_score` (3 cases)

### mock_data.py
- Updated `build_score_clusters_response` to call the two new functions and pass 4 args to `compute_trust_score`

## Notes
- `ScoringWeights` in schemas.py still holds `agreement_weight` / `verification_weight` fields but they are no longer used in scoring. Consider removing in a follow-up if not needed elsewhere.
- `test_modal_functions.py` uses `MagicMock` for the entire scoring module — no changes needed there.
