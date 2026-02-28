# 003 -- Scoring Module Tests

**Date:** 2026-02-27
**Branch:** `feat/spec-alignment-v1`

## Changes Made

### New Files
- `tests/test_scoring.py` -- 22 tests covering all 7 public functions in `scoring.py`

### Test Coverage by Function

| Function | Tests | Cases |
|---|---|---|
| `clamp` | 3 | below min, above max, within range |
| `compute_agreement_score` | 4 | 0/5, 3/5, 5/5 models, 0 total_models edge case |
| `compute_verification_score` | 3 | high ent low contra, low ent high contra, both zero |
| `compute_trust_score` | 2 | typical weighted combo, zero weights |
| `determine_verdict` | 4 | SAFE, CAUTION, REJECT, CAUTION-on-high-contradiction |
| `find_supporting_models` | 3 | multiple models, single model, unknown claim_id skipped |
| `find_best_nli_for_cluster` | 3 | multiple results, no results, claim not in cluster skipped |

### Design Decisions
- Used `types.SimpleNamespace` for mock objects instead of importing from `schemas.py` to avoid coupling the test to Pydantic models
- Two small helper factories (`_make_claim`, `_make_nli`) keep tests DRY without introducing test fixtures
- All tests are pure functions with no I/O, no recursion, and no unbounded loops

## Bugs Encountered
None.

## Test Results
- `pytest tests/test_scoring.py -v` -- **22 passed** in 0.01s

## Next Steps
1. Integrate `scoring.py` into the cluster scoring pipeline (wire it into the Modal function or backend endpoint)
2. Add property-based tests (e.g., with Hypothesis) for `clamp` and `compute_verification_score` if fuzzing coverage is desired
3. Add integration-level tests that exercise the full scoring pipeline from NLI results through to verdict
