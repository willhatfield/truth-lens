# 012 - E2E Test Chaining & Field Validation

## Date: 2026-02-28

## Changes Made

### New File: `e2e_request_builders.py`
- 7 builder functions that construct request payloads from prior phase responses
- Each builder falls back to mock data if upstream dependencies are missing
- `build_request()` dispatch function uses if/elif chain (no function pointers per CLAUDE.md)
- `CHAIN_DEPS` dict defines upstream dependencies per builder key
- `VALID_BUILDER_KEYS` frozenset for validation
- `_has_deps()` helper checks if all upstream deps are present
- All loops bounded by explicit `MAX_*` constants

### Modified: `e2e_modal_test.py`
- Replaced `from mock_data import build_full_pipeline_data` with `from e2e_request_builders import build_request, CHAIN_DEPS, VALID_BUILDER_KEYS`
- Changed PHASES tuple 3rd element from `req_key` (e.g. "extract_request") to `builder_key` (e.g. "extract")
- `run_e2e()` now chains responses: stores successful phase responses in `collected_responses` dict, passes it to `build_request()` for subsequent phases
- Added `_validate_response_fields(data, resp_fields)` — returns `(False, "MISSING: ...")` if any expected field is absent from the response; phase marked FAILED with status `-1`
- Added `_is_chained(builder_key, responses)` — returns True if all upstream deps are present
- Each phase prints `[chained]` or `[mock]` tag during execution
- `_print_summary()` now displays "FIELDS" status for field-validation failures (status -1)

### New File: `tests/test_e2e_request_builders.py`
- 42 tests across 12 test classes
- Tests all 7 builders (mock fallback + chained behavior)
- Tests dispatch function (valid keys, invalid key raises, chained data)
- Tests `_has_deps`, `_validate_response_fields`, `_is_chained`
- Tests chaining detection across full pipeline
- Tests `CHAIN_DEPS` completeness

### Modified: `tests/test_e2e_structure.py`
- Added import for `VALID_BUILDER_KEYS`
- Added `test_builder_keys_are_valid` — verifies each phase's builder key is in the known set

## Bugs Encountered
- None during this change
- Pre-existing failure: `test_fallback_probs_neutral_is_highest` in `test_modal_functions.py` (unrelated)

## Next Steps
- Investigate and fix the pre-existing `test_fallback_probs_neutral_is_highest` failure
- Run live E2E test (`MODAL_API_KEY=xxx python e2e_modal_test.py`) to verify chaining works against deployed endpoints
- Consider adding integration-level tests that mock HTTP responses to verify the full chaining flow without a live Modal deployment
