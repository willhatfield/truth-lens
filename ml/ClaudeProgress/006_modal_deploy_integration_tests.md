# 006 -- Modal Deployment Integration Tests

**Date:** 2026-02-27
**Branch:** `feat/modal-deploy-integration-tests`

## Changes Made

### New Files

1. **`pytest.ini`** -- Registers the `modal_server` custom marker to avoid pytest warnings when running `pytest -m modal_server`.

### Modified Files

1. **`tests/test_modal_deploy.py`** -- Added `TestModalServerConnectivity` class (5 test methods, 17 total test cases with parametrize).

### TestModalServerConnectivity Details (17 test cases)

All tests are marked with `@pytest.mark.modal_server` for selective execution.

1. **test_client_authenticates** -- Calls `modal.Client.from_env()` and `client.hello()` to verify valid credentials and server connectivity.
2. **test_function_reference_creates** (parametrized x7) -- Creates a lazy `modal.Function.from_name("truthlens-ml", name)` reference for each of the 7 functions.
3. **test_build_def_is_valid** (parametrized x7) -- Verifies `fn.get_build_def()` returns a non-empty string for each registered function (Modal uses this to build container images).
4. **test_volume_reference_creates** -- Creates a `modal.Volume.from_name("truthlens-model-cache")` reference without error.
5. **test_source_modules_in_mounts** -- Verifies that `extract_claims` has at least one mount in `fn.spec.mounts`.

### Running the tests

- All tests: `pytest tests/test_modal_deploy.py -v` (58 tests)
- Server connectivity only: `pytest -m modal_server -v` (17 tests)
- Full suite: `pytest tests/ -v` (263 tests)

## Test Results

- `pytest tests/test_modal_deploy.py -v` -- **58 passed in 0.36s**
- `pytest -m modal_server -v` -- **17 passed, 254 deselected in 0.49s**
- `pytest tests/ -v` -- **263 passed in 1.70s**

## Bugs Encountered

None.

## Next Steps

1. Deploy the app to Modal (`modal deploy modal_app.py`) so that `Function.from_name` lazy references can actually resolve at call time
2. Add end-to-end tests that call `.remote()` on deployed functions with sample payloads
3. Consider adding a `modal_e2e` marker for tests that require a deployed app vs just server connectivity
