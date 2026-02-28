# 007 -- Codex Review P1/P2: Guard Modal Auth & Hydrate Lazy References

**Date:** 2026-02-27
**Branch:** `fix/codex-review-p1-p2-modal-upgrade`

## Changes Made

### Modified Files

1. **`pytest.ini`** -- Added `addopts = -m "not modal_server"` so server-connectivity tests are excluded from default `pytest` runs. Users opt-in with `pytest -m modal_server`.

2. **`tests/test_modal_deploy.py`** -- Three fixes:
   - **P1 fix:** Added `HAS_MODAL_CREDS` module-level check and `@pytest.mark.skipif(not HAS_MODAL_CREDS, ...)` on `TestModalServerConnectivity` so tests skip cleanly when `MODAL_TOKEN_ID`/`MODAL_TOKEN_SECRET` are absent.
   - **P2 fix (functions):** `test_function_reference_creates` now calls `fn_ref.hydrate()` and asserts `fn_ref.object_id is not None` instead of the always-true `assert fn_ref is not None`.
   - **P2 fix (volume):** `test_volume_reference_creates` now calls `vol.hydrate()` and asserts `vol.object_id is not None`.

## Bugs Encountered

None.

## Next Steps

1. Merge this branch into `main` after CI passes.
2. Deploy the app to Modal (`modal deploy modal_app.py`) so that hydrated references resolve to real objects.
3. Add end-to-end tests that call `.remote()` on deployed functions with sample payloads.
