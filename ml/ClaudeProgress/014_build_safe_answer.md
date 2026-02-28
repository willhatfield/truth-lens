# 014 — `build_safe_answer` Builder

**Date:** 2026-02-28
**Branch:** `feat/build-safe-answer`

## Changes Made

### New Files
- **`backend/app/safe_answer.py`** — Deterministic safe-answer builder (10 functions, 0 ML imports)
- **`backend/tests/test_safe_answer.py`** — Full test suite (72 tests across 10 test classes)
- **`backend/tests/__init__.py`** — Package init for test discovery

### Implementation Summary

`build_safe_answer(clusters, cluster_scores, *, rewrite_client=None)` consumes the ML pipeline's final outputs and returns a `(safe_answer_dict, warnings_list)` tuple. It never raises.

**Internal functions:**
1. `_try_parse_cluster` / `_try_parse_score` — Defensive parsers, return None on bad input
2. `_dedup_clusters` / `_dedup_scores` — Deterministic dedup with tie-breaking
3. `_verdict_rank` — SAFE=0, CAUTION=1, REJECT=2
4. `_sorted_scores` — 3-key sort: verdict rank, -trust_score, cluster_id
5. `_validate_prefix` / `_apply_transition_rewrite` — Per-sentence rewrite with fallback
6. `_base_sentences` — Builds transition-free factual sentences

**Key design decisions:**
- `_MAX_INPUT = 10000` for parse/dedup loop bounds; `_MAX_CLUSTERS = 1000` for final cap
- All loops use fixed upper bounds (`min(len(x), _MAX_INPUT)`)
- Malformed inputs skipped with warnings (never fatal)
- Deterministic output when `rewrite_client=None`

## Bugs Encountered

- Initial implementation capped parsing loops at `_MAX_CLUSTERS` (1000), which prevented the truncation test from ever triggering (inputs > 1000 were silently dropped during parsing instead of reaching the sort-then-cap step). Fixed by introducing `_MAX_INPUT = 10000` as the parse/dedup bound.

## Test Results

72/72 tests pass (`python -m pytest tests/test_safe_answer.py -v`):
- TestTryParseCluster (7), TestTryParseScore (10), TestDedupClusters (4)
- TestDedupScores (4), TestVerdictRank (4), TestSortedScores (5)
- TestValidatePrefix (5), TestApplyTransitionRewrite (6), TestBaseSentences (8)
- TestBuildSafeAnswer (19 integration tests)

## Next Steps

- Wire `build_safe_answer` into the orchestrator (`backend/app/orchestrator.py`)
- Implement a real `rewrite_client` (LLM-based transition prefix generator)
- Add integration tests that chain ML pipeline output into `build_safe_answer`
