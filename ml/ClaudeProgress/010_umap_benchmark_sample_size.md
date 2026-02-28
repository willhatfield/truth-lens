# 010 -- Fix UMAP Benchmark Sample Size

**Date:** 2026-02-27
**Branch:** `fix/010-umap-benchmark-sample-size`
**Purpose:** Fix Codex P1 review finding -- UMAP benchmark payload had too few vectors.

---

## Problem

`make_compute_umap_payload()` supplied only 3 vectors, but `compute_umap` uses `umap.UMAP(n_components=3)`. With `n_samples == n_components`, UMAP's spectral init raises a `k >= N` error, causing the function to hit the zero-coordinate fallback path. The benchmark was therefore measuring fallback latency, not real UMAP projection performance.

## Fix

Increased the vector count from 3 to 6 so `n_samples (6) > n_components (3)`, ensuring UMAP performs real spectral initialization and projection.

## Files Changed

| File | Change |
|------|--------|
| `benchmark.py` | `make_compute_umap_payload()` now generates 6 vectors instead of 3 |
| `tests/test_benchmark.py` | Added `test_sample_count_exceeds_n_components` to guard against regression |
| `ClaudeProgress/010_umap_benchmark_sample_size.md` | Created (this file) |

## Test Summary

| Suite | Tests | Status |
|-------|-------|--------|
| `tests/` (all unit tests) | 344 | Pass |
| Deselected (modal_server marker) | 17 | -- |

## Next Steps

1. Items from 009 remain applicable
2. Consider adding a runtime assertion in `compute_umap` to warn when `n_samples <= n_components`
