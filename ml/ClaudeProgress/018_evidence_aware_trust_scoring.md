# 018 — Evidence-Aware Trust Score Redistribution

## Problem

With evidence retrieval stubbed out (`evidence.py` returns empty lists), the verification component (35% of trust score) always contributes **0**. This made 4/5 LLM agreement produce only ~55% trust (CAUTION) instead of the expected ~80% (SAFE).

The orchestrator creates synthetic `p_default_*` NLI pairs with `"[no evidence provided]"` text, so the NLI model runs on meaningless input and returns near-zero entailment.

## Solution

Detect when no real evidence exists and redistribute weights to agreement + independence only.

### Weight Formulas

- **With evidence:** `TRUST = 0.35*Agreement + 0.35*Verification + 0.15*Independence + 0.15*Consistency`
- **No evidence:** `TRUST = 0.70*Agreement + 0.30*Independence`

### Detection Logic

`check_has_evidence(evidence_passage_id)` returns `False` for:
- Empty string (no NLI results matched)
- IDs starting with `p_default_` (synthetic stubs from orchestrator)

## Files Changed

| File | Change |
|------|--------|
| `ml/scoring.py` | Added `check_has_evidence()`, added `has_evidence` param to `compute_trust_score()` |
| `ml/modal_app.py` | Imported `check_has_evidence`, wired into `_score_single_cluster` |
| `ml/mock_data.py` | Passed `has_evidence=True` (mock data uses real evidence IDs) |
| `ml/tests/test_scoring.py` | Added 10 new tests (check_has_evidence, no-evidence scenarios, default behavior) |
| `ml/tests/test_modal_functions.py` | Added `check_has_evidence` mock return values to all TestScoreClusters tests |

## Expected Results (No Evidence)

| LLMs Agree | Trust Score | Verdict |
|------------|-------------|---------|
| 5/5        | 100         | SAFE    |
| 4/5        | 80          | SAFE    |
| 3/5        | 60          | CAUTION |
| 2/5        | 40          | REJECT  |
| 1/5        | 20          | REJECT  |

## Next Steps

- Implement real `evidence.py` with web search (real passage IDs will automatically activate 4-component formula)
- No code changes needed when evidence becomes available — `check_has_evidence` returns `True` for non-stub IDs
