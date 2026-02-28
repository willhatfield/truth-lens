"""Tests for orchestrator.py — full TruthLens pipeline coordinator."""

import json
import os
from unittest.mock import patch

from orchestrator import run_full_pipeline


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MOCK_API_KEYS = {"openai": "key-openai", "anthropic": "key-anth",
                  "gemini": "key-gem"}

_MOCK_LLM_RESPONSES = [
    {"model_id": "openai_gpt4", "response_text": "The sky is blue."},
    {"model_id": "claude_sonnet_4", "response_text": "Water is wet."},
]


def _mock_call_all_llms_success(prompt, api_keys):
    """Return two valid LLM responses with no warnings."""
    return list(_MOCK_LLM_RESPONSES), []


def _mock_call_all_llms_empty(prompt, api_keys):
    """Return zero LLM responses."""
    return [], ["All LLM providers failed"]


def _mock_ml_functions():
    """Create a dict of mock ML functions returning valid responses."""
    return {
        "extract_claims": lambda req: {
            "schema_version": "1.0",
            "analysis_id": req["analysis_id"],
            "claims": [
                {"claim_id": "c_1", "model_id": "openai_gpt4",
                 "claim_text": "The sky is blue.", "span": None},
                {"claim_id": "c_2", "model_id": "claude_sonnet_4",
                 "claim_text": "Water is wet.", "span": None},
            ],
            "warnings": [],
        },
        "embed_claims": lambda req: {
            "schema_version": "1.0",
            "analysis_id": req["analysis_id"],
            "vectors": {"c_1": [0.1] * 768, "c_2": [0.2] * 768},
            "dim": 768,
            "warnings": [],
        },
        "cluster_claims": lambda req: {
            "schema_version": "1.0",
            "analysis_id": req["analysis_id"],
            "clusters": [
                {"cluster_id": "cl_1", "claim_ids": ["c_1", "c_2"],
                 "representative_claim_id": "c_1",
                 "representative_text": "The sky is blue."},
            ],
            "warnings": [],
        },
        "compute_umap": lambda req: {
            "schema_version": "1.0",
            "analysis_id": req["analysis_id"],
            "coords3d": {"c_1": [0.1, 0.2, 0.3],
                         "c_2": [0.4, 0.5, 0.6]},
            "warnings": [],
        },
        "rerank_evidence_batch": lambda req: {
            "schema_version": "1.0",
            "analysis_id": req["analysis_id"],
            "rankings": [],
            "warnings": [],
        },
        "nli_verify_batch": lambda req: {
            "schema_version": "1.0",
            "analysis_id": req["analysis_id"],
            "results": [
                {"pair_id": "nli_1", "claim_id": "c_1",
                 "passage_id": "p_default_c_1",
                 "label": "entailment",
                 "probs": {"entailment": 0.9,
                           "contradiction": 0.05, "neutral": 0.05}},
                {"pair_id": "nli_2", "claim_id": "c_2",
                 "passage_id": "p_default_c_2",
                 "label": "entailment",
                 "probs": {"entailment": 0.8,
                           "contradiction": 0.1, "neutral": 0.1}},
            ],
            "warnings": [],
        },
        "score_clusters": lambda req: {
            "schema_version": "1.0",
            "analysis_id": req["analysis_id"],
            "cluster_scores": [
                {"cluster_id": "cl_1", "trust_score": 85,
                 "verdict": "SAFE",
                 "agreement": {
                     "models_supporting": ["openai_gpt4",
                                            "claude_sonnet_4"],
                     "count": 2},
                 "verification": {
                     "best_entailment_prob": 0.9,
                     "best_contradiction_prob": 0.05,
                     "evidence_passage_id": "p_default_c_1"}},
            ],
            "warnings": [],
        },
    }


_REQUIRED_RESULT_KEYS = [
    "schema_version", "analysis_id", "prompt", "models", "claims",
    "clusters", "evidence", "nli_results", "cluster_scores",
    "coords3d", "safe_answer", "model_metrics", "warnings", "status",
]


# ---------------------------------------------------------------------------
# 1. test_full_pipeline_success
# ---------------------------------------------------------------------------


@patch("orchestrator.call_all_llms")
def test_full_pipeline_success(mock_llms, tmp_path):
    """Full pipeline with 2 LLM responses produces a result with all
    required keys and status 'done'."""
    mock_llms.side_effect = _mock_call_all_llms_success

    result = run_full_pipeline(
        "aid_1", "Is the sky blue?", _MOCK_API_KEYS,
        _mock_ml_functions(), results_dir=str(tmp_path))

    for key in _REQUIRED_RESULT_KEYS:
        assert key in result, f"Missing key: {key}"

    assert result["status"] == "done"
    assert result["analysis_id"] == "aid_1"
    assert result["prompt"] == "Is the sky blue?"
    assert len(result["models"]) == 2
    assert len(result["claims"]) == 2
    assert len(result["clusters"]) >= 1
    assert result["safe_answer"] is not None


# ---------------------------------------------------------------------------
# 2. test_no_llm_responses
# ---------------------------------------------------------------------------


@patch("orchestrator.call_all_llms")
def test_no_llm_responses(mock_llms, tmp_path):
    """When all LLM providers fail, the result has status 'error'."""
    mock_llms.side_effect = _mock_call_all_llms_empty

    result = run_full_pipeline(
        "aid_err", "test prompt", _MOCK_API_KEYS,
        _mock_ml_functions(), results_dir=str(tmp_path))

    assert result["status"] == "error"
    assert "No LLM responses" in result["error"]
    assert len(result["warnings"]) >= 1


# ---------------------------------------------------------------------------
# 3. test_extract_claims_failure
# ---------------------------------------------------------------------------


@patch("orchestrator.call_all_llms")
def test_extract_claims_failure(mock_llms, tmp_path):
    """When extract_claims raises, the pipeline returns early with
    empty claims and status 'done'."""
    mock_llms.side_effect = _mock_call_all_llms_success

    fns = _mock_ml_functions()
    fns["extract_claims"] = lambda req: (_ for _ in ()).throw(
        RuntimeError("extract boom"))

    result = run_full_pipeline(
        "aid_ext", "test", _MOCK_API_KEYS,
        fns, results_dir=str(tmp_path))

    assert result["status"] == "done"
    assert result["claims"] == []
    found_warning = False
    for i in range(len(result["warnings"])):
        if "extract_claims failed" in result["warnings"][i]:
            found_warning = True
            break
    assert found_warning


# ---------------------------------------------------------------------------
# 4. test_embed_failure
# ---------------------------------------------------------------------------


@patch("orchestrator.call_all_llms")
def test_embed_failure(mock_llms, tmp_path):
    """When embed_claims raises, the pipeline returns early with
    empty vectors and status 'done'."""
    mock_llms.side_effect = _mock_call_all_llms_success

    fns = _mock_ml_functions()
    fns["embed_claims"] = lambda req: (_ for _ in ()).throw(
        RuntimeError("embed boom"))

    result = run_full_pipeline(
        "aid_emb", "test", _MOCK_API_KEYS,
        fns, results_dir=str(tmp_path))

    assert result["status"] == "done"
    assert result["claims"] != []
    assert result["clusters"] == []
    assert result["coords3d"] == {} or result["coords3d"] is None


# ---------------------------------------------------------------------------
# 5. test_cluster_failure_fallback
# ---------------------------------------------------------------------------


@patch("orchestrator.call_all_llms")
def test_cluster_failure_fallback(mock_llms, tmp_path):
    """When cluster_claims raises, the pipeline falls back to
    singleton clusters (one cluster per claim)."""
    mock_llms.side_effect = _mock_call_all_llms_success

    fns = _mock_ml_functions()
    fns["cluster_claims"] = lambda req: (_ for _ in ()).throw(
        RuntimeError("cluster boom"))

    result = run_full_pipeline(
        "aid_cl", "test", _MOCK_API_KEYS,
        fns, results_dir=str(tmp_path))

    assert result["status"] == "done"
    # Singleton fallback: one cluster per claim
    assert len(result["clusters"]) == 2
    for i in range(len(result["clusters"])):
        cl = result["clusters"][i]
        assert len(cl["claim_ids"]) == 1

    found_warning = False
    for i in range(len(result["warnings"])):
        if "cluster_claims failed" in result["warnings"][i]:
            found_warning = True
            break
    assert found_warning


# ---------------------------------------------------------------------------
# 6. test_nli_failure_fallback
# ---------------------------------------------------------------------------


@patch("orchestrator.call_all_llms")
def test_nli_failure_fallback(mock_llms, tmp_path):
    """When nli_verify_batch raises, the pipeline falls back to
    neutral probabilities for all NLI pairs."""
    mock_llms.side_effect = _mock_call_all_llms_success

    fns = _mock_ml_functions()
    fns["nli_verify_batch"] = lambda req: (_ for _ in ()).throw(
        RuntimeError("nli boom"))

    result = run_full_pipeline(
        "aid_nli", "test", _MOCK_API_KEYS,
        fns, results_dir=str(tmp_path))

    assert result["status"] == "done"
    assert len(result["nli_results"]) >= 1
    for i in range(len(result["nli_results"])):
        nli = result["nli_results"][i]
        assert nli["label"] == "neutral"
        assert abs(nli["probs"]["neutral"] - 0.34) < 0.01

    found_warning = False
    for i in range(len(result["warnings"])):
        if "nli_verify failed" in result["warnings"][i]:
            found_warning = True
            break
    assert found_warning


# ---------------------------------------------------------------------------
# 7. test_progress_files_written
# ---------------------------------------------------------------------------


@patch("orchestrator.call_all_llms")
def test_progress_files_written(mock_llms, tmp_path):
    """Verify that a progress JSON file exists after pipeline run."""
    mock_llms.side_effect = _mock_call_all_llms_success

    run_full_pipeline(
        "aid_prog", "test", _MOCK_API_KEYS,
        _mock_ml_functions(), results_dir=str(tmp_path))

    progress_path = os.path.join(str(tmp_path),
                                 "aid_prog_progress.json")
    assert os.path.exists(progress_path)

    with open(progress_path, "r") as f:
        progress = json.load(f)

    assert progress["status"] == "done"
    assert progress["stage"] == "complete"
    assert len(progress["stages_completed"]) >= 1
    assert len(progress["models_completed"]) >= 1


# ---------------------------------------------------------------------------
# 8. test_result_file_written
# ---------------------------------------------------------------------------


@patch("orchestrator.call_all_llms")
def test_result_file_written(mock_llms, tmp_path):
    """Verify final result JSON file is written with correct structure."""
    mock_llms.side_effect = _mock_call_all_llms_success

    run_full_pipeline(
        "aid_res", "test", _MOCK_API_KEYS,
        _mock_ml_functions(), results_dir=str(tmp_path))

    result_path = os.path.join(str(tmp_path), "aid_res.json")
    assert os.path.exists(result_path)

    with open(result_path, "r") as f:
        saved = json.load(f)

    for key in _REQUIRED_RESULT_KEYS:
        assert key in saved, f"Missing key in saved file: {key}"

    assert saved["status"] == "done"
    assert saved["analysis_id"] == "aid_res"
    assert len(saved["models"]) == 2
