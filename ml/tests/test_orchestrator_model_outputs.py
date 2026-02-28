"""Tests for run_full_pipeline model_outputs parameter."""

import os
import json
import tempfile
import pytest
from unittest.mock import patch, MagicMock


def _make_ml_functions():
    """Create mock ML functions that return minimal valid responses."""
    def extract_claims(payload):
        return {"claims": [
            {"claim_id": "c1", "claim_text": "test claim",
             "model_id": "gpt4"}
        ], "warnings": []}

    def embed_claims(payload):
        return {"vectors": {"c1": [0.1, 0.2, 0.3]}, "warnings": []}

    def cluster_claims(payload):
        return {"clusters": [
            {"cluster_id": "cl1", "claim_ids": ["c1"],
             "representative_claim_id": "c1",
             "representative_text": "test claim"}
        ], "warnings": []}

    def compute_umap(payload):
        return {"coords3d": {}, "warnings": []}

    def rerank_evidence_batch(payload):
        return {"rankings": [], "warnings": []}

    def nli_verify_batch(payload):
        return {"results": [], "warnings": []}

    def score_clusters(payload):
        return {"cluster_scores": [], "warnings": []}

    return {
        "extract_claims": extract_claims,
        "embed_claims": embed_claims,
        "cluster_claims": cluster_claims,
        "compute_umap": compute_umap,
        "rerank_evidence_batch": rerank_evidence_batch,
        "nli_verify_batch": nli_verify_batch,
        "score_clusters": score_clusters,
    }


def test_model_outputs_skips_llm_stage():
    """run_full_pipeline with model_outputs skips _run_llm_stage."""
    import sys
    sys.path.insert(0, "/Users/willhatfield/Desktop/Developer/truth-lens/ml")

    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("llm_providers.call_all_llms") as mock_llm:
            from orchestrator import run_full_pipeline

            model_outputs = [
                {"model_id": "gpt4", "response_text": "The sky is blue."}
            ]
            result = run_full_pipeline(
                analysis_id="a_test_skip",
                prompt="Is the sky blue?",
                api_keys={},
                ml_functions=_make_ml_functions(),
                results_dir=tmpdir,
                model_outputs=model_outputs,
            )

            mock_llm.assert_not_called()

    assert isinstance(result, dict)


def test_model_outputs_none_calls_llm_stage():
    """run_full_pipeline with model_outputs=None calls _run_llm_stage."""
    import sys
    sys.path.insert(0, "/Users/willhatfield/Desktop/Developer/truth-lens/ml")

    with tempfile.TemporaryDirectory() as tmpdir:
        mock_responses = [
            {"model_id": "gpt4", "response_text": "The sky is blue."}
        ]
        with patch("llm_providers.call_all_llms",
                   return_value=(mock_responses, [])) as mock_llm:
            from orchestrator import run_full_pipeline

            run_full_pipeline(
                analysis_id="a_test_llm",
                prompt="Is the sky blue?",
                api_keys={"openai": "k"},
                ml_functions=_make_ml_functions(),
                results_dir=tmpdir,
                model_outputs=None,
            )

            mock_llm.assert_called_once()


def test_llm_calls_in_stages_completed_when_skipping():
    """llm_calls appears in stages_completed when model_outputs provided."""
    import sys
    sys.path.insert(0, "/Users/willhatfield/Desktop/Developer/truth-lens/ml")

    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("llm_providers.call_all_llms"):
            from orchestrator import run_full_pipeline

            model_outputs = [
                {"model_id": "gpt4", "response_text": "The sky is blue."}
            ]
            run_full_pipeline(
                analysis_id="a_stages",
                prompt="test",
                api_keys={},
                ml_functions=_make_ml_functions(),
                results_dir=tmpdir,
                model_outputs=model_outputs,
            )

    progress_path = os.path.join(tmpdir, "a_stages_progress.json")
    assert os.path.exists(progress_path), "Progress file should be written"
    with open(progress_path) as f:
        progress = json.load(f)
    assert "llm_calls" in progress["stages_completed"]


def test_progress_file_written_before_extract():
    """Progress file written immediately after skipping LLM stage."""
    import sys
    sys.path.insert(0, "/Users/willhatfield/Desktop/Developer/truth-lens/ml")

    written_stages = []

    original_write_progress = None

    def capture_write_progress(analysis_id, stage, models_completed,
                               stages_completed, warnings, results_dir):
        written_stages.append((stage, list(stages_completed)))
        # Actually write the file too
        import json, os
        path = os.path.join(results_dir, f"{analysis_id}_progress.json")
        data = {
            "status": "running",
            "stage": stage,
            "models_completed": models_completed,
            "stages_completed": stages_completed,
            "warnings": warnings,
        }
        with open(path, "w") as f:
            json.dump(data, f)

    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("llm_providers.call_all_llms"):
            with patch("orchestrator._write_progress",
                       side_effect=capture_write_progress):
                from orchestrator import run_full_pipeline

                model_outputs = [
                    {"model_id": "gpt4", "response_text": "blue sky"}
                ]
                run_full_pipeline(
                    analysis_id="a_progress",
                    prompt="test",
                    api_keys={},
                    ml_functions=_make_ml_functions(),
                    results_dir=tmpdir,
                    model_outputs=model_outputs,
                )

    first_call = written_stages[0] if written_stages else None
    assert first_call is not None
    assert first_call[0] == "extract_claims"
    assert "llm_calls" in first_call[1]
