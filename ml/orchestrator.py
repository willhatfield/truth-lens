"""Full pipeline orchestrator for the TruthLens ML service.

Coordinates LLM calls, ML processing stages, safe answer building,
and model metrics computation. Writes progress and results to disk.
"""

import json
import os

from llm_providers import call_all_llms
from safe_answer import build_safe_answer
from model_metrics import compute_model_metrics
from evidence import retrieve_evidence

_MAX_STAGES = 20
_MAX_MODELS = 10
_MAX_CLAIMS = 50_000
_MAX_WARNINGS = 1_000
_MAX_PASSAGES_PER_CLAIM = 10
_MAX_CLUSTER_CAP = 1000
_RESULTS_DIR = "/results"

# Stage names in execution order
_STAGES = [
    "llm_calls",
    "extract_claims",
    "embed_claims",
    "cluster_claims",
    "compute_umap",
    "rerank_evidence",
    "nli_verify",
    "score_clusters",
    "safe_answer",
    "model_metrics",
]


def _write_progress(analysis_id, stage, models_completed,
                    stages_completed, warnings, results_dir):
    """Write progress JSON to results directory."""
    path = os.path.join(results_dir, f"{analysis_id}_progress.json")
    data = {
        "status": "running",
        "stage": stage,
        "models_completed": models_completed[:_MAX_MODELS],
        "stages_completed": stages_completed[:_MAX_STAGES],
        "warnings": warnings[:_MAX_WARNINGS],
    }
    with open(path, "w") as f:
        json.dump(data, f)


def _write_result(analysis_id, result, results_dir):
    """Write final result JSON to results directory."""
    path = os.path.join(results_dir, f"{analysis_id}.json")
    with open(path, "w") as f:
        json.dump(result, f)


def _write_error(analysis_id, error_msg, warnings, results_dir):
    """Write error JSON to results directory."""
    path = os.path.join(results_dir, f"{analysis_id}.json")
    data = {
        "status": "error",
        "error": error_msg,
        "warnings": warnings[:_MAX_WARNINGS],
    }
    with open(path, "w") as f:
        json.dump(data, f)
    progress_path = os.path.join(results_dir,
                                 f"{analysis_id}_progress.json")
    progress = {
        "status": "error",
        "stage": "failed",
        "models_completed": [],
        "stages_completed": [],
        "warnings": warnings[:_MAX_WARNINGS],
    }
    with open(progress_path, "w") as f:
        json.dump(progress, f)


def _write_done_progress(analysis_id, models_completed,
                         stages_completed, warnings, results_dir):
    """Write final done-status progress JSON."""
    progress_path = os.path.join(results_dir,
                                 f"{analysis_id}_progress.json")
    done_progress = {
        "status": "done",
        "stage": "complete",
        "models_completed": models_completed[:_MAX_MODELS],
        "stages_completed": stages_completed[:_MAX_STAGES],
        "warnings": warnings[:_MAX_WARNINGS],
    }
    with open(progress_path, "w") as f:
        json.dump(done_progress, f)


def _build_nli_pairs(claims, evidence_map):
    """Build NLI claim-passage pairs from claims and evidence.

    When no real evidence exists, creates a default pair with
    "[no evidence provided]" so the NLI model can still run.
    """
    from id_utils import make_pair_id

    pairs = []
    bound = min(len(claims), _MAX_CLAIMS)
    for i in range(bound):
        claim = claims[i]
        cid = claim.get("claim_id", "")
        passages = evidence_map.get(cid, [])
        if not passages:
            pid = f"p_default_{cid}"
            pair_id = make_pair_id(cid, pid)
            pairs.append({
                "pair_id": pair_id,
                "claim_id": cid,
                "passage_id": pid,
                "claim_text": claim.get("claim_text", ""),
                "passage_text": "[no evidence provided]",
            })
        else:
            for p in passages[:_MAX_PASSAGES_PER_CLAIM]:
                pid = p.get("passage_id", "")
                pair_id = make_pair_id(cid, pid)
                pairs.append({
                    "pair_id": pair_id,
                    "claim_id": cid,
                    "passage_id": pid,
                    "claim_text": claim.get("claim_text", ""),
                    "passage_text": p.get("text", ""),
                })
    return pairs


def _build_rerank_items(claims, evidence_map):
    """Build rerank items from claims and evidence."""
    items = []
    bound = min(len(claims), _MAX_CLAIMS)
    for i in range(bound):
        c = claims[i]
        cid = c.get("claim_id", "")
        passages = evidence_map.get(cid, [])
        if not passages:
            passages = [{"passage_id": f"p_default_{cid}",
                         "text": "[no evidence provided]"}]
        items.append({
            "claim_id": cid,
            "claim_text": c.get("claim_text", ""),
            "passages": passages[:_MAX_PASSAGES_PER_CLAIM],
        })
    return items


def _singleton_clusters(claims):
    """Fallback: create one cluster per claim."""
    from id_utils import make_cluster_id

    clusters = []
    bound = min(len(claims), _MAX_CLAIMS)
    for i in range(bound):
        c = claims[i]
        cid = c["claim_id"]
        cluster_id = make_cluster_id([cid])
        clusters.append({
            "cluster_id": cluster_id,
            "claim_ids": [cid],
            "representative_claim_id": cid,
            "representative_text": c.get("claim_text", ""),
        })
    return clusters


def _neutral_nli_results(pairs):
    """Fallback: return neutral probabilities for all pairs."""
    results = []
    bound = min(len(pairs), _MAX_CLAIMS)
    for i in range(bound):
        p = pairs[i]
        results.append({
            "pair_id": p.get("pair_id", ""),
            "claim_id": p.get("claim_id", ""),
            "passage_id": p.get("passage_id", ""),
            "label": "neutral",
            "probs": {"entailment": 0.33,
                      "contradiction": 0.33, "neutral": 0.34},
        })
    return results


def _build_final_result(analysis_id, prompt, models, claims, clusters,
                        evidence, nli_results, cluster_scores, coords3d,
                        safe_answer, model_metrics_list, warnings):
    """Build the flat AnalysisResult dict per spec section 7."""
    return {
        "schema_version": "1.0",
        "analysis_id": analysis_id,
        "prompt": prompt,
        "models": models,
        "claims": claims,
        "clusters": clusters,
        "evidence": evidence,
        "nli_results": nli_results,
        "cluster_scores": cluster_scores,
        "coords3d": coords3d,
        "safe_answer": safe_answer,
        "model_metrics": model_metrics_list,
        "warnings": warnings[:_MAX_WARNINGS],
    }


# ---------------------------------------------------------------------------
# Stage runners — each handles one or two pipeline stages
# ---------------------------------------------------------------------------


def _run_llm_stage(analysis_id, prompt, api_keys, state):
    """Stage 1: Call LLMs concurrently. Returns models_output list."""
    _write_progress(analysis_id, "llm_calls",
                    state["models_completed"],
                    state["stages_completed"],
                    state["warnings"], state["results_dir"])

    llm_responses, llm_warnings = call_all_llms(prompt, api_keys)
    state["warnings"].extend(llm_warnings[:_MAX_WARNINGS])

    for i in range(min(len(llm_responses), _MAX_MODELS)):
        mid = llm_responses[i].get("model_id", "")
        if mid:
            state["models_completed"].append(mid)

    state["stages_completed"].append("llm_calls")

    models_output = []
    for i in range(min(len(llm_responses), _MAX_MODELS)):
        r = llm_responses[i]
        models_output.append({
            "model_id": r["model_id"],
            "response_text": r["response_text"],
        })
    return models_output


def _run_extract_stage(analysis_id, models_output, ml_functions, state):
    """Stage 2: Extract claims from LLM responses."""
    _write_progress(analysis_id, "extract_claims",
                    state["models_completed"],
                    state["stages_completed"],
                    state["warnings"], state["results_dir"])
    try:
        extract_resp = ml_functions["extract_claims"]({
            "schema_version": "1.0",
            "analysis_id": analysis_id,
            "responses": models_output,
        })
        claims = extract_resp.get("claims", [])
        state["warnings"].extend(
            extract_resp.get("warnings", [])[:_MAX_WARNINGS])
    except Exception as exc:
        state["warnings"].append(f"extract_claims failed: {str(exc)}")
        claims = []

    state["stages_completed"].append("extract_claims")
    return claims


def _run_embed_stage(analysis_id, claims, ml_functions, state):
    """Stage 3: Embed claims into vectors."""
    _write_progress(analysis_id, "embed_claims",
                    state["models_completed"],
                    state["stages_completed"],
                    state["warnings"], state["results_dir"])

    claim_inputs = []
    for i in range(min(len(claims), _MAX_CLAIMS)):
        c = claims[i]
        claim_inputs.append({
            "claim_id": c["claim_id"],
            "claim_text": c["claim_text"],
        })

    try:
        embed_resp = ml_functions["embed_claims"]({
            "schema_version": "1.0",
            "analysis_id": analysis_id,
            "claims": claim_inputs,
        })
        vectors = embed_resp.get("vectors", {})
        state["warnings"].extend(
            embed_resp.get("warnings", [])[:_MAX_WARNINGS])
    except Exception as exc:
        state["warnings"].append(f"embed_claims failed: {str(exc)}")
        vectors = {}

    state["stages_completed"].append("embed_claims")
    return vectors


def _run_cluster_umap_stage(analysis_id, claims, vectors,
                            ml_functions, state):
    """Stages 4-5: Cluster claims and compute UMAP coordinates."""
    # --- Stage 4: Cluster ---
    _write_progress(analysis_id, "cluster_claims",
                    state["models_completed"],
                    state["stages_completed"],
                    state["warnings"], state["results_dir"])

    claim_metadata = {}
    for i in range(min(len(claims), _MAX_CLAIMS)):
        c = claims[i]
        claim_metadata[c["claim_id"]] = {
            "model_id": c["model_id"],
            "claim_text": c["claim_text"],
        }

    try:
        cluster_resp = ml_functions["cluster_claims"]({
            "schema_version": "1.0",
            "analysis_id": analysis_id,
            "vectors": vectors,
            "claims": claim_metadata,
        })
        clusters = cluster_resp.get("clusters", [])
        state["warnings"].extend(
            cluster_resp.get("warnings", [])[:_MAX_WARNINGS])
    except Exception as exc:
        state["warnings"].append(f"cluster_claims failed: {str(exc)}")
        clusters = _singleton_clusters(claims)

    state["stages_completed"].append("cluster_claims")

    # --- Stage 5: UMAP ---
    _write_progress(analysis_id, "compute_umap",
                    state["models_completed"],
                    state["stages_completed"],
                    state["warnings"], state["results_dir"])

    try:
        umap_resp = ml_functions["compute_umap"]({
            "schema_version": "1.0",
            "analysis_id": analysis_id,
            "vectors": vectors,
        })
        coords3d = umap_resp.get("coords3d", {})
        state["warnings"].extend(
            umap_resp.get("warnings", [])[:_MAX_WARNINGS])
    except Exception as exc:
        state["warnings"].append(f"compute_umap failed: {str(exc)}")
        coords3d = {}

    state["stages_completed"].append("compute_umap")
    return clusters, coords3d


def _run_evidence_rerank_nli_stage(analysis_id, claims,
                                   ml_functions, state):
    """Stages 6-8: Retrieve evidence, rerank, and NLI verify."""
    # --- Stage 6: Evidence retrieval ---
    evidence_map = retrieve_evidence(claims, analysis_id)

    # --- Stage 7: Rerank ---
    _write_progress(analysis_id, "rerank_evidence",
                    state["models_completed"],
                    state["stages_completed"],
                    state["warnings"], state["results_dir"])

    rerank_items = _build_rerank_items(claims, evidence_map)
    try:
        rerank_resp = ml_functions["rerank_evidence_batch"]({
            "schema_version": "1.0",
            "analysis_id": analysis_id,
            "items": rerank_items,
        })
        state["warnings"].extend(
            rerank_resp.get("warnings", [])[:_MAX_WARNINGS])
    except Exception as exc:
        state["warnings"].append(
            f"rerank_evidence failed: {str(exc)}")

    state["stages_completed"].append("rerank_evidence")

    # --- Stage 8: NLI verify ---
    _write_progress(analysis_id, "nli_verify",
                    state["models_completed"],
                    state["stages_completed"],
                    state["warnings"], state["results_dir"])

    nli_pairs = _build_nli_pairs(claims, evidence_map)
    try:
        nli_resp = ml_functions["nli_verify_batch"]({
            "schema_version": "1.0",
            "analysis_id": analysis_id,
            "pairs": nli_pairs,
        })
        nli_results = nli_resp.get("results", [])
        state["warnings"].extend(
            nli_resp.get("warnings", [])[:_MAX_WARNINGS])
    except Exception as exc:
        state["warnings"].append(f"nli_verify failed: {str(exc)}")
        nli_results = _neutral_nli_results(nli_pairs)

    state["stages_completed"].append("nli_verify")
    return nli_results


def _run_score_stage(analysis_id, claims, clusters,
                     nli_results, ml_functions, state):
    """Stage 9: Score clusters."""
    _write_progress(analysis_id, "score_clusters",
                    state["models_completed"],
                    state["stages_completed"],
                    state["warnings"], state["results_dir"])

    claims_map = {}
    for i in range(min(len(claims), _MAX_CLAIMS)):
        c = claims[i]
        claims_map[c["claim_id"]] = {
            "model_id": c["model_id"],
            "claim_text": c["claim_text"],
        }

    cluster_input = []
    for i in range(min(len(clusters), _MAX_CLUSTER_CAP)):
        cl = clusters[i]
        cluster_input.append({
            "cluster_id": cl["cluster_id"],
            "claim_ids": cl["claim_ids"],
            "representative_claim_id": cl["representative_claim_id"],
            "representative_text": cl["representative_text"],
        })

    try:
        score_resp = ml_functions["score_clusters"]({
            "schema_version": "1.0",
            "analysis_id": analysis_id,
            "clusters": cluster_input,
            "claims": claims_map,
            "nli_results": nli_results,
        })
        cluster_scores = score_resp.get("cluster_scores", [])
        state["warnings"].extend(
            score_resp.get("warnings", [])[:_MAX_WARNINGS])
    except Exception as exc:
        state["warnings"].append(
            f"score_clusters failed: {str(exc)}")
        cluster_scores = []

    state["stages_completed"].append("score_clusters")
    return cluster_scores


def _run_safe_answer_stage(analysis_id, clusters,
                           cluster_scores, state):
    """Stage 10: Build safe answer."""
    _write_progress(analysis_id, "safe_answer",
                    state["models_completed"],
                    state["stages_completed"],
                    state["warnings"], state["results_dir"])

    cluster_dicts = []
    for i in range(min(len(clusters), _MAX_CLUSTER_CAP)):
        cl = clusters[i]
        cluster_dicts.append({
            "cluster_id": cl["cluster_id"],
            "claim_ids": cl["claim_ids"],
            "representative_claim_id": cl.get(
                "representative_claim_id", ""),
            "representative_text": cl.get("representative_text", ""),
        })

    safe_answer_result, sa_warnings = build_safe_answer(
        cluster_dicts, cluster_scores)
    state["warnings"].extend(sa_warnings[:_MAX_WARNINGS])
    state["stages_completed"].append("safe_answer")
    return safe_answer_result


def _run_metrics_stage(analysis_id, claims, cluster_scores,
                       clusters, state):
    """Stage 11: Compute model metrics."""
    _write_progress(analysis_id, "model_metrics",
                    state["models_completed"],
                    state["stages_completed"],
                    state["warnings"], state["results_dir"])

    metrics = compute_model_metrics(claims, cluster_scores, clusters)
    state["stages_completed"].append("model_metrics")
    return metrics


def _early_done(analysis_id, prompt, models_output, claims,
                warnings, results_dir):
    """Build and write an early-return 'done' result (no ML output)."""
    result = _build_final_result(
        analysis_id, prompt, models_output, claims, [], [],
        [], [], {}, None, [], warnings)
    result["status"] = "done"
    _write_result(analysis_id, result, results_dir)
    return result


def _run_ml_stages(analysis_id, prompt, models_output, claims,
                   ml_functions, state):
    """Run stages 3-11 (embed through metrics) and return result."""
    results_dir = state["results_dir"]

    vectors = _run_embed_stage(
        analysis_id, claims, ml_functions, state)
    if not vectors:
        return _early_done(analysis_id, prompt, models_output,
                           claims, state["warnings"], results_dir)

    clusters, coords3d = _run_cluster_umap_stage(
        analysis_id, claims, vectors, ml_functions, state)

    nli_results = _run_evidence_rerank_nli_stage(
        analysis_id, claims, ml_functions, state)

    cluster_scores = _run_score_stage(
        analysis_id, claims, clusters, nli_results,
        ml_functions, state)

    safe_answer_result = _run_safe_answer_stage(
        analysis_id, clusters, cluster_scores, state)

    metrics = _run_metrics_stage(
        analysis_id, claims, cluster_scores, clusters, state)

    result = _build_final_result(
        analysis_id, prompt, models_output, claims, clusters,
        [], nli_results, cluster_scores, coords3d,
        safe_answer_result, metrics, state["warnings"])
    result["status"] = "done"
    _write_result(analysis_id, result, results_dir)
    _write_done_progress(
        analysis_id, state["models_completed"],
        state["stages_completed"], state["warnings"], results_dir)
    return result


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_full_pipeline(analysis_id, prompt, api_keys,
                      ml_functions, results_dir=None,
                      model_outputs=None):
    """Run the complete TruthLens analysis pipeline.

    Args:
        analysis_id: str, unique analysis identifier
        prompt: str, the user's question/prompt
        api_keys: dict with keys "openai", "anthropic", "gemini"
        ml_functions: dict of function name to callable
        results_dir: str, directory for results (default: /results)
        model_outputs: list of pre-computed model outputs (skips LLM stage)

    Returns:
        dict: Full AnalysisResult per spec section 7
    """
    if results_dir is None:
        results_dir = _RESULTS_DIR

    state = {
        "warnings": [],
        "stages_completed": [],
        "models_completed": [],
        "results_dir": results_dir,
    }

    if model_outputs is not None:
        models_output = list(model_outputs)
        state["models_completed"] = [r.get("model_id", "") for r in models_output]
        state["stages_completed"].append("llm_calls")
        _write_progress(analysis_id, "extract_claims",
                        state["models_completed"], state["stages_completed"],
                        state["warnings"], results_dir)
    else:
        models_output = _run_llm_stage(
            analysis_id, prompt, api_keys, state)

    if not models_output:
        _write_error(analysis_id, "No LLM responses received",
                     state["warnings"], results_dir)
        return {"status": "error",
                "error": "No LLM responses received",
                "warnings": state["warnings"]}

    claims = _run_extract_stage(
        analysis_id, models_output, ml_functions, state)

    if not claims:
        return _early_done(analysis_id, prompt, models_output,
                           [], state["warnings"], results_dir)

    return _run_ml_stages(
        analysis_id, prompt, models_output, claims,
        ml_functions, state)
