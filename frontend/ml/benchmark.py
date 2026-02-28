"""Benchmark cold/warm start latency for all 7 TruthLens Modal functions.

Usage:
    python benchmark.py

Requires MODAL_TOKEN_ID and MODAL_TOKEN_SECRET env vars.
"""

import os
import sys
import time

# -- Constants (bounded) ----------------------------------------------------

NUM_WARM_CALLS = 3
MAX_FUNCTIONS = 10
MODAL_APP_NAME = "truthlens-ml"
EMBEDDING_DIM = 8

FUNCTION_NAMES = [
    "extract_claims",
    "embed_claims",
    "cluster_claims",
    "rerank_evidence_batch",
    "nli_verify_batch",
    "compute_umap",
    "score_clusters",
]


# -- Synthetic Payload Generators -------------------------------------------


def make_extract_claims_payload() -> dict:
    """Build a minimal valid ExtractClaimsRequest dict."""
    return {
        "analysis_id": "bench-001",
        "responses": [
            {
                "model_id": "model-a",
                "response_text": "The Earth orbits the Sun.",
            },
        ],
    }


def make_embed_claims_payload() -> dict:
    """Build a minimal valid EmbedClaimsRequest dict."""
    return {
        "analysis_id": "bench-001",
        "claims": [
            {"claim_id": "c1", "claim_text": "The Earth orbits the Sun."},
            {"claim_id": "c2", "claim_text": "Water boils at 100 degrees."},
        ],
        "model_name": "BAAI/bge-large-en-v1.5",
    }


def make_cluster_claims_payload() -> dict:
    """Build a minimal valid ClusterClaimsRequest dict."""
    vec_a = [0.1] * EMBEDDING_DIM
    vec_b = [0.2] * EMBEDDING_DIM
    return {
        "analysis_id": "bench-001",
        "vectors": {"c1": vec_a, "c2": vec_b},
        "claims": {
            "c1": {"model_id": "model-a", "claim_text": "Claim one."},
            "c2": {"model_id": "model-b", "claim_text": "Claim two."},
        },
        "sim_threshold": 0.85,
    }


def make_rerank_evidence_batch_payload() -> dict:
    """Build a minimal valid RerankEvidenceBatchRequest dict."""
    return {
        "analysis_id": "bench-001",
        "items": [
            {
                "claim_id": "c1",
                "claim_text": "The Earth orbits the Sun.",
                "passages": [
                    {"passage_id": "p1", "text": "Earth revolves around the Sun."},
                    {"passage_id": "p2", "text": "Mars is the red planet."},
                ],
            },
        ],
        "top_k": 2,
    }


def make_nli_verify_batch_payload() -> dict:
    """Build a minimal valid NliVerifyBatchRequest dict."""
    return {
        "analysis_id": "bench-001",
        "pairs": [
            {
                "pair_id": "pair-1",
                "claim_id": "c1",
                "passage_id": "p1",
                "claim_text": "The Earth orbits the Sun.",
                "passage_text": "Earth revolves around the Sun annually.",
            },
        ],
        "batch_size": 16,
    }


def make_compute_umap_payload() -> dict:
    """Build a minimal valid ComputeUmapRequest dict.

    Uses 6 vectors so n_samples (6) > n_components (3), avoiding
    UMAP spectral-init failure that forces the zero-coordinate fallback.
    """
    vectors: dict = {}
    for i in range(6):
        vectors[f"c{i + 1}"] = [0.1 * (i + 1)] * EMBEDDING_DIM
    return {
        "analysis_id": "bench-001",
        "vectors": vectors,
        "n_neighbors": 2,
        "min_dist": 0.1,
    }


def make_score_clusters_payload() -> dict:
    """Build a minimal valid ScoreClustersRequest dict."""
    return {
        "analysis_id": "bench-001",
        "clusters": [
            {
                "cluster_id": "cl-1",
                "claim_ids": ["c1", "c2"],
                "representative_claim_id": "c1",
                "representative_text": "The Earth orbits the Sun.",
            },
        ],
        "claims": {
            "c1": {"model_id": "model-a", "claim_text": "The Earth orbits the Sun."},
            "c2": {"model_id": "model-b", "claim_text": "Earth goes around the Sun."},
        },
        "nli_results": [
            {
                "pair_id": "pair-1",
                "claim_id": "c1",
                "passage_id": "p1",
                "label": "entailment",
                "probs": {
                    "entailment": 0.85,
                    "contradiction": 0.05,
                    "neutral": 0.10,
                },
            },
        ],
    }


# -- Payload dispatch table -------------------------------------------------

PAYLOAD_GENERATORS = {
    "extract_claims": make_extract_claims_payload,
    "embed_claims": make_embed_claims_payload,
    "cluster_claims": make_cluster_claims_payload,
    "rerank_evidence_batch": make_rerank_evidence_batch_payload,
    "nli_verify_batch": make_nli_verify_batch_payload,
    "compute_umap": make_compute_umap_payload,
    "score_clusters": make_score_clusters_payload,
}


# -- Timing helper ----------------------------------------------------------


def time_remote_call(fn_ref, payload: dict) -> float:
    """Invoke fn_ref.remote(payload) and return elapsed milliseconds."""
    start = time.monotonic()
    fn_ref.remote(payload)
    end = time.monotonic()
    return (end - start) * 1000.0


# -- Benchmark runner -------------------------------------------------------


def run_benchmark() -> list:
    """Time cold + warm calls for each Modal function.

    Returns a list of result dicts, one per function.
    """
    import modal  # imported here so tests can import generators freely

    results: list = []
    fn_count = 0

    for fn_name in FUNCTION_NAMES:
        if fn_count >= MAX_FUNCTIONS:
            break
        fn_count += 1

        gen = PAYLOAD_GENERATORS.get(fn_name)
        if gen is None:
            continue

        payload = gen()
        fn_ref = modal.Function.from_name(MODAL_APP_NAME, fn_name)

        # Cold call (first invocation)
        cold_ms = time_remote_call(fn_ref, payload)

        # Warm calls
        warm_times: list = []
        for i in range(NUM_WARM_CALLS):
            warm_ms = time_remote_call(fn_ref, payload)
            warm_times.append(warm_ms)

        avg_warm = sum(warm_times) / len(warm_times) if len(warm_times) > 0 else 0.0
        min_warm = min(warm_times) if len(warm_times) > 0 else 0.0
        max_warm = max(warm_times) if len(warm_times) > 0 else 0.0

        results.append({
            "function": fn_name,
            "cold_ms": round(cold_ms, 1),
            "avg_warm_ms": round(avg_warm, 1),
            "min_warm_ms": round(min_warm, 1),
            "max_warm_ms": round(max_warm, 1),
        })

    return results


# -- Summary printer --------------------------------------------------------


def print_summary(results: list) -> None:
    """Print a formatted table of benchmark results."""
    header = (
        f"{'Function':<28} {'Cold (ms)':>10} {'Avg Warm':>10} "
        f"{'Min Warm':>10} {'Max Warm':>10}"
    )
    print(header)
    print("-" * len(header))

    for i in range(len(results)):
        row = results[i]
        print(
            f"{row['function']:<28} {row['cold_ms']:>10.1f} "
            f"{row['avg_warm_ms']:>10.1f} {row['min_warm_ms']:>10.1f} "
            f"{row['max_warm_ms']:>10.1f}"
        )


# -- Main -------------------------------------------------------------------


def main() -> None:
    """Entry point: check credentials, run benchmark, print results."""
    token_id = os.environ.get("MODAL_TOKEN_ID", "")
    token_secret = os.environ.get("MODAL_TOKEN_SECRET", "")

    if len(token_id) == 0 or len(token_secret) == 0:
        print("Skipping benchmark: MODAL_TOKEN_ID / MODAL_TOKEN_SECRET not set.")
        sys.exit(0)

    print(f"Running benchmark with {NUM_WARM_CALLS} warm calls per function...")
    print()

    results = run_benchmark()
    print_summary(results)


if __name__ == "__main__":
    main()
