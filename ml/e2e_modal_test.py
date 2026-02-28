"""End-to-end test for TruthLens ML pipeline via Modal HTTP endpoints.

Sends mock data to all 7 deployed HTTP endpoints and reports results.
Modal uses subdomain-based routing: each web endpoint gets its own
subdomain in the form {workspace}--{app-name}-{function-name}.modal.run

Usage:
    MODAL_API_KEY=xxx python e2e_modal_test.py [--workspace W --app-name A]
"""

import json
import os
import sys
import time
import urllib.request
import urllib.error

from mock_data import build_full_pipeline_data

DEFAULT_WORKSPACE = "willhatfield"
DEFAULT_APP_NAME = "truthlens-ml"

# Phase definitions: (name, endpoint_path, request_key, response_fields)
PHASES = [
    ("extract_claims", "http_extract_claims",
     "extract_request", ["claims"]),
    ("embed_claims", "http_embed_claims",
     "embed_request", ["vectors", "dim"]),
    ("cluster_claims", "http_cluster_claims",
     "cluster_request", ["clusters"]),
    ("rerank_evidence_batch", "http_rerank_evidence_batch",
     "rerank_request", ["rankings"]),
    ("nli_verify_batch", "http_nli_verify_batch",
     "nli_request", ["results"]),
    ("compute_umap", "http_compute_umap",
     "umap_request", ["coords3d"]),
    ("score_clusters", "http_score_clusters",
     "score_request", ["scores"]),
]

MAX_PHASES = 10


def _build_url(workspace: str, app_name: str, function_name: str) -> str:
    """Construct subdomain-based URL for a Modal HTTP endpoint.

    Modal routes each web endpoint to its own subdomain:
        https://{workspace}--{app_name}-{function_name}.modal.run
    Underscores in app_name and function_name are replaced with hyphens.
    """
    label = f"{app_name}-{function_name}".replace("_", "-")
    return f"https://{workspace}--{label}.modal.run"


def _send_request(
    url: str, payload: dict, api_key: str,
) -> tuple:
    """Send POST request and return (status_code, response_dict, elapsed_s).

    Uses urllib to avoid adding requests as a dependency.
    """
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    start = time.time()
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            elapsed = time.time() - start
            data = json.loads(resp.read().decode("utf-8"))
            return (resp.status, data, elapsed)
    except urllib.error.HTTPError as exc:
        elapsed = time.time() - start
        detail = ""
        try:
            detail = exc.read().decode("utf-8")
        except Exception:
            pass
        return (exc.code, {"error": detail}, elapsed)
    except Exception as exc:
        elapsed = time.time() - start
        return (0, {"error": str(exc)}, elapsed)


def _print_summary(results: list) -> None:
    """Print formatted summary table of all phase results."""
    print("\n" + "=" * 70)
    print(f"{'Phase':<25} {'Status':<8} {'Time':>8}  {'Details'}")
    print("-" * 70)

    total_time = 0.0
    pass_count = 0
    for i in range(len(results)):
        if i >= MAX_PHASES:
            break
        name, status, elapsed, detail = results[i]
        total_time += elapsed
        status_str = "OK" if status == 200 else f"ERR {status}"
        if status == 200:
            pass_count += 1
        print(f"  {name:<23} {status_str:<8} {elapsed:>7.2f}s  {detail}")

    print("-" * 70)
    print(
        f"  {'TOTAL':<23} {pass_count}/{len(results):<5} "
        f"{total_time:>7.2f}s"
    )
    print("=" * 70)


def run_e2e(workspace: str, app_name: str, api_key: str) -> int:
    """Execute all E2E phases and return exit code (0=success)."""
    print(f"Workspace: {workspace}")
    print(f"App Name:  {app_name}")
    print(f"API Key:   {'*' * 4}{api_key[-4:]}")
    print()

    pipeline_data = build_full_pipeline_data()
    results = []

    for i in range(len(PHASES)):
        if i >= MAX_PHASES:
            break
        name, endpoint, req_key, resp_fields = PHASES[i]
        url = _build_url(workspace, app_name, endpoint)
        payload = pipeline_data[req_key]

        print(f"[{i + 1}/{len(PHASES)}] {name}...", end=" ", flush=True)
        status, data, elapsed = _send_request(url, payload, api_key)

        if status == 200:
            detail_parts = []
            for j in range(len(resp_fields)):
                field = resp_fields[j]
                if field in data:
                    val = data[field]
                    if isinstance(val, list):
                        detail_parts.append(f"{field}={len(val)}")
                    elif isinstance(val, dict):
                        detail_parts.append(f"{field}={len(val)}")
                    else:
                        detail_parts.append(f"{field}={val}")
            detail = ", ".join(detail_parts)
            print(f"OK ({elapsed:.2f}s) - {detail}")
        else:
            detail = str(data.get("error", ""))[:80]
            print(f"FAILED ({status}) - {detail}")

        results.append((name, status, elapsed, detail))

    _print_summary(results)

    fail_count = 0
    for i in range(len(results)):
        if results[i][1] != 200:
            fail_count += 1
    return 1 if fail_count > 0 else 0


def main() -> None:
    """Entry point: parse args, validate env, run E2E."""
    api_key = os.environ.get("MODAL_API_KEY", "")
    if len(api_key) == 0:
        print("ERROR: MODAL_API_KEY environment variable not set.")
        print("Usage: MODAL_API_KEY=xxx python e2e_modal_test.py")
        sys.exit(1)

    workspace = DEFAULT_WORKSPACE
    app_name = DEFAULT_APP_NAME

    i = 1
    max_args = 10
    while i < len(sys.argv) and i < max_args:
        if sys.argv[i] == "--workspace" and i + 1 < len(sys.argv):
            workspace = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--app-name" and i + 1 < len(sys.argv):
            app_name = sys.argv[i + 1]
            i += 2
        else:
            i += 1

    exit_code = run_e2e(workspace, app_name, api_key)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
