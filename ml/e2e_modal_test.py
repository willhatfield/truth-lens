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

from e2e_request_builders import build_request, CHAIN_DEPS, VALID_BUILDER_KEYS

DEFAULT_WORKSPACE = "willhatfield"
DEFAULT_APP_NAME = "truthlens-ml"

# Phase definitions: (name, endpoint_path, builder_key, response_fields)
PHASES = [
    ("extract_claims", "http_extract_claims",
     "extract", ["claims"]),
    ("embed_claims", "http_embed_claims",
     "embed", ["vectors", "dim"]),
    ("cluster_claims", "http_cluster_claims",
     "cluster", ["clusters"]),
    ("rerank_evidence_batch", "http_rerank_evidence_batch",
     "rerank", ["rankings"]),
    ("nli_verify_batch", "http_nli_verify_batch",
     "nli", ["results"]),
    ("compute_umap", "http_compute_umap",
     "umap", ["coords3d"]),
    ("score_clusters", "http_score_clusters",
     "score", ["scores"]),
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


MAX_RESP_FIELDS = 20


def _validate_response_fields(data: dict, resp_fields: list) -> tuple:
    """Check that all expected fields exist in *data*.

    Returns (is_valid, detail_str).  If any field in resp_fields is
    missing, returns (False, "MISSING: field1, field2, ...").
    """
    missing = []
    for i in range(len(resp_fields)):
        if i >= MAX_RESP_FIELDS:
            break
        if resp_fields[i] not in data:
            missing.append(resp_fields[i])
    if len(missing) > 0:
        return (False, "MISSING: " + ", ".join(missing))
    return (True, "")


def _is_chained(builder_key: str, responses: dict) -> bool:
    """Return True if all upstream deps for *builder_key* are present."""
    deps = CHAIN_DEPS.get(builder_key, [])
    if len(deps) == 0:
        return True
    for i in range(len(deps)):
        if deps[i] not in responses:
            return False
    return True


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
        if status == 200:
            status_str = "OK"
            pass_count += 1
        elif status == -1:
            status_str = "FIELDS"
        else:
            status_str = f"ERR {status}"
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

    collected_responses = {}
    results = []

    for i in range(len(PHASES)):
        if i >= MAX_PHASES:
            break
        name, endpoint, builder_key, resp_fields = PHASES[i]
        url = _build_url(workspace, app_name, endpoint)
        payload = build_request(builder_key, collected_responses)

        chained = _is_chained(builder_key, collected_responses)
        tag = "[chained]" if chained else "[mock]"
        print(
            f"[{i + 1}/{len(PHASES)}] {name} {tag}...",
            end=" ",
            flush=True,
        )
        status, data, elapsed = _send_request(url, payload, api_key)

        detail = ""
        if status == 200:
            valid, field_detail = _validate_response_fields(
                data, resp_fields,
            )
            if not valid:
                status = -1
                detail = field_detail
                print(f"FIELDS ({elapsed:.2f}s) - {detail}")
            else:
                detail_parts = []
                for j in range(len(resp_fields)):
                    if j >= MAX_RESP_FIELDS:
                        break
                    field = resp_fields[j]
                    val = data[field]
                    if isinstance(val, list):
                        detail_parts.append(f"{field}={len(val)}")
                    elif isinstance(val, dict):
                        detail_parts.append(f"{field}={len(val)}")
                    else:
                        detail_parts.append(f"{field}={val}")
                detail = ", ".join(detail_parts)
                print(f"OK ({elapsed:.2f}s) - {detail}")
                collected_responses[name] = data
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
