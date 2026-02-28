# TruthLens ML Pipeline — Makefile
# Runs unit tests, integration harness, and Modal deployment commands.

.PHONY: test harness test-all e2e mock-data lint deploy serve clean \
        run-extract run-embed run-cluster run-rerank \
        run-nli run-umap run-score

# ---------------------------------------------------------------------------
# Testing
# ---------------------------------------------------------------------------

test:                          ## Run all pytest unit tests (202 tests)
	python -m pytest tests/ -v

harness:                       ## Run local integration harness (7-function pipeline)
	python test_harness.py

test-all: test harness         ## Run unit tests then integration harness

e2e:                           ## Run E2E test against deployed Modal endpoints
	MODAL_API_KEY=$(MODAL_API_KEY) python e2e_modal_test.py

mock-data:                     ## Print mock pipeline data summary
	python -c "from mock_data import build_full_pipeline_data; d = build_full_pipeline_data(); print(f'{len(d)} payloads: {list(d.keys())}')"

# ---------------------------------------------------------------------------
# Modal deployment
# ---------------------------------------------------------------------------

deploy:                        ## Deploy all functions to Modal
	modal deploy modal_app.py

serve:                         ## Start Modal dev server (hot-reload)
	modal serve modal_app.py

# ---------------------------------------------------------------------------
# Modal run — individual functions (requires Modal auth)
# ---------------------------------------------------------------------------

run-extract:                   ## Run extract_claims on Modal
	modal run modal_app.py::extract_claims

run-embed:                     ## Run embed_claims on Modal
	modal run modal_app.py::embed_claims

run-cluster:                   ## Run cluster_claims on Modal
	modal run modal_app.py::cluster_claims

run-rerank:                    ## Run rerank_evidence_batch on Modal
	modal run modal_app.py::rerank_evidence_batch

run-nli:                       ## Run nli_verify_batch on Modal
	modal run modal_app.py::nli_verify_batch

run-umap:                      ## Run compute_umap on Modal
	modal run modal_app.py::compute_umap

run-score:                     ## Run score_clusters on Modal
	modal run modal_app.py::score_clusters

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

lint:                          ## Run ruff linter (if installed)
	python -m ruff check .

clean:                         ## Remove __pycache__ and .pytest_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true

help:                          ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
