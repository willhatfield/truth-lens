"""Tests for e2e_modal_test module structure and configuration."""

import pytest

from e2e_modal_test import (
    PHASES,
    _build_url,
    _send_request,
    _print_summary,
    run_e2e,
    DEFAULT_WORKSPACE,
    DEFAULT_APP_NAME,
)
from e2e_request_builders import VALID_BUILDER_KEYS


class TestPhaseConfiguration:
    def test_seven_phases_defined(self):
        assert len(PHASES) == 7

    def test_phase_names_match_core_functions(self):
        expected = [
            "extract_claims",
            "embed_claims",
            "cluster_claims",
            "rerank_evidence_batch",
            "nli_verify_batch",
            "compute_umap",
            "score_clusters",
        ]
        actual = [p[0] for p in PHASES]
        assert actual == expected

    def test_each_phase_has_four_elements(self):
        for i in range(len(PHASES)):
            assert len(PHASES[i]) == 4, f"Phase {i} should have 4 elements"

    def test_endpoint_names_start_with_http(self):
        for i in range(len(PHASES)):
            assert PHASES[i][1].startswith("http_"), (
                f"Phase {PHASES[i][0]} endpoint should start with http_"
            )

    def test_builder_keys_are_valid(self):
        for i in range(len(PHASES)):
            builder_key = PHASES[i][2]
            assert builder_key in VALID_BUILDER_KEYS, (
                f"Phase {PHASES[i][0]} has unknown builder key '{builder_key}'"
            )


class TestUrlBuilding:
    def test_build_url_basic(self):
        url = _build_url("myworkspace", "myapp", "http_extract_claims")
        assert url == "https://myworkspace--myapp-http-extract-claims.modal.run"

    def test_build_url_underscores_become_hyphens(self):
        url = _build_url("ws", "my_app", "http_nli_verify_batch")
        assert url == "https://ws--my-app-http-nli-verify-batch.modal.run"

    def test_build_url_with_defaults(self):
        url = _build_url(
            DEFAULT_WORKSPACE, DEFAULT_APP_NAME, "http_embed_claims",
        )
        assert url == (
            f"https://{DEFAULT_WORKSPACE}--"
            f"{DEFAULT_APP_NAME}-http-embed-claims.modal.run"
        )

    def test_default_workspace_not_empty(self):
        assert len(DEFAULT_WORKSPACE) > 0

    def test_default_app_name_not_empty(self):
        assert len(DEFAULT_APP_NAME) > 0
