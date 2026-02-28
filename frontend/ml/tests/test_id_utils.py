"""Tests for id_utils.py -- deterministic SHA1 ID generation."""

import re

import pytest
from id_utils import make_claim_id, make_cluster_id, make_pair_id


HEX_PATTERN = re.compile(r"^[0-9a-f]{40}$")


class TestMakeClaimId:
    def test_prefix(self):
        result = make_claim_id("a1", "m1", "The sky is blue")
        assert result.startswith("c_")

    def test_deterministic(self):
        id1 = make_claim_id("a1", "m1", "The sky is blue")
        id2 = make_claim_id("a1", "m1", "The sky is blue")
        assert id1 == id2

    def test_different_inputs_different_output(self):
        id1 = make_claim_id("a1", "m1", "The sky is blue")
        id2 = make_claim_id("a1", "m1", "The sky is green")
        assert id1 != id2

    def test_valid_hex_after_prefix(self):
        result = make_claim_id("a1", "m1", "The sky is blue")
        hex_part = result[2:]
        assert HEX_PATTERN.match(hex_part)


class TestMakeClusterId:
    def test_prefix(self):
        result = make_cluster_id(["c_abc", "c_def"])
        assert result.startswith("cl_")

    def test_order_independent(self):
        id1 = make_cluster_id(["c_abc", "c_def", "c_ghi"])
        id2 = make_cluster_id(["c_ghi", "c_abc", "c_def"])
        assert id1 == id2

    def test_valid_hex_after_prefix(self):
        result = make_cluster_id(["c_abc"])
        hex_part = result[3:]
        assert HEX_PATTERN.match(hex_part)


class TestMakePairId:
    def test_prefix(self):
        result = make_pair_id("c_abc", "p_def")
        assert result.startswith("nli_")

    def test_deterministic(self):
        id1 = make_pair_id("c_abc", "p_def")
        id2 = make_pair_id("c_abc", "p_def")
        assert id1 == id2

    def test_valid_hex_after_prefix(self):
        result = make_pair_id("c_abc", "p_def")
        hex_part = result[4:]
        assert HEX_PATTERN.match(hex_part)
