"""Tests for batch_config.py -- profiles, lookup, and validation."""

import pytest
from pydantic import ValidationError

from batch_config import (
    AGGRESSIVE,
    BALANCED,
    BatchProfile,
    CONSERVATIVE,
    get_profile,
)
from batch_utils import MAX_BATCHES_LIMIT


# ── Profile existence and values ─────────────────────────────────────────────

class TestProfileValues:
    def test_conservative_values(self):
        assert CONSERVATIVE.embed_batch_size == 32
        assert CONSERVATIVE.nli_batch_size == 8
        assert CONSERVATIVE.rerank_batch_size == 16
        assert CONSERVATIVE.chunk_max_batches == 500

    def test_balanced_values(self):
        assert BALANCED.embed_batch_size == 64
        assert BALANCED.nli_batch_size == 16
        assert BALANCED.rerank_batch_size == 32
        assert BALANCED.chunk_max_batches == 1000

    def test_aggressive_values(self):
        assert AGGRESSIVE.embed_batch_size == 128
        assert AGGRESSIVE.nli_batch_size == 32
        assert AGGRESSIVE.rerank_batch_size == 64
        assert AGGRESSIVE.chunk_max_batches == 2000


# ── All profiles have positive batch sizes ───────────────────────────────────

class TestAllProfilesPositive:
    @pytest.mark.parametrize("profile", [CONSERVATIVE, BALANCED, AGGRESSIVE])
    def test_embed_positive(self, profile):
        assert profile.embed_batch_size > 0

    @pytest.mark.parametrize("profile", [CONSERVATIVE, BALANCED, AGGRESSIVE])
    def test_nli_positive(self, profile):
        assert profile.nli_batch_size > 0

    @pytest.mark.parametrize("profile", [CONSERVATIVE, BALANCED, AGGRESSIVE])
    def test_rerank_positive(self, profile):
        assert profile.rerank_batch_size > 0

    @pytest.mark.parametrize("profile", [CONSERVATIVE, BALANCED, AGGRESSIVE])
    def test_chunk_max_batches_positive(self, profile):
        assert profile.chunk_max_batches > 0

    @pytest.mark.parametrize("profile", [CONSERVATIVE, BALANCED, AGGRESSIVE])
    def test_all_fields_are_int(self, profile):
        assert isinstance(profile.embed_batch_size, int)
        assert isinstance(profile.nli_batch_size, int)
        assert isinstance(profile.rerank_batch_size, int)
        assert isinstance(profile.chunk_max_batches, int)


# ── get_profile lookup ───────────────────────────────────────────────────────

class TestGetProfile:
    def test_returns_conservative(self):
        assert get_profile("conservative") is CONSERVATIVE

    def test_returns_balanced(self):
        assert get_profile("balanced") is BALANCED

    def test_returns_aggressive(self):
        assert get_profile("aggressive") is AGGRESSIVE

    def test_unknown_name_returns_balanced(self):
        assert get_profile("turbo") is BALANCED

    def test_empty_string_returns_balanced(self):
        assert get_profile("") is BALANCED

    def test_case_insensitive_upper(self):
        assert get_profile("CONSERVATIVE") is CONSERVATIVE

    def test_case_insensitive_mixed(self):
        assert get_profile("Aggressive") is AGGRESSIVE

    def test_whitespace_stripped_leading(self):
        assert get_profile("  balanced") is BALANCED

    def test_whitespace_stripped_trailing(self):
        assert get_profile("aggressive  ") is AGGRESSIVE

    def test_whitespace_stripped_both(self):
        assert get_profile("  conservative  ") is CONSERVATIVE

    def test_whitespace_and_case_combined(self):
        assert get_profile("  BALANCED  ") is BALANCED


# ── Validation rejects bad values ────────────────────────────────────────────

class TestValidation:
    def test_zero_embed_batch_size_rejected(self):
        with pytest.raises(ValidationError):
            BatchProfile(
                embed_batch_size=0,
                nli_batch_size=8,
                rerank_batch_size=16,
                chunk_max_batches=500,
            )

    def test_negative_embed_batch_size_rejected(self):
        with pytest.raises(ValidationError):
            BatchProfile(
                embed_batch_size=-1,
                nli_batch_size=8,
                rerank_batch_size=16,
                chunk_max_batches=500,
            )

    def test_zero_nli_batch_size_rejected(self):
        with pytest.raises(ValidationError):
            BatchProfile(
                embed_batch_size=32,
                nli_batch_size=0,
                rerank_batch_size=16,
                chunk_max_batches=500,
            )

    def test_negative_nli_batch_size_rejected(self):
        with pytest.raises(ValidationError):
            BatchProfile(
                embed_batch_size=32,
                nli_batch_size=-5,
                rerank_batch_size=16,
                chunk_max_batches=500,
            )

    def test_zero_rerank_batch_size_rejected(self):
        with pytest.raises(ValidationError):
            BatchProfile(
                embed_batch_size=32,
                nli_batch_size=8,
                rerank_batch_size=0,
                chunk_max_batches=500,
            )

    def test_negative_rerank_batch_size_rejected(self):
        with pytest.raises(ValidationError):
            BatchProfile(
                embed_batch_size=32,
                nli_batch_size=8,
                rerank_batch_size=-10,
                chunk_max_batches=500,
            )

    def test_zero_chunk_max_batches_rejected(self):
        with pytest.raises(ValidationError):
            BatchProfile(
                embed_batch_size=32,
                nli_batch_size=8,
                rerank_batch_size=16,
                chunk_max_batches=0,
            )

    def test_negative_chunk_max_batches_rejected(self):
        with pytest.raises(ValidationError):
            BatchProfile(
                embed_batch_size=32,
                nli_batch_size=8,
                rerank_batch_size=16,
                chunk_max_batches=-1,
            )

    def test_chunk_max_batches_exceeds_limit_rejected(self):
        with pytest.raises(ValidationError):
            BatchProfile(
                embed_batch_size=32,
                nli_batch_size=8,
                rerank_batch_size=16,
                chunk_max_batches=MAX_BATCHES_LIMIT + 1,
            )

    def test_chunk_max_batches_at_limit_accepted(self):
        profile = BatchProfile(
            embed_batch_size=32,
            nli_batch_size=8,
            rerank_batch_size=16,
            chunk_max_batches=MAX_BATCHES_LIMIT,
        )
        assert profile.chunk_max_batches == MAX_BATCHES_LIMIT
