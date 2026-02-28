"""Tests for claim_extraction.py -- sentence splitting edge cases."""

import pytest
from claim_extraction import sentence_split_claims, MAX_SENTENCES


class TestSentenceSplitClaims:
    def test_simple_sentences(self):
        text = "The sky is blue. Grass is green. Water is wet."
        result = sentence_split_claims(text)
        assert result == ["The sky is blue", "Grass is green", "Water is wet"]

    def test_question_mark_delimiter(self):
        text = "Is the sky blue? Yes it is."
        result = sentence_split_claims(text)
        assert "Is the sky blue" in result
        assert "Yes it is" in result

    def test_exclamation_mark_delimiter(self):
        text = "Amazing! This works great."
        result = sentence_split_claims(text)
        assert "Amazing" in result
        assert "This works great" in result

    def test_empty_string(self):
        result = sentence_split_claims("")
        assert result == []

    def test_single_sentence(self):
        text = "The sky is blue"
        result = sentence_split_claims(text)
        assert result == ["The sky is blue"]

    def test_whitespace_only_segments_skipped(self):
        text = "Hello.   . .World."
        result = sentence_split_claims(text)
        assert result == ["Hello", "World"]

    def test_trailing_period(self):
        text = "Hello world."
        result = sentence_split_claims(text)
        assert result == ["Hello world"]
        assert len(result) == 1

    def test_unicode_text(self):
        text = "Caf\u00e9 is great. \u00dcber cool."
        result = sentence_split_claims(text)
        assert len(result) == 2

    def test_max_sentences_bound(self):
        parts = []
        for i in range(MAX_SENTENCES + 100):
            parts.append(f"Claim {i}")
        text = ". ".join(parts) + "."
        result = sentence_split_claims(text)
        assert len(result) <= MAX_SENTENCES

    def test_newlines_in_text(self):
        text = "Line one.\nLine two."
        result = sentence_split_claims(text)
        assert len(result) == 2
