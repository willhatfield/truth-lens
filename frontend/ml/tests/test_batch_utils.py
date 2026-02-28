"""Tests for batch_utils.py — chunking, flattening, edge cases."""

import pytest
from batch_utils import chunk_list, flatten_batch_results


class TestChunkList:
    def test_even_split(self):
        result = chunk_list([1, 2, 3, 4], 2)
        assert result == [[1, 2], [3, 4]]

    def test_uneven_split(self):
        result = chunk_list([1, 2, 3, 4, 5], 2)
        assert result == [[1, 2], [3, 4], [5]]

    def test_batch_size_larger_than_list(self):
        result = chunk_list([1, 2], 10)
        assert result == [[1, 2]]

    def test_single_item(self):
        result = chunk_list([42], 1)
        assert result == [[42]]

    def test_empty_list(self):
        result = chunk_list([], 5)
        assert result == []

    def test_batch_size_one(self):
        result = chunk_list([1, 2, 3], 1)
        assert result == [[1], [2], [3]]

    def test_max_batches_limits_output(self):
        result = chunk_list(list(range(100)), 1, max_batches=3)
        assert len(result) == 3
        assert result == [[0], [1], [2]]

    def test_batch_size_zero_clamped(self):
        result = chunk_list([1, 2, 3], 0)
        assert len(result) == 3

    def test_max_batches_zero_clamped(self):
        result = chunk_list([1, 2], 1, max_batches=0)
        assert len(result) == 1

    def test_string_items(self):
        result = chunk_list(["a", "b", "c"], 2)
        assert result == [["a", "b"], ["c"]]


class TestFlattenBatchResults:
    def test_simple_flatten(self):
        result = flatten_batch_results([[1, 2], [3, 4]])
        assert result == [1, 2, 3, 4]

    def test_single_batch(self):
        result = flatten_batch_results([[1, 2, 3]])
        assert result == [1, 2, 3]

    def test_empty_batches(self):
        result = flatten_batch_results([])
        assert result == []

    def test_empty_inner_batches(self):
        result = flatten_batch_results([[], [1], []])
        assert result == [1]

    def test_max_batches_limits(self):
        batches = [[i] for i in range(100)]
        result = flatten_batch_results(batches, max_batches=5)
        assert result == [0, 1, 2, 3, 4]

    def test_max_batches_zero_clamped(self):
        result = flatten_batch_results([[1, 2]], max_batches=0)
        assert result == [1, 2]

    def test_preserves_order(self):
        result = flatten_batch_results([[3, 1], [4, 1], [5, 9]])
        assert result == [3, 1, 4, 1, 5, 9]
