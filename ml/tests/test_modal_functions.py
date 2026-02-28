"""Unit tests for modal_app.py functions — mocked ML models."""

import sys
import pytest
from unittest.mock import patch, MagicMock
import numpy as np

from modal_app import (
    embed_claims,
    rerank_evidence,
    nli_verify,
    cluster_claims,
    compute_umap,
    _softmax,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

class TestSoftmax:
    def test_uniform(self):
        result = _softmax([1.0, 1.0, 1.0])
        for val in result:
            assert abs(val - 1.0 / 3.0) < 1e-6

    def test_dominant(self):
        result = _softmax([10.0, 0.0, 0.0])
        assert result[0] > 0.99

    def test_single(self):
        result = _softmax([5.0])
        assert abs(result[0] - 1.0) < 1e-6

    def test_negative_values(self):
        result = _softmax([-1.0, -2.0, -3.0])
        total = sum(result)
        assert abs(total - 1.0) < 1e-6
        assert result[0] > result[1] > result[2]


# ── embed_claims ──────────────────────────────────────────────────────────────

class TestEmbedClaims:
    def test_valid_embedding(self):
        mock_model = MagicMock()
        fake_embeddings = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        mock_model.encode.return_value = fake_embeddings

        mock_st_module = MagicMock()
        mock_st_module.SentenceTransformer.return_value = mock_model

        with patch.dict("sys.modules", {"sentence_transformers": mock_st_module}):
            result = embed_claims.local(
                {"claim_texts": ["hello", "world"], "batch_size": 2}
            )

        assert result["error"] is None
        assert len(result["vectors"]) == 2
        assert result["dimension"] == 3

    def test_batching_multiple_batches(self):
        mock_model = MagicMock()
        batch1 = np.array([[0.1, 0.2]])
        batch2 = np.array([[0.3, 0.4]])
        mock_model.encode.side_effect = [batch1, batch2]

        mock_st_module = MagicMock()
        mock_st_module.SentenceTransformer.return_value = mock_model

        with patch.dict("sys.modules", {"sentence_transformers": mock_st_module}):
            result = embed_claims.local(
                {"claim_texts": ["a", "b"], "batch_size": 1}
            )

        assert result["error"] is None
        assert len(result["vectors"]) == 2

    def test_invalid_payload(self):
        result = embed_claims.local({"claim_texts": []})
        assert result["error"] is not None

    def test_missing_field(self):
        result = embed_claims.local({})
        assert result["error"] is not None

    def test_model_load_failure(self):
        mock_st_module = MagicMock()
        mock_st_module.SentenceTransformer.side_effect = RuntimeError("GPU OOM")

        with patch.dict("sys.modules", {"sentence_transformers": mock_st_module}):
            result = embed_claims.local({"claim_texts": ["hello"]})

        assert result["error"] is not None
        assert "embed_claims failed" in result["error"]
        assert result["vectors"] == []


# ── rerank_evidence ───────────────────────────────────────────────────────────

class TestRerankEvidence:
    def test_valid_rerank(self):
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.1, 0.9, 0.5])

        mock_st_module = MagicMock()
        mock_st_module.CrossEncoder.return_value = mock_model

        with patch.dict("sys.modules", {"sentence_transformers": mock_st_module}):
            result = rerank_evidence.local({
                "claim": "test claim",
                "passages": ["p1", "p2", "p3"],
                "top_k": 2,
            })

        assert result["error"] is None
        assert len(result["ranked_passages"]) == 2
        assert result["ranked_passages"][0]["score"] >= result["ranked_passages"][1]["score"]

    def test_top_k_larger_than_passages(self):
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.5, 0.8])

        mock_st_module = MagicMock()
        mock_st_module.CrossEncoder.return_value = mock_model

        with patch.dict("sys.modules", {"sentence_transformers": mock_st_module}):
            result = rerank_evidence.local({
                "claim": "test",
                "passages": ["p1", "p2"],
                "top_k": 10,
            })

        assert result["error"] is None
        assert len(result["ranked_passages"]) == 2

    def test_invalid_payload(self):
        result = rerank_evidence.local({"claim": "", "passages": ["p"]})
        assert result["error"] is not None

    def test_missing_passages(self):
        result = rerank_evidence.local({"claim": "test"})
        assert result["error"] is not None

    def test_model_failure_fallback(self):
        mock_st_module = MagicMock()
        mock_st_module.CrossEncoder.side_effect = RuntimeError("load fail")

        with patch.dict("sys.modules", {"sentence_transformers": mock_st_module}):
            result = rerank_evidence.local({
                "claim": "test",
                "passages": ["p1", "p2"],
                "top_k": 2,
            })

        assert result["error"] is not None
        assert "returning original order" in result["error"]
        assert len(result["ranked_passages"]) == 2
        assert result["ranked_passages"][0]["index"] == 0


# ── nli_verify ────────────────────────────────────────────────────────────────

class TestNliVerify:
    def test_invalid_payload_empty_pairs(self):
        result = nli_verify.local({"pairs": []})
        assert result["error"] is not None

    def test_invalid_payload_missing_field(self):
        result = nli_verify.local({})
        assert result["error"] is not None

    def test_invalid_pair_empty_premise(self):
        result = nli_verify.local({
            "pairs": [{"premise": "", "hypothesis": "h"}]
        })
        assert result["error"] is not None

    def test_valid_nli(self):
        mock_tokenizer = MagicMock()
        mock_inputs = MagicMock()
        mock_inputs.to.return_value = mock_inputs
        mock_tokenizer.return_value = mock_inputs

        mock_model = MagicMock()
        mock_outputs = MagicMock()
        # Simulate logits: [contradiction, neutral, entailment]
        # Make a plain list so .cpu().tolist() works
        mock_logits = MagicMock()
        mock_logits.cpu.return_value = mock_logits
        mock_logits.tolist.return_value = [[2.0, 0.5, 0.1]]
        mock_outputs.logits = mock_logits
        mock_model.return_value = mock_outputs
        mock_model.eval.return_value = None
        mock_model.to.return_value = mock_model

        mock_transformers = MagicMock()
        mock_transformers.AutoTokenizer.from_pretrained.return_value = mock_tokenizer
        mock_transformers.AutoModelForSequenceClassification.from_pretrained.return_value = mock_model

        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        mock_torch.no_grad.return_value = MagicMock(
            __enter__=MagicMock(return_value=None),
            __exit__=MagicMock(return_value=False),
        )

        with patch.dict("sys.modules", {
            "transformers": mock_transformers,
            "torch": mock_torch,
        }):
            result = nli_verify.local({
                "pairs": [{"premise": "sky is blue", "hypothesis": "daytime"}],
                "batch_size": 8,
            })

        assert result["error"] is None
        assert len(result["results"]) == 1
        assert result["results"][0]["label"] in ["contradiction", "neutral", "entailment"]

    def test_model_failure_neutral_fallback(self):
        mock_transformers = MagicMock()
        mock_transformers.AutoTokenizer.from_pretrained.side_effect = RuntimeError("fail")

        mock_torch = MagicMock()

        with patch.dict("sys.modules", {
            "transformers": mock_transformers,
            "torch": mock_torch,
        }):
            result = nli_verify.local({
                "pairs": [
                    {"premise": "p1", "hypothesis": "h1"},
                    {"premise": "p2", "hypothesis": "h2"},
                ],
            })

        assert result["error"] is not None
        assert "returning neutral" in result["error"]
        assert len(result["results"]) == 2
        for r in result["results"]:
            assert r["label"] == "neutral"


# ── cluster_claims ────────────────────────────────────────────────────────────

class TestClusterClaims:
    def test_valid_clustering(self):
        vectors = [
            [1.0, 0.0, 0.0],
            [0.99, 0.01, 0.0],
            [0.0, 0.0, 1.0],
            [0.01, 0.0, 0.99],
        ]
        result = cluster_claims.local({
            "vectors": vectors,
            "threshold": 0.1,
        })
        assert result["error"] is None
        assert result["num_clusters"] >= 1
        all_indices = []
        for cluster in result["clusters"]:
            all_indices.extend(cluster)
        assert sorted(all_indices) == [0, 1, 2, 3]

    def test_single_vector(self):
        result = cluster_claims.local({
            "vectors": [[1.0, 2.0, 3.0]],
            "threshold": 0.5,
        })
        assert result["error"] is None
        assert result["num_clusters"] == 1
        assert result["clusters"] == [[0]]

    def test_two_identical_vectors(self):
        result = cluster_claims.local({
            "vectors": [[1.0, 0.0], [1.0, 0.0]],
            "threshold": 0.5,
        })
        assert result["error"] is None
        assert result["num_clusters"] >= 1

    def test_invalid_payload(self):
        result = cluster_claims.local({"vectors": []})
        assert result["error"] is not None


# ── compute_umap ──────────────────────────────────────────────────────────────

class TestComputeUmap:
    def test_valid_umap(self):
        mock_reducer = MagicMock()
        mock_reducer.fit_transform.return_value = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ])

        mock_umap_module = MagicMock()
        mock_umap_module.UMAP.return_value = mock_reducer

        with patch.dict("sys.modules", {"umap": mock_umap_module}):
            result = compute_umap.local({
                "vectors": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
                "n_neighbors": 2,
                "min_dist": 0.1,
            })

        assert result["error"] is None
        assert len(result["coords_3d"]) == 2
        assert result["coords_3d"][0]["x"] == 1.0

    def test_invalid_payload(self):
        result = compute_umap.local({"vectors": []})
        assert result["error"] is not None

    def test_missing_vectors(self):
        result = compute_umap.local({})
        assert result["error"] is not None

    def test_umap_failure_zero_fallback(self):
        mock_umap_module = MagicMock()
        mock_umap_module.UMAP.side_effect = RuntimeError("UMAP fail")

        with patch.dict("sys.modules", {"umap": mock_umap_module}):
            result = compute_umap.local({
                "vectors": [[0.1, 0.2], [0.3, 0.4]],
                "n_neighbors": 2,
                "min_dist": 0.1,
            })

        assert result["error"] is not None
        assert "returning zeros" in result["error"]
        assert len(result["coords_3d"]) == 2
        for pt in result["coords_3d"]:
            assert pt["x"] == 0.0
            assert pt["y"] == 0.0
            assert pt["z"] == 0.0
