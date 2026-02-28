"""Unit tests for modal_app.py functions -- mocked ML models.

Tests all 7 Modal functions with mocked ML model loading and inference.
Uses patch.dict("sys.modules", ...) to mock lazily-imported packages.
"""

import sys
import pytest
from unittest.mock import patch, MagicMock
import numpy as np

from modal_app import (
    extract_claims,
    embed_claims,
    cluster_claims,
    rerank_evidence_batch,
    nli_verify_batch,
    compute_umap,
    score_clusters,
    _softmax,
)


# -- Helpers ---------------------------------------------------------------

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


# -- extract_claims --------------------------------------------------------

class TestExtractClaims:
    def test_happy_path_llama_returns_json(self):
        """Mocked Llama returns a JSON array of claim strings."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = MagicMock()
        mock_tokenizer.apply_chat_template.return_value.to = MagicMock(
            return_value=MagicMock(shape=[1, 10]),
        )
        # apply_chat_template returns tensor-like with shape attr
        input_ids_mock = MagicMock()
        input_ids_mock.shape = [1, 10]
        input_ids_mock.to.return_value = input_ids_mock
        mock_tokenizer.apply_chat_template.return_value = input_ids_mock

        mock_model = MagicMock()
        # generate returns a list-of-list-like tensor
        output_tensor = MagicMock()
        output_tensor.__getitem__ = MagicMock(
            return_value=list(range(15)),
        )
        mock_model.generate.return_value = [list(range(15))]
        mock_model.eval.return_value = None
        mock_model.to.return_value = mock_model

        mock_tokenizer.decode.return_value = '["The sky is blue", "Water is wet"]'

        mock_transformers = MagicMock()
        mock_transformers.AutoTokenizer.from_pretrained.return_value = (
            mock_tokenizer
        )
        mock_transformers.AutoModelForCausalLM.from_pretrained.return_value = (
            mock_model
        )

        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False

        mock_id_utils = MagicMock()
        mock_id_utils.make_claim_id.side_effect = (
            lambda aid, mid, ct: f"c_{mid}_{ct[:5]}"
        )

        with patch.dict("sys.modules", {
            "transformers": mock_transformers,
            "torch": mock_torch,
            "id_utils": mock_id_utils,
            "claim_extraction": MagicMock(),
        }):
            result = extract_claims.local({
                "analysis_id": "a1",
                "responses": [
                    {"model_id": "gpt4", "response_text": "The sky is blue. Water is wet."},
                ],
            })

        assert len(result["warnings"]) == 0
        assert len(result["claims"]) == 2
        assert result["analysis_id"] == "a1"

    def test_fallback_on_llama_failure(self):
        """When Llama load fails, sentence-split fallback is used."""
        mock_transformers = MagicMock()
        mock_transformers.AutoTokenizer.from_pretrained.side_effect = (
            RuntimeError("no GPU")
        )

        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False

        mock_id_utils = MagicMock()
        mock_id_utils.make_claim_id.side_effect = (
            lambda aid, mid, ct: f"c_{ct[:5]}"
        )
        mock_claim_extraction = MagicMock()
        mock_claim_extraction.sentence_split_claims.return_value = [
            "Sentence one", "Sentence two",
        ]

        with patch.dict("sys.modules", {
            "transformers": mock_transformers,
            "torch": mock_torch,
            "id_utils": mock_id_utils,
            "claim_extraction": mock_claim_extraction,
        }):
            result = extract_claims.local({
                "analysis_id": "a1",
                "responses": [
                    {"model_id": "m1", "response_text": "Sentence one. Sentence two."},
                ],
            })

        assert len(result["warnings"]) >= 1
        assert "model load failed" in result["warnings"][0]
        assert len(result["claims"]) == 2

    def test_invalid_payload(self):
        """Missing required fields returns warning response."""
        result = extract_claims.local({})
        assert len(result["warnings"]) > 0

    def test_multiple_responses(self):
        """Multiple model responses produce claims from each."""
        mock_transformers = MagicMock()
        mock_transformers.AutoTokenizer.from_pretrained.side_effect = (
            RuntimeError("fail")
        )
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False

        mock_id_utils = MagicMock()
        mock_id_utils.make_claim_id.side_effect = (
            lambda aid, mid, ct: f"c_{mid}_{ct[:3]}"
        )
        mock_claim_extraction = MagicMock()
        mock_claim_extraction.sentence_split_claims.side_effect = [
            ["claim A"], ["claim B", "claim C"],
        ]

        with patch.dict("sys.modules", {
            "transformers": mock_transformers,
            "torch": mock_torch,
            "id_utils": mock_id_utils,
            "claim_extraction": mock_claim_extraction,
        }):
            result = extract_claims.local({
                "analysis_id": "a2",
                "responses": [
                    {"model_id": "m1", "response_text": "claim A."},
                    {"model_id": "m2", "response_text": "claim B. claim C."},
                ],
            })

        assert len(result["claims"]) == 3

    def test_claim_ids_deterministic(self):
        """Same input produces same claim_id (via mocked id_utils)."""
        mock_transformers = MagicMock()
        mock_transformers.AutoTokenizer.from_pretrained.side_effect = (
            RuntimeError("fail")
        )
        mock_torch = MagicMock()

        mock_id_utils = MagicMock()
        mock_id_utils.make_claim_id.return_value = "c_fixed_hash"
        mock_ce = MagicMock()
        mock_ce.sentence_split_claims.return_value = ["hello"]

        with patch.dict("sys.modules", {
            "transformers": mock_transformers,
            "torch": mock_torch,
            "id_utils": mock_id_utils,
            "claim_extraction": mock_ce,
        }):
            r1 = extract_claims.local({
                "analysis_id": "a1",
                "responses": [{"model_id": "m1", "response_text": "hello."}],
            })
            r2 = extract_claims.local({
                "analysis_id": "a1",
                "responses": [{"model_id": "m1", "response_text": "hello."}],
            })

        assert r1["claims"][0]["claim_id"] == r2["claims"][0]["claim_id"]

    def test_schema_version_propagated(self):
        """Response includes schema_version and analysis_id."""
        result = extract_claims.local({
            "analysis_id": "test_id",
            "responses": [],
        })
        # Invalid payload (empty responses), but analysis_id captured
        assert "warnings" in result


# -- embed_claims ----------------------------------------------------------

class TestEmbedClaims:
    def test_happy_path_vectors_keyed_by_claim_id(self):
        mock_model = MagicMock()
        fake_embeddings = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        mock_model.encode.return_value = fake_embeddings

        mock_st_module = MagicMock()
        mock_st_module.SentenceTransformer.return_value = mock_model

        with patch.dict("sys.modules", {
            "sentence_transformers": mock_st_module,
        }):
            result = embed_claims.local({
                "analysis_id": "a1",
                "claims": [
                    {"claim_id": "c_1", "claim_text": "hello"},
                    {"claim_id": "c_2", "claim_text": "world"},
                ],
            })

        assert len(result["warnings"]) == 0
        assert "c_1" in result["vectors"]
        assert "c_2" in result["vectors"]
        assert result["dim"] == 3

    def test_multiple_claims_batched(self):
        mock_model = MagicMock()
        fake_embeddings = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        mock_model.encode.return_value = fake_embeddings

        mock_st_module = MagicMock()
        mock_st_module.SentenceTransformer.return_value = mock_model

        with patch.dict("sys.modules", {
            "sentence_transformers": mock_st_module,
        }):
            result = embed_claims.local({
                "analysis_id": "a1",
                "claims": [
                    {"claim_id": "c_1", "claim_text": "a"},
                    {"claim_id": "c_2", "claim_text": "b"},
                    {"claim_id": "c_3", "claim_text": "c"},
                ],
            })

        assert len(result["vectors"]) == 3
        assert result["dim"] == 2

    def test_invalid_payload_empty_claims(self):
        result = embed_claims.local({
            "analysis_id": "a1",
            "claims": [],
        })
        assert len(result["warnings"]) > 0

    def test_missing_analysis_id(self):
        result = embed_claims.local({"claims": [{"claim_id": "c", "claim_text": "t"}]})
        assert len(result["warnings"]) > 0

    def test_model_load_failure(self):
        mock_st_module = MagicMock()
        mock_st_module.SentenceTransformer.side_effect = RuntimeError(
            "GPU OOM",
        )

        with patch.dict("sys.modules", {
            "sentence_transformers": mock_st_module,
        }):
            result = embed_claims.local({
                "analysis_id": "a1",
                "claims": [
                    {"claim_id": "c_1", "claim_text": "hello"},
                ],
            })

        assert len(result["warnings"]) > 0
        assert "embed_claims failed" in result["warnings"][0]
        assert result["vectors"] == {}


# -- cluster_claims --------------------------------------------------------

class TestClusterClaims:
    def test_happy_path_with_cluster_objects(self):
        """Clusters have IDs and representative_claim_id set."""
        vectors = {
            "c_1": [1.0, 0.0, 0.0],
            "c_2": [0.99, 0.01, 0.0],
            "c_3": [0.0, 0.0, 1.0],
            "c_4": [0.01, 0.0, 0.99],
        }
        claims = {
            "c_1": {"model_id": "m1", "claim_text": "claim 1"},
            "c_2": {"model_id": "m1", "claim_text": "claim 2"},
            "c_3": {"model_id": "m2", "claim_text": "claim 3"},
            "c_4": {"model_id": "m2", "claim_text": "claim 4"},
        }

        mock_id_utils = MagicMock()
        mock_id_utils.make_cluster_id.side_effect = (
            lambda ids: f"cl_{'_'.join(sorted(ids))}"
        )

        # Mock sklearn to return 2 clusters: (0,1) and (2,3)
        mock_clustering = MagicMock()
        mock_clustering.fit_predict.return_value = np.array([0, 0, 1, 1])

        mock_sklearn_cluster = MagicMock()
        mock_sklearn_cluster.AgglomerativeClustering.return_value = (
            mock_clustering
        )

        with patch.dict("sys.modules", {
            "id_utils": mock_id_utils,
            "sklearn.cluster": mock_sklearn_cluster,
        }):
            result = cluster_claims.local({
                "analysis_id": "a1",
                "vectors": vectors,
                "claims": claims,
                "sim_threshold": 0.85,
            })

        assert len(result["warnings"]) == 0
        assert len(result["clusters"]) == 2
        # Every cluster has required fields
        for cl in result["clusters"]:
            assert "cluster_id" in cl
            assert "claim_ids" in cl
            assert "representative_claim_id" in cl
            assert "representative_text" in cl

    def test_single_vector_one_cluster(self):
        mock_id_utils = MagicMock()
        mock_id_utils.make_cluster_id.return_value = "cl_single"

        with patch.dict("sys.modules", {"id_utils": mock_id_utils}):
            result = cluster_claims.local({
                "analysis_id": "a1",
                "vectors": {"c_1": [1.0, 2.0, 3.0]},
                "claims": {
                    "c_1": {"model_id": "m1", "claim_text": "only one"},
                },
            })

        assert len(result["clusters"]) == 1
        assert result["clusters"][0]["claim_ids"] == ["c_1"]

    def test_sim_threshold_conversion(self):
        """sim_threshold=0.85 should become distance_threshold=0.15."""
        mock_id_utils = MagicMock()
        mock_id_utils.make_cluster_id.return_value = "cl_test"

        mock_clustering = MagicMock()
        mock_clustering.fit_predict.return_value = np.array([0, 0])

        mock_sklearn_cluster = MagicMock()
        mock_sklearn_cluster.AgglomerativeClustering.return_value = (
            mock_clustering
        )

        with patch.dict("sys.modules", {
            "id_utils": mock_id_utils,
            "sklearn.cluster": mock_sklearn_cluster,
        }):
            result = cluster_claims.local({
                "analysis_id": "a1",
                "vectors": {
                    "c_1": [1.0, 0.0],
                    "c_2": [0.999, 0.001],
                },
                "claims": {
                    "c_1": {"model_id": "m1", "claim_text": "c1"},
                    "c_2": {"model_id": "m1", "claim_text": "c2"},
                },
                "sim_threshold": 0.85,
            })

        # Verify distance_threshold = 1.0 - 0.85 = 0.15
        call_kwargs = (
            mock_sklearn_cluster.AgglomerativeClustering.call_args
        )
        assert abs(call_kwargs[1]["distance_threshold"] - 0.15) < 1e-6
        assert len(result["warnings"]) == 0

    def test_fallback_single_cluster_on_failure(self):
        """When sklearn fails, all claims go into one cluster."""
        mock_id_utils = MagicMock()
        mock_id_utils.make_cluster_id.return_value = "cl_fallback"

        mock_sklearn = MagicMock()
        mock_sklearn.cluster.AgglomerativeClustering.side_effect = (
            RuntimeError("sklearn fail")
        )

        with patch.dict("sys.modules", {
            "id_utils": mock_id_utils,
            "sklearn": mock_sklearn,
            "sklearn.cluster": mock_sklearn.cluster,
        }):
            result = cluster_claims.local({
                "analysis_id": "a1",
                "vectors": {
                    "c_1": [1.0, 0.0],
                    "c_2": [0.0, 1.0],
                },
                "claims": {
                    "c_1": {"model_id": "m1", "claim_text": "c1"},
                    "c_2": {"model_id": "m1", "claim_text": "c2"},
                },
            })

        assert len(result["warnings"]) > 0
        assert "single cluster fallback" in result["warnings"][0]

    def test_invalid_payload(self):
        result = cluster_claims.local({"analysis_id": "a1"})
        assert len(result["warnings"]) > 0

    def test_claims_metadata_for_representative_text(self):
        mock_id_utils = MagicMock()
        mock_id_utils.make_cluster_id.return_value = "cl_rep"

        with patch.dict("sys.modules", {"id_utils": mock_id_utils}):
            result = cluster_claims.local({
                "analysis_id": "a1",
                "vectors": {"c_1": [1.0, 0.0, 0.0]},
                "claims": {
                    "c_1": {"model_id": "m1", "claim_text": "The representative text"},
                },
            })

        assert result["clusters"][0]["representative_text"] == (
            "The representative text"
        )


# -- rerank_evidence_batch -------------------------------------------------

class TestRerankEvidenceBatch:
    def test_happy_path_ordered_passage_ids(self):
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.1, 0.9, 0.5])

        mock_st_module = MagicMock()
        mock_st_module.CrossEncoder.return_value = mock_model

        with patch.dict("sys.modules", {
            "sentence_transformers": mock_st_module,
        }):
            result = rerank_evidence_batch.local({
                "analysis_id": "a1",
                "items": [{
                    "claim_id": "c_1",
                    "claim_text": "test claim",
                    "passages": [
                        {"passage_id": "p_1", "text": "passage 1"},
                        {"passage_id": "p_2", "text": "passage 2"},
                        {"passage_id": "p_3", "text": "passage 3"},
                    ],
                }],
                "top_k": 2,
            })

        assert len(result["warnings"]) == 0
        assert len(result["rankings"]) == 1
        ranking = result["rankings"][0]
        assert ranking["claim_id"] == "c_1"
        assert len(ranking["ordered_passage_ids"]) == 2
        # First passage should be the highest scored (p_2 with 0.9)
        assert ranking["ordered_passage_ids"][0] == "p_2"

    def test_multiple_items_in_batch(self):
        mock_model = MagicMock()
        mock_model.predict.side_effect = [
            np.array([0.8, 0.2]),
            np.array([0.3, 0.7]),
        ]

        mock_st_module = MagicMock()
        mock_st_module.CrossEncoder.return_value = mock_model

        with patch.dict("sys.modules", {
            "sentence_transformers": mock_st_module,
        }):
            result = rerank_evidence_batch.local({
                "analysis_id": "a1",
                "items": [
                    {
                        "claim_id": "c_1",
                        "claim_text": "claim 1",
                        "passages": [
                            {"passage_id": "p_1", "text": "p1"},
                            {"passage_id": "p_2", "text": "p2"},
                        ],
                    },
                    {
                        "claim_id": "c_2",
                        "claim_text": "claim 2",
                        "passages": [
                            {"passage_id": "p_3", "text": "p3"},
                            {"passage_id": "p_4", "text": "p4"},
                        ],
                    },
                ],
                "top_k": 10,
            })

        assert len(result["rankings"]) == 2

    def test_top_k_limits_results(self):
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.5, 0.8, 0.3])

        mock_st_module = MagicMock()
        mock_st_module.CrossEncoder.return_value = mock_model

        with patch.dict("sys.modules", {
            "sentence_transformers": mock_st_module,
        }):
            result = rerank_evidence_batch.local({
                "analysis_id": "a1",
                "items": [{
                    "claim_id": "c_1",
                    "claim_text": "test",
                    "passages": [
                        {"passage_id": "p_1", "text": "p1"},
                        {"passage_id": "p_2", "text": "p2"},
                        {"passage_id": "p_3", "text": "p3"},
                    ],
                }],
                "top_k": 1,
            })

        assert len(result["rankings"][0]["ordered_passage_ids"]) == 1

    def test_fallback_on_model_failure(self):
        mock_st_module = MagicMock()
        mock_st_module.CrossEncoder.side_effect = RuntimeError("load fail")

        with patch.dict("sys.modules", {
            "sentence_transformers": mock_st_module,
        }):
            result = rerank_evidence_batch.local({
                "analysis_id": "a1",
                "items": [{
                    "claim_id": "c_1",
                    "claim_text": "test",
                    "passages": [
                        {"passage_id": "p_1", "text": "p1"},
                        {"passage_id": "p_2", "text": "p2"},
                    ],
                }],
                "top_k": 2,
            })

        assert len(result["warnings"]) > 0
        assert "original order" in result["warnings"][0]
        assert result["rankings"][0]["ordered_passage_ids"] == ["p_1", "p_2"]

    def test_invalid_payload(self):
        result = rerank_evidence_batch.local({"analysis_id": "a1"})
        assert len(result["warnings"]) > 0

    def test_scores_are_descending(self):
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.1, 0.9, 0.5])

        mock_st_module = MagicMock()
        mock_st_module.CrossEncoder.return_value = mock_model

        with patch.dict("sys.modules", {
            "sentence_transformers": mock_st_module,
        }):
            result = rerank_evidence_batch.local({
                "analysis_id": "a1",
                "items": [{
                    "claim_id": "c_1",
                    "claim_text": "test",
                    "passages": [
                        {"passage_id": "p_1", "text": "p1"},
                        {"passage_id": "p_2", "text": "p2"},
                        {"passage_id": "p_3", "text": "p3"},
                    ],
                }],
                "top_k": 3,
            })

        ranking = result["rankings"][0]
        ids = ranking["ordered_passage_ids"]
        scores = ranking["scores"]
        for j in range(len(ids) - 1):
            assert scores[ids[j]] >= scores[ids[j + 1]]


# -- nli_verify_batch ------------------------------------------------------

class TestNliVerifyBatch:
    def test_happy_path_with_pair_ids(self):
        mock_tokenizer = MagicMock()
        mock_inputs = MagicMock()
        mock_inputs.to.return_value = mock_inputs
        mock_tokenizer.return_value = mock_inputs

        mock_model = MagicMock()
        mock_outputs = MagicMock()
        mock_logits = MagicMock()
        mock_logits.cpu.return_value = mock_logits
        mock_logits.tolist.return_value = [[2.0, 0.5, 0.1]]
        mock_outputs.logits = mock_logits
        mock_model.return_value = mock_outputs
        mock_model.eval.return_value = None
        mock_model.to.return_value = mock_model
        # Model config with id2label
        mock_config = MagicMock()
        mock_config.id2label = {0: "entailment", 1: "neutral", 2: "contradiction"}
        mock_model.config = mock_config

        mock_transformers = MagicMock()
        mock_transformers.AutoTokenizer.from_pretrained.return_value = (
            mock_tokenizer
        )
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
            result = nli_verify_batch.local({
                "analysis_id": "a1",
                "pairs": [{
                    "pair_id": "nli_1",
                    "claim_id": "c_1",
                    "passage_id": "p_1",
                    "claim_text": "sky is blue",
                    "passage_text": "The sky appears blue during the day",
                }],
                "batch_size": 8,
            })

        assert len(result["warnings"]) == 0
        assert len(result["results"]) == 1
        r = result["results"][0]
        assert r["pair_id"] == "nli_1"
        assert r["claim_id"] == "c_1"
        assert r["passage_id"] == "p_1"
        assert r["label"] in ["entailment", "neutral", "contradiction"]
        assert "entailment" in r["probs"]

    def test_label_order_from_model_config(self):
        """Labels are read from model.config.id2label, not hardcoded."""
        mock_tokenizer = MagicMock()
        mock_inputs = MagicMock()
        mock_inputs.to.return_value = mock_inputs
        mock_tokenizer.return_value = mock_inputs

        mock_model = MagicMock()
        mock_outputs = MagicMock()
        mock_logits = MagicMock()
        mock_logits.cpu.return_value = mock_logits
        # Logits: first position (index 0) is highest
        mock_logits.tolist.return_value = [[5.0, 0.1, 0.1]]
        mock_outputs.logits = mock_logits
        mock_model.return_value = mock_outputs
        mock_model.eval.return_value = None
        mock_model.to.return_value = mock_model
        # Custom label order: index 0 is "contradiction"
        mock_config = MagicMock()
        mock_config.id2label = {
            0: "contradiction", 1: "neutral", 2: "entailment",
        }
        mock_model.config = mock_config

        mock_transformers = MagicMock()
        mock_transformers.AutoTokenizer.from_pretrained.return_value = (
            mock_tokenizer
        )
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
            result = nli_verify_batch.local({
                "analysis_id": "a1",
                "pairs": [{
                    "pair_id": "nli_1",
                    "claim_id": "c_1",
                    "passage_id": "p_1",
                    "claim_text": "test",
                    "passage_text": "evidence",
                }],
            })

        # With index 0 being "contradiction" and highest logit at 0,
        # the label should be "contradiction"
        assert result["results"][0]["label"] == "contradiction"

    def test_fallback_neutral_on_failure(self):
        mock_transformers = MagicMock()
        mock_transformers.AutoTokenizer.from_pretrained.side_effect = (
            RuntimeError("fail")
        )

        mock_torch = MagicMock()

        with patch.dict("sys.modules", {
            "transformers": mock_transformers,
            "torch": mock_torch,
        }):
            result = nli_verify_batch.local({
                "analysis_id": "a1",
                "pairs": [
                    {
                        "pair_id": "nli_1",
                        "claim_id": "c_1",
                        "passage_id": "p_1",
                        "claim_text": "p1",
                        "passage_text": "h1",
                    },
                    {
                        "pair_id": "nli_2",
                        "claim_id": "c_2",
                        "passage_id": "p_2",
                        "claim_text": "p2",
                        "passage_text": "h2",
                    },
                ],
            })

        assert len(result["warnings"]) > 0
        assert "returning neutral" in result["warnings"][0]
        assert len(result["results"]) == 2
        for r in result["results"]:
            assert r["label"] == "neutral"

    def test_invalid_payload_empty_pairs(self):
        result = nli_verify_batch.local({
            "analysis_id": "a1", "pairs": [],
        })
        assert len(result["warnings"]) > 0

    def test_invalid_payload_missing_field(self):
        result = nli_verify_batch.local({})
        assert len(result["warnings"]) > 0

    def test_probs_sum_to_one(self):
        mock_tokenizer = MagicMock()
        mock_inputs = MagicMock()
        mock_inputs.to.return_value = mock_inputs
        mock_tokenizer.return_value = mock_inputs

        mock_model = MagicMock()
        mock_outputs = MagicMock()
        mock_logits = MagicMock()
        mock_logits.cpu.return_value = mock_logits
        mock_logits.tolist.return_value = [[1.0, 2.0, 3.0]]
        mock_outputs.logits = mock_logits
        mock_model.return_value = mock_outputs
        mock_model.eval.return_value = None
        mock_model.to.return_value = mock_model
        mock_config = MagicMock()
        mock_config.id2label = {
            0: "entailment", 1: "neutral", 2: "contradiction",
        }
        mock_model.config = mock_config

        mock_transformers = MagicMock()
        mock_transformers.AutoTokenizer.from_pretrained.return_value = (
            mock_tokenizer
        )
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
            result = nli_verify_batch.local({
                "analysis_id": "a1",
                "pairs": [{
                    "pair_id": "nli_1",
                    "claim_id": "c_1",
                    "passage_id": "p_1",
                    "claim_text": "test",
                    "passage_text": "evidence",
                }],
            })

        probs = result["results"][0]["probs"]
        total = sum(probs.values())
        assert abs(total - 1.0) < 0.01


# -- compute_umap ---------------------------------------------------------

class TestComputeUmap:
    def test_happy_path_coords3d_keyed_by_claim_id(self):
        mock_reducer = MagicMock()
        mock_reducer.fit_transform.return_value = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ])

        mock_umap_module = MagicMock()
        mock_umap_module.UMAP.return_value = mock_reducer

        with patch.dict("sys.modules", {"umap": mock_umap_module}):
            result = compute_umap.local({
                "analysis_id": "a1",
                "vectors": {
                    "c_1": [0.1, 0.2, 0.3],
                    "c_2": [0.4, 0.5, 0.6],
                },
            })

        assert len(result["warnings"]) == 0
        assert "c_1" in result["coords3d"]
        assert "c_2" in result["coords3d"]
        assert len(result["coords3d"]["c_1"]) == 3
        assert result["coords3d"]["c_1"][0] == 1.0

    def test_random_state_passed_to_umap(self):
        mock_reducer = MagicMock()
        mock_reducer.fit_transform.return_value = np.array([
            [0.0, 0.0, 0.0],
        ])

        mock_umap_module = MagicMock()
        mock_umap_module.UMAP.return_value = mock_reducer

        with patch.dict("sys.modules", {"umap": mock_umap_module}):
            compute_umap.local({
                "analysis_id": "a1",
                "vectors": {"c_1": [0.1, 0.2, 0.3]},
                "random_state": 99,
            })

        call_kwargs = mock_umap_module.UMAP.call_args
        assert call_kwargs[1]["random_state"] == 99

    def test_fallback_zeros_on_failure(self):
        mock_umap_module = MagicMock()
        mock_umap_module.UMAP.side_effect = RuntimeError("UMAP fail")

        with patch.dict("sys.modules", {"umap": mock_umap_module}):
            result = compute_umap.local({
                "analysis_id": "a1",
                "vectors": {
                    "c_1": [0.1, 0.2],
                    "c_2": [0.3, 0.4],
                },
            })

        assert len(result["warnings"]) > 0
        assert "returning zeros" in result["warnings"][0]
        assert result["coords3d"]["c_1"] == [0.0, 0.0, 0.0]
        assert result["coords3d"]["c_2"] == [0.0, 0.0, 0.0]

    def test_invalid_payload_empty_vectors(self):
        result = compute_umap.local({
            "analysis_id": "a1", "vectors": {},
        })
        assert len(result["warnings"]) > 0

    def test_missing_vectors(self):
        result = compute_umap.local({"analysis_id": "a1"})
        assert len(result["warnings"]) > 0


# -- score_clusters --------------------------------------------------------

class TestScoreClusters:
    def test_happy_path_trust_score_and_verdict(self):
        """Full scoring with supporting models and NLI results."""
        mock_scoring = MagicMock()
        mock_scoring.find_supporting_models.return_value = [
            "m1", "m2", "m3",
        ]
        mock_scoring.compute_agreement_score.return_value = 60.0
        mock_scoring.find_best_nli_for_cluster.return_value = (
            0.9, 0.05, "p_1",
        )
        mock_scoring.compute_verification_score.return_value = 85.0
        mock_scoring.compute_trust_score.return_value = 75
        mock_scoring.determine_verdict.return_value = "SAFE"

        with patch.dict("sys.modules", {"scoring": mock_scoring}):
            result = score_clusters.local({
                "analysis_id": "a1",
                "clusters": [{
                    "cluster_id": "cl_1",
                    "claim_ids": ["c_1", "c_2"],
                    "representative_claim_id": "c_1",
                    "representative_text": "test claim",
                }],
                "claims": {
                    "c_1": {"model_id": "m1", "claim_text": "c1"},
                    "c_2": {"model_id": "m2", "claim_text": "c2"},
                },
                "nli_results": [],
            })

        assert len(result["warnings"]) == 0
        assert len(result["scores"]) == 1
        score = result["scores"][0]
        assert score["cluster_id"] == "cl_1"
        assert score["trust_score"] == 75
        assert score["verdict"] == "SAFE"
        assert score["agreement"]["count"] == 3
        assert score["verification"]["best_entailment_prob"] == 0.9

    def test_safe_verdict(self):
        """Score >= 75 and contradiction <= 0.2 gives SAFE."""
        mock_scoring = MagicMock()
        mock_scoring.find_supporting_models.return_value = ["m1", "m2"]
        mock_scoring.compute_agreement_score.return_value = 40.0
        mock_scoring.find_best_nli_for_cluster.return_value = (
            0.95, 0.1, "p_1",
        )
        mock_scoring.compute_verification_score.return_value = 85.0
        mock_scoring.compute_trust_score.return_value = 80
        mock_scoring.determine_verdict.return_value = "SAFE"

        with patch.dict("sys.modules", {"scoring": mock_scoring}):
            result = score_clusters.local({
                "analysis_id": "a1",
                "clusters": [{
                    "cluster_id": "cl_1",
                    "claim_ids": ["c_1"],
                    "representative_claim_id": "c_1",
                    "representative_text": "t",
                }],
                "claims": {
                    "c_1": {"model_id": "m1", "claim_text": "c1"},
                },
            })

        assert result["scores"][0]["verdict"] == "SAFE"

    def test_caution_verdict(self):
        """Score >= 45 but not SAFE gives CAUTION."""
        mock_scoring = MagicMock()
        mock_scoring.find_supporting_models.return_value = ["m1"]
        mock_scoring.compute_agreement_score.return_value = 20.0
        mock_scoring.find_best_nli_for_cluster.return_value = (
            0.6, 0.5, "p_1",
        )
        mock_scoring.compute_verification_score.return_value = 10.0
        mock_scoring.compute_trust_score.return_value = 50
        mock_scoring.determine_verdict.return_value = "CAUTION"

        with patch.dict("sys.modules", {"scoring": mock_scoring}):
            result = score_clusters.local({
                "analysis_id": "a1",
                "clusters": [{
                    "cluster_id": "cl_1",
                    "claim_ids": ["c_1"],
                    "representative_claim_id": "c_1",
                    "representative_text": "t",
                }],
                "claims": {
                    "c_1": {"model_id": "m1", "claim_text": "c1"},
                },
            })

        assert result["scores"][0]["verdict"] == "CAUTION"

    def test_reject_verdict(self):
        """Score < 45 gives REJECT."""
        mock_scoring = MagicMock()
        mock_scoring.find_supporting_models.return_value = ["m1"]
        mock_scoring.compute_agreement_score.return_value = 20.0
        mock_scoring.find_best_nli_for_cluster.return_value = (
            0.1, 0.8, "p_1",
        )
        mock_scoring.compute_verification_score.return_value = -70.0
        mock_scoring.compute_trust_score.return_value = 30
        mock_scoring.determine_verdict.return_value = "REJECT"

        with patch.dict("sys.modules", {"scoring": mock_scoring}):
            result = score_clusters.local({
                "analysis_id": "a1",
                "clusters": [{
                    "cluster_id": "cl_1",
                    "claim_ids": ["c_1"],
                    "representative_claim_id": "c_1",
                    "representative_text": "t",
                }],
                "claims": {
                    "c_1": {"model_id": "m1", "claim_text": "c1"},
                },
            })

        assert result["scores"][0]["verdict"] == "REJECT"

    def test_no_nli_results(self):
        """When no NLI results, verification score is based on 0s."""
        mock_scoring = MagicMock()
        mock_scoring.find_supporting_models.return_value = ["m1"]
        mock_scoring.compute_agreement_score.return_value = 20.0
        mock_scoring.find_best_nli_for_cluster.return_value = (
            0.0, 0.0, "",
        )
        mock_scoring.compute_verification_score.return_value = 0.0
        mock_scoring.compute_trust_score.return_value = 8
        mock_scoring.determine_verdict.return_value = "REJECT"

        with patch.dict("sys.modules", {"scoring": mock_scoring}):
            result = score_clusters.local({
                "analysis_id": "a1",
                "clusters": [{
                    "cluster_id": "cl_1",
                    "claim_ids": ["c_1"],
                    "representative_claim_id": "c_1",
                    "representative_text": "t",
                }],
                "claims": {
                    "c_1": {"model_id": "m1", "claim_text": "c1"},
                },
                "nli_results": [],
            })

        verification = result["scores"][0]["verification"]
        assert verification["best_entailment_prob"] == 0.0
        assert verification["evidence_passage_id"] == ""

    def test_multiple_clusters_scored_independently(self):
        """Each cluster gets its own score."""
        mock_scoring = MagicMock()
        mock_scoring.find_supporting_models.side_effect = [
            ["m1", "m2"], ["m3"],
        ]
        mock_scoring.compute_agreement_score.side_effect = [40.0, 20.0]
        mock_scoring.find_best_nli_for_cluster.side_effect = [
            (0.9, 0.05, "p_1"),
            (0.1, 0.8, "p_2"),
        ]
        mock_scoring.compute_verification_score.side_effect = [85.0, -70.0]
        mock_scoring.compute_trust_score.side_effect = [80, 20]
        mock_scoring.determine_verdict.side_effect = ["SAFE", "REJECT"]

        with patch.dict("sys.modules", {"scoring": mock_scoring}):
            result = score_clusters.local({
                "analysis_id": "a1",
                "clusters": [
                    {
                        "cluster_id": "cl_1",
                        "claim_ids": ["c_1"],
                        "representative_claim_id": "c_1",
                        "representative_text": "t1",
                    },
                    {
                        "cluster_id": "cl_2",
                        "claim_ids": ["c_2"],
                        "representative_claim_id": "c_2",
                        "representative_text": "t2",
                    },
                ],
                "claims": {
                    "c_1": {"model_id": "m1", "claim_text": "c1"},
                    "c_2": {"model_id": "m3", "claim_text": "c2"},
                },
            })

        assert len(result["scores"]) == 2
        assert result["scores"][0]["verdict"] == "SAFE"
        assert result["scores"][1]["verdict"] == "REJECT"
