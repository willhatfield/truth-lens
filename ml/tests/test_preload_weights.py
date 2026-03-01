"""Unit tests for preload_weights.py -- mocked model downloads.

Verifies:
  - Script imports correctly
  - Model name constants match modal_app.py
  - Modal function is registered on the app
  - Download helpers call the correct libraries
  - preload_all_weights handles partial failures gracefully
"""

import pytest
from unittest.mock import patch, MagicMock

pytestmark = pytest.mark.filterwarnings(
    "ignore:.*executing locally.*:UserWarning"
)


# -- Import correctness ----------------------------------------------------

class TestImports:
    def test_module_imports_without_error(self):
        """preload_weights.py can be imported."""
        import preload_weights
        assert preload_weights is not None

    def test_app_attribute_exists(self):
        """Module exposes a Modal App object."""
        import preload_weights
        assert hasattr(preload_weights, "app")

    def test_preload_function_exists(self):
        """Module exposes the preload_all_weights function."""
        import preload_weights
        assert hasattr(preload_weights, "preload_all_weights")

    def test_all_model_names_list_exists(self):
        """Module exposes ALL_MODEL_NAMES list."""
        import preload_weights
        assert hasattr(preload_weights, "ALL_MODEL_NAMES")
        assert len(preload_weights.ALL_MODEL_NAMES) == 4

    def test_download_functions_dict_exists(self):
        """Module exposes DOWNLOAD_FUNCTIONS mapping."""
        import preload_weights
        assert hasattr(preload_weights, "DOWNLOAD_FUNCTIONS")
        assert len(preload_weights.DOWNLOAD_FUNCTIONS) == 4


# -- Model name constants match modal_app.py --------------------------------

class TestModelNameConstants:
    def test_embed_model_name_matches(self):
        import preload_weights
        import modal_app
        assert preload_weights.EMBED_MODEL_NAME == modal_app.EMBED_MODEL_NAME

    def test_rerank_model_name_matches(self):
        import preload_weights
        import modal_app
        assert preload_weights.RERANK_MODEL_NAME == modal_app.RERANK_MODEL_NAME

    def test_nli_model_name_matches(self):
        import preload_weights
        import modal_app
        assert preload_weights.NLI_MODEL_NAME == modal_app.NLI_MODEL_NAME

    def test_llama_model_name_matches(self):
        import preload_weights
        import modal_app
        assert preload_weights.LLAMA_MODEL_NAME == modal_app.LLAMA_MODEL_NAME

    def test_all_model_names_contains_all_four(self):
        import preload_weights
        import modal_app
        expected = [
            modal_app.EMBED_MODEL_NAME,
            modal_app.RERANK_MODEL_NAME,
            modal_app.NLI_MODEL_NAME,
            modal_app.LLAMA_MODEL_NAME,
        ]
        for name in expected:
            assert name in preload_weights.ALL_MODEL_NAMES


# -- Volume and env config match modal_app.py --------------------------------

class TestVolumeConfig:
    def test_volume_mount_matches(self):
        import preload_weights
        import modal_app
        assert preload_weights.VOLUME_MOUNT == modal_app.VOLUME_MOUNT

    def test_shared_env_matches(self):
        import preload_weights
        import modal_app
        assert preload_weights.SHARED_ENV == modal_app.SHARED_ENV


# -- Download helper functions ----------------------------------------------

class TestDownloadEmbedModel:
    def test_calls_sentence_transformer(self):
        mock_st = MagicMock()
        with patch.dict("sys.modules", {
            "sentence_transformers": mock_st,
        }):
            from preload_weights import download_embed_model
            result = download_embed_model()

        mock_st.SentenceTransformer.assert_called_once_with(
            "BAAI/bge-large-en-v1.5",
        )
        assert result == "BAAI/bge-large-en-v1.5"


class TestDownloadRerankModel:
    def test_calls_cross_encoder(self):
        mock_st = MagicMock()
        with patch.dict("sys.modules", {
            "sentence_transformers": mock_st,
        }):
            from preload_weights import download_rerank_model
            result = download_rerank_model()

        mock_st.CrossEncoder.assert_called_once_with(
            "cross-encoder/ms-marco-MiniLM-L-6-v2",
        )
        assert result == "cross-encoder/ms-marco-MiniLM-L-6-v2"


class TestDownloadNliModel:
    def test_calls_auto_model_for_sequence_classification(self):
        mock_transformers = MagicMock()
        with patch.dict("sys.modules", {
            "transformers": mock_transformers,
        }):
            from preload_weights import download_nli_model
            result = download_nli_model()

        nli_name = "cross-encoder/nli-deberta-v3-large"
        mock_transformers.AutoTokenizer.from_pretrained.assert_called_once_with(
            nli_name,
        )
        mock_transformers.AutoModelForSequenceClassification.from_pretrained.assert_called_once_with(
            nli_name,
        )
        assert result == nli_name


class TestDownloadLlamaModel:
    def test_calls_auto_model_for_causal_lm(self):
        mock_transformers = MagicMock()
        with patch.dict("sys.modules", {
            "transformers": mock_transformers,
        }):
            from preload_weights import download_llama_model
            result = download_llama_model()

        llama_name = "meta-llama/Llama-3.1-8B-Instruct"
        mock_transformers.AutoTokenizer.from_pretrained.assert_called_once_with(
            llama_name,
        )
        mock_transformers.AutoModelForCausalLM.from_pretrained.assert_called_once_with(
            llama_name,
        )
        assert result == llama_name


# -- preload_all_weights integration ----------------------------------------

class TestPreloadAllWeights:
    def test_all_succeed(self):
        """When all downloads succeed, succeeded list has 4 items."""
        import preload_weights

        mock_st = MagicMock()
        mock_transformers = MagicMock()

        with patch.dict("sys.modules", {
            "sentence_transformers": mock_st,
            "transformers": mock_transformers,
        }):
            with patch.object(
                preload_weights.model_volume, "commit",
            ):
                result = preload_weights.preload_all_weights.local()

        assert len(result["succeeded"]) == 4
        assert len(result["failed"]) == 0

    def test_partial_failure_continues(self):
        """When one model fails, others still download."""
        import preload_weights

        mock_st = MagicMock()
        # CrossEncoder (rerank) fails
        mock_st.CrossEncoder.side_effect = RuntimeError("download error")

        mock_transformers = MagicMock()

        with patch.dict("sys.modules", {
            "sentence_transformers": mock_st,
            "transformers": mock_transformers,
        }):
            with patch.object(
                preload_weights.model_volume, "commit",
            ):
                result = preload_weights.preload_all_weights.local()

        assert len(result["succeeded"]) == 3
        assert len(result["failed"]) == 1
        assert "cross-encoder/ms-marco-MiniLM-L-6-v2" in result["failed"]

    def test_all_fail_returns_empty_succeeded(self):
        """When all downloads fail, succeeded is empty."""
        import preload_weights

        mock_st = MagicMock()
        mock_st.SentenceTransformer.side_effect = RuntimeError("fail")
        mock_st.CrossEncoder.side_effect = RuntimeError("fail")

        mock_transformers = MagicMock()
        mock_transformers.AutoTokenizer.from_pretrained.side_effect = (
            RuntimeError("fail")
        )

        with patch.dict("sys.modules", {
            "sentence_transformers": mock_st,
            "transformers": mock_transformers,
        }):
            with patch.object(
                preload_weights.model_volume, "commit",
            ):
                result = preload_weights.preload_all_weights.local()

        assert len(result["succeeded"]) == 0
        assert len(result["failed"]) == 4

    def test_result_dict_has_expected_keys(self):
        """Return value always has 'succeeded' and 'failed' keys."""
        import preload_weights

        mock_st = MagicMock()
        mock_transformers = MagicMock()

        with patch.dict("sys.modules", {
            "sentence_transformers": mock_st,
            "transformers": mock_transformers,
        }):
            with patch.object(
                preload_weights.model_volume, "commit",
            ):
                result = preload_weights.preload_all_weights.local()

        assert "succeeded" in result
        assert "failed" in result

    def test_volume_commit_called(self):
        """Volume.commit() is called after downloads."""
        import preload_weights

        mock_st = MagicMock()
        mock_transformers = MagicMock()

        with patch.dict("sys.modules", {
            "sentence_transformers": mock_st,
            "transformers": mock_transformers,
        }):
            with patch.object(
                preload_weights.model_volume, "commit",
            ) as mock_commit:
                preload_weights.preload_all_weights.local()

        mock_commit.assert_called_once()
