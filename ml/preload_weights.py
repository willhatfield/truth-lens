"""Pre-download all model weights into the Modal shared volume.

Run with:
    modal run preload_weights.py

Downloads 4 models used by modal_app.py into the truthlens-model-cache volume
so that cold-start times are reduced for GPU functions.
"""

import modal
import os

# -- Modal App + shared volume (mirrors modal_app.py) ----------------------

app = modal.App("truthlens-ml-preload")

model_volume = modal.Volume.from_name(
    "truthlens-model-cache", create_if_missing=True,
)

VOLUME_MOUNT = "/models"
SHARED_ENV = {
    "HF_HOME": "/models/hf",
    "TRANSFORMERS_CACHE": "/models/hf",
    "SENTENCE_TRANSFORMERS_HOME": "/models/st",
}

# -- Image (same deps as gpu_image in modal_app.py) -------------------------

preload_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.5.1",
        "transformers==4.47.1",
        "sentence-transformers==3.3.1",
    )
    .env(SHARED_ENV)
)

# -- Model name constants (must match modal_app.py) -------------------------

EMBED_MODEL_NAME = "BAAI/bge-large-en-v1.5"
RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
NLI_MODEL_NAME = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli"
LLAMA_MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"

ALL_MODEL_NAMES = [
    EMBED_MODEL_NAME,
    RERANK_MODEL_NAME,
    NLI_MODEL_NAME,
    LLAMA_MODEL_NAME,
]

MAX_MODELS = 10  # bounded upper limit for loops


# -- Per-model download helpers ---------------------------------------------

def download_embed_model() -> str:
    """Download the BGE sentence-transformer embedding model."""
    from sentence_transformers import SentenceTransformer

    print(f"Downloading embedding model: {EMBED_MODEL_NAME}")
    SentenceTransformer(EMBED_MODEL_NAME)
    print(f"Done: {EMBED_MODEL_NAME}")
    return EMBED_MODEL_NAME


def download_rerank_model() -> str:
    """Download the cross-encoder reranking model."""
    from sentence_transformers import CrossEncoder

    print(f"Downloading reranker model: {RERANK_MODEL_NAME}")
    CrossEncoder(RERANK_MODEL_NAME)
    print(f"Done: {RERANK_MODEL_NAME}")
    return RERANK_MODEL_NAME


def download_nli_model() -> str:
    """Download the DeBERTa NLI model and tokenizer."""
    from transformers import AutoTokenizer
    from transformers import AutoModelForSequenceClassification

    print(f"Downloading NLI model: {NLI_MODEL_NAME}")
    AutoTokenizer.from_pretrained(NLI_MODEL_NAME, token=os.environ.get("HF_TOKEN"))
    AutoModelForSequenceClassification.from_pretrained(NLI_MODEL_NAME, token=os.environ.get("HF_TOKEN"))
    print(f"Done: {NLI_MODEL_NAME}")
    return NLI_MODEL_NAME


def download_llama_model() -> str:
    """Download the Llama causal-LM model and tokenizer."""
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print(f"Downloading Llama model: {LLAMA_MODEL_NAME}")
    AutoTokenizer.from_pretrained(LLAMA_MODEL_NAME, token=os.environ.get("HF_TOKEN"))
    AutoModelForCausalLM.from_pretrained(LLAMA_MODEL_NAME, token=os.environ.get("HF_TOKEN"))
    print(f"Done: {LLAMA_MODEL_NAME}")
    return LLAMA_MODEL_NAME


# -- Dispatcher table (avoids function pointers per CLAUDE.md) --------------

DOWNLOAD_FUNCTIONS = {
    EMBED_MODEL_NAME: download_embed_model,
    RERANK_MODEL_NAME: download_rerank_model,
    NLI_MODEL_NAME: download_nli_model,
    LLAMA_MODEL_NAME: download_llama_model,
}


# -- Main Modal function ----------------------------------------------------

@app.function(
    image=preload_image,
    volumes={VOLUME_MOUNT: model_volume},
    timeout=1800,
    memory=32768,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def preload_all_weights() -> dict:
    """Download all 4 model weights into the shared volume.

    Returns a dict with 'succeeded' and 'failed' lists.
    """
    succeeded: list = []
    failed: list = []

    for i in range(len(ALL_MODEL_NAMES)):
        if i >= MAX_MODELS:
            break
        name = ALL_MODEL_NAMES[i]
        download_fn = DOWNLOAD_FUNCTIONS.get(name)
        if download_fn is None:
            failed.append(name)
            print(f"WARNING: No download function for {name}")
            continue
        try:
            download_fn()
            succeeded.append(name)
        except Exception as exc:
            msg = f"WARNING: Failed to download {name}: {exc}"
            print(msg)
            failed.append(name)

    model_volume.commit()
    print(f"\nPreload complete: {len(succeeded)} succeeded, {len(failed)} failed")

    for j in range(len(succeeded)):
        print(f"  OK: {succeeded[j]}")

    for j in range(len(failed)):
        print(f"  FAILED: {failed[j]}")

    return {"succeeded": succeeded, "failed": failed}


@app.local_entrypoint()
def main():
    """Entry point for `modal run preload_weights.py`."""
    result = preload_all_weights.remote()
    print(f"\nResult: {result}")
