"""Tests that Modal functions are properly configured for deployment.

Validates app registration, GPU/CPU assignments, memory, volumes,
image configuration, and server connectivity for all 7 Modal functions.
"""

import pytest
import modal

from modal_app import (
    app,
    extract_claims,
    embed_claims,
    cluster_claims,
    rerank_evidence_batch,
    nli_verify_batch,
    compute_umap,
    score_clusters,
    VOLUME_MOUNT,
)


# -- App registration --------------------------------------------------------

EXPECTED_FUNCTIONS = [
    "extract_claims",
    "embed_claims",
    "cluster_claims",
    "rerank_evidence_batch",
    "nli_verify_batch",
    "compute_umap",
    "score_clusters",
]


class TestAppRegistration:
    def test_app_name(self):
        assert app.name == "truthlens-ml"

    def test_all_seven_functions_registered(self):
        registered = list(app.registered_functions.keys())
        for name in EXPECTED_FUNCTIONS:
            assert name in registered, f"{name} not registered"

    def test_exactly_seven_functions(self):
        registered = list(app.registered_functions.keys())
        assert len(registered) == 7


# -- GPU function specs ------------------------------------------------------

GPU_FUNCTIONS = [
    ("extract_claims", extract_claims, "A10G", 16384),
    ("embed_claims", embed_claims, "A10G", 16384),
    ("rerank_evidence_batch", rerank_evidence_batch, "A10G", 16384),
    ("nli_verify_batch", nli_verify_batch, "A10G", 24576),
]


class TestGpuFunctions:
    @pytest.mark.parametrize(
        "name,fn,expected_gpu,expected_memory", GPU_FUNCTIONS,
    )
    def test_gpu_assigned(self, name, fn, expected_gpu, expected_memory):
        spec = fn.spec
        assert str(spec.gpus) == expected_gpu, (
            f"{name} expected GPU {expected_gpu}, got {spec.gpus}"
        )

    @pytest.mark.parametrize(
        "name,fn,expected_gpu,expected_memory", GPU_FUNCTIONS,
    )
    def test_memory(self, name, fn, expected_gpu, expected_memory):
        spec = fn.spec
        assert spec.memory == expected_memory, (
            f"{name} expected memory {expected_memory}, got {spec.memory}"
        )

    @pytest.mark.parametrize(
        "name,fn,expected_gpu,expected_memory", GPU_FUNCTIONS,
    )
    def test_volume_mounted(self, name, fn, expected_gpu, expected_memory):
        spec = fn.spec
        assert VOLUME_MOUNT in spec.volumes, (
            f"{name} missing volume at {VOLUME_MOUNT}"
        )


# -- CPU function specs ------------------------------------------------------

CPU_FUNCTIONS = [
    ("cluster_claims", cluster_claims, 4, 8192),
    ("compute_umap", compute_umap, 8, 16384),
    ("score_clusters", score_clusters, 4, 8192),
]


class TestCpuFunctions:
    @pytest.mark.parametrize(
        "name,fn,expected_cpu,expected_memory", CPU_FUNCTIONS,
    )
    def test_no_gpu(self, name, fn, expected_cpu, expected_memory):
        spec = fn.spec
        assert spec.gpus is None, (
            f"{name} should not have a GPU, got {spec.gpus}"
        )

    @pytest.mark.parametrize(
        "name,fn,expected_cpu,expected_memory", CPU_FUNCTIONS,
    )
    def test_cpu_count(self, name, fn, expected_cpu, expected_memory):
        spec = fn.spec
        assert spec.cpu == expected_cpu, (
            f"{name} expected CPU {expected_cpu}, got {spec.cpu}"
        )

    @pytest.mark.parametrize(
        "name,fn,expected_cpu,expected_memory", CPU_FUNCTIONS,
    )
    def test_memory(self, name, fn, expected_cpu, expected_memory):
        spec = fn.spec
        assert spec.memory == expected_memory, (
            f"{name} expected memory {expected_memory}, got {spec.memory}"
        )


# -- Image configuration ----------------------------------------------------

class TestImageConfiguration:
    def test_gpu_functions_share_same_image(self):
        """All GPU functions should use the same gpu_image."""
        images = set()
        for _, fn, _, _ in GPU_FUNCTIONS:
            images.add(id(fn.spec.image))
        assert len(images) == 1, "GPU functions should share one image"

    def test_cpu_functions_share_same_image(self):
        """All CPU functions should use the same cpu_image."""
        images = set()
        for _, fn, _, _ in CPU_FUNCTIONS:
            images.add(id(fn.spec.image))
        assert len(images) == 1, "CPU functions should share one image"

    def test_gpu_and_cpu_images_are_different(self):
        """GPU and CPU images should be distinct."""
        gpu_image_id = id(GPU_FUNCTIONS[0][1].spec.image)
        cpu_image_id = id(CPU_FUNCTIONS[0][1].spec.image)
        assert gpu_image_id != cpu_image_id


# -- Function callability ---------------------------------------------------

class TestFunctionsCallable:
    """Verify .local() is available for all functions (Modal SDK contract)."""

    @pytest.mark.parametrize("name", EXPECTED_FUNCTIONS)
    def test_local_method_exists(self, name):
        fn = app.registered_functions[name]
        assert hasattr(fn, "local"), f"{name} missing .local() method"

    @pytest.mark.parametrize("name", EXPECTED_FUNCTIONS)
    def test_remote_method_exists(self, name):
        fn = app.registered_functions[name]
        assert hasattr(fn, "remote"), f"{name} missing .remote() method"


# -- Modal server connectivity -----------------------------------------------

@pytest.mark.modal_server
class TestModalServerConnectivity:
    """Tests that require a live connection to Modal's servers.

    Run selectively with: pytest -m modal_server -v
    """

    def test_client_authenticates(self):
        """Modal client can connect with current credentials."""
        client = modal.Client.from_env()
        client.hello()

    @pytest.mark.parametrize("name", EXPECTED_FUNCTIONS)
    def test_function_reference_creates(self, name):
        """Modal can create a lazy reference for each function."""
        fn_ref = modal.Function.from_name("truthlens-ml", name)
        assert fn_ref is not None

    @pytest.mark.parametrize("name", EXPECTED_FUNCTIONS)
    def test_build_def_is_valid(self, name):
        """Each function produces a non-empty build definition."""
        fn = app.registered_functions[name]
        build_def = fn.get_build_def()
        assert len(build_def) > 0

    def test_volume_reference_creates(self):
        """Volume reference creates without error."""
        vol = modal.Volume.from_name("truthlens-model-cache")
        assert vol is not None

    def test_source_modules_in_mounts(self):
        """GPU and CPU functions include local source mounts."""
        fn = app.registered_functions["extract_claims"]
        assert len(fn.spec.mounts) > 0, (
            "extract_claims should have at least one mount"
        )
