# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os

import pytest
import pytest_asyncio

import torch

from forge.actors.policy import Policy
from forge.data.sharding import Llama3vLLMSharding
from monarch.actor import proc_mesh
from torchstore import MultiProcessStore
from torchstore._state_dict_utils import push_state_dict
from transformers import AutoModelForCausalLM

from vllm.utils import get_open_port

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available",
)


def convert_state_dict(saved_sd):
    """
    Convert transformers state dict to vLLM format.

    Key conversions:
    1. Copy over directly mapped keys (down_proj, input_layernorm, etc.)
    2. Fuse QKV projections: combine q_proj, k_proj, v_proj into qkv_proj
    3. Fuse MLP projections: combine gate_proj and up_proj into gate_up_proj
    """
    load_sd = {}
    num_layers = 32  # For Llama-8B-3.1

    # Copy over directly mapped keys
    for k in saved_sd:
        if any(
            x in k
            for x in [
                "down_proj",
                "input_layernorm",
                "post_attention_layernorm",
                "o_proj",
                "norm.weight",
                "embed_tokens.weight",
                "lm_head.weight",
            ]
        ):
            load_sd[k] = saved_sd[k]

    # Fuse QKV and gate_up_proj
    for i in range(num_layers):
        prefix = f"model.layers.{i}."

        # QKV fusion
        q = saved_sd[prefix + "self_attn.q_proj.weight"]
        k = saved_sd[prefix + "self_attn.k_proj.weight"]
        v = saved_sd[prefix + "self_attn.v_proj.weight"]
        load_sd[prefix + "self_attn.qkv_proj.weight"] = torch.cat([q, k, v], dim=0)

        # MLP gate_up_proj fusion
        gate = saved_sd[prefix + "mlp.gate_proj.weight"]
        up = saved_sd[prefix + "mlp.up_proj.weight"]
        load_sd[prefix + "mlp.gate_up_proj.weight"] = torch.cat([gate, up], dim=0)

    return load_sd


async def save_state_dict(
    store: MultiProcessStore, state_dict: dict[str, torch.Tensor], key_prefix: str
):
    """
    Custom function to save state dict by iterating key by key
    """
    print(f"Saving {len(state_dict)} tensors")

    await push_state_dict(store, state_dict, key_prefix)

    print(f"Successfully saved {len(state_dict)} tensors")


def calculate_expected_shard(
    full_tensor: torch.Tensor,
    param_name: str,
    expected_shape: torch.Size,
    tensor_parallel_size: int,
    rank: int,
) -> torch.Tensor:
    """
    Calculate the expected shard of a full tensor for comparison with loaded tensor.
    This is mainly used for validation in tests.

    Args:
        full_tensor: The full tensor to shard
        param_name: Name of the parameter (used to determine sharding strategy)
        expected_shape: Expected shape of the sharded tensor
        tensor_parallel_size: Number of tensor parallel ranks
        rank: Current rank

    Returns:
        torch.Tensor: The expected sharded tensor for this rank
    """

    sharding = Llama3vLLMSharding(tensor_parallel_size, rank)
    # Get sharding strategy for this parameter
    shard_dim, is_sharded = sharding._get_tensor_parallel_sharding_strategy(param_name)

    if not is_sharded:
        # Parameter is replicated - should match exactly
        return full_tensor

    sharded_tensor = sharding._calculate_tensor_shard(
        full_tensor, shard_dim, tensor_parallel_size, rank
    )
    return sharded_tensor


def validate_loaded_tensors_equals_original(
    loaded_state_dict: dict[str, torch.Tensor],
    original_state_dict: dict[str, torch.Tensor],
    tensor_parallel_size: int,
    rank: int,
):
    """
    Shared validation function to verify that every tensor loaded by the policy
    equals the original tensor.

    For tensor parallel cases, instead of gathering sharded tensors, we shard
    the original tensor and compare it with the loaded shard.
    """
    print("Validating that loaded tensors equal original tensors * 2...")

    validation_errors = []

    for param_name, loaded_tensor in loaded_state_dict.items():
        if param_name in original_state_dict:
            expected_tensor = original_state_dict[param_name]

            if tensor_parallel_size > 1:
                # For tensor parallel case, shard the expected tensor to match the loaded shard
                expected_shard = calculate_expected_shard(
                    expected_tensor,
                    param_name,
                    loaded_tensor.shape,
                    tensor_parallel_size,
                    rank,
                )
                tensor_to_compare = expected_shard.cpu().float()
            else:
                # Single GPU case - compare directly
                tensor_to_compare = expected_tensor.cpu().float()

            if not torch.allclose(
                loaded_tensor.float(),
                tensor_to_compare,
                rtol=1e-5,
                atol=1e-8,
            ):
                validation_errors.append(
                    f"Loaded tensor {param_name} does not equal original "
                    f"(shapes: loaded={loaded_tensor.shape}, expected={tensor_to_compare.shape})"
                )
            else:
                print(f"Loaded tensor {param_name} correctly validated")

    if validation_errors:
        raise ValueError(f"Validation failed: {validation_errors}")

    print(
        f"Successfully validated that all {len(loaded_state_dict)} loaded tensors equal original"
    )


async def run_policy_integration(store, original_state_dict, num_gpus):
    """
    Common helper function to test Policy integration with different GPU configurations.

    Args:
        store: TorchStore instance
        original_state_dict: Original state dict for validation
        num_gpus: Number of GPUs to use (1 for single GPU, 2+ for tensor parallel)
        test_name: Name for test identification in validation messages
    """
    print(f"\n=== PHASE 2: Testing Policy Integration (GPUs: {num_gpus}) ===")

    state_dict_key = "llama3_8b_state_dict"

    # Set up environment variables for vLLM distributed initialization
    if num_gpus == 1:
        # Single GPU setup
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "12355")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        master_addr = os.environ.get("MASTER_ADDR", "localhost")
        master_port = os.environ.get("MASTER_PORT", "12355")
    else:
        # Multi-GPU setup
        master_addr = "localhost"
        master_port = str(get_open_port())
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = master_port
        print(f"Using MASTER_PORT: {master_port} for tensor parallel Policy")

    rank = int(os.environ.get("RANK", "0"))

    policy_mesh = await proc_mesh(
        gpus=num_gpus,
        env={
            "MASTER_ADDR": master_addr,
            "MASTER_PORT": master_port,
        },
    )

    # Spawn Policy as a proper Monarch actor
    policy = await policy_mesh.spawn(
        "policy",
        Policy,
        model="meta-llama/Meta-Llama-3.1-8B-Instruct",
        tensor_parallel_size=num_gpus,
        pipeline_parallel_size=1,
        enforce_eager=True,
        resources=num_gpus,
        torchstore=store,
        state_dict_key=state_dict_key,
    )

    await policy.setup.call()
    print("Setup completed successfully!")

    # Call update to load weights from torchstore
    print("Calling Policy.update() to load weights from torchstore...")
    await policy.update.call()
    print("Successfully called Policy.update() to load weights from torchstore!")

    # Get model info including state dict after update
    model_params = await policy.get_model_params.call()
    loaded_state_dict = (
        model_params._values[0] if hasattr(model_params, "_values") else model_params
    )
    print("Successfully got model state dict after update")

    # Validate that every tensor loaded by the policy equals the original tensor
    validate_loaded_tensors_equals_original(
        loaded_state_dict, original_state_dict, tensor_parallel_size=num_gpus, rank=rank
    )

    print("\nTest passed! State dict successfully loaded into Policy!")


@pytest_asyncio.fixture(scope="session")
async def llama3_torchstore_setup():
    """
    Pytest fixture to load Llama 3.1 8B-Instruct and write state dict to torchstore.
    Uses session scope so it's only called once when both tests are run.
    """
    print("=== PHASE 1: Writing Llama 3.1 8B-Instruct to TorchStore ===")

    # Use the class method create_store() which properly spawns the actors
    store = await MultiProcessStore.create_store()

    # Load from local directory instead of HuggingFace download
    model_path = "/tmp/Meta-Llama-3.1-8B-Instruct"

    # Load the model from local path - using device_map="auto" for efficient loading
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,  # Use half precision to save memory
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,  # Ensure we don't try to download
    )

    # Get the model's state dict
    original_state_dict = model.state_dict()
    print(f"Original state dict has {len(original_state_dict)} parameters")

    # Convert transformers state dict to vLLM format
    print("Converting transformers state dict to vLLM format...")
    converted_state_dict = convert_state_dict(original_state_dict)
    print(f"Converted state dict has {len(converted_state_dict)} parameters")

    state_dict_key = "llama3_8b_state_dict"
    # Write converted state dict to torchstore
    await save_state_dict(store, converted_state_dict, state_dict_key)
    print(
        f"Successfully wrote converted state dict to torchstore with key: {state_dict_key}"
    )

    return store, converted_state_dict


@pytest.mark.asyncio
@requires_cuda
async def test_llama3_policy_update_single(llama3_torchstore_setup):
    print("Starting Llama 3 8B torchstore test (single GPU)...")

    # Get store and original state dict from fixture
    store, original_state_dict = llama3_torchstore_setup

    # Phase 2: Test Policy integration with 1 GPU
    await run_policy_integration(store, original_state_dict, num_gpus=1)

    print(
        "\nSingle GPU test passed! Llama 3.1 8B-Instruct model successfully loaded into Policy via TorchStore!"
    )


@pytest.mark.asyncio
@requires_cuda
async def test_llama3_policy_update_tp(llama3_torchstore_setup):
    print("Starting tensor parallel test (load full state dict into sharded model)...")

    if torch.cuda.device_count() < 2:
        pytest.skip(
            f"Only {torch.cuda.device_count()} GPU(s) available, need 2+ for tensor parallel"
        )

    # Get store and original state dict from fixture
    store, original_state_dict = llama3_torchstore_setup

    # Phase 2: Test Policy integration with 2 GPUs
    await run_policy_integration(store, original_state_dict, num_gpus=2)

    print(
        "\nTensor parallel test passed! Full state dict successfully loaded into tensor parallel model!"
    )
