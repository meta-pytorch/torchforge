#!/usr/bin/env python3
"""
Test script to:
1. Initialize Llama 3.1 8B-Instruct model from HuggingFace transformers
2. Write its state dict to torchstore
3. Initialize Policy with torchstore
4. Call update to load model weights into Policy
5. Verify the model works correctly
"""

import argparse
import asyncio
import os
import sys
import traceback

import numpy as np

import torch
import torch.distributed as dist

from forge.actors.policy import Policy
from monarch.actor import proc_mesh
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torchstore import MultiProcessStore
from torchstore._state_dict_utils import DELIM, push_state_dict
from transformers import AutoModelForCausalLM, AutoTokenizer

from vllm.utils import get_open_port

STATE_DICT_KEY = "llama3_8b_state_dict"


async def save_state_dict_multiplied_by_2(store, state_dict, key_prefix):
    """
    Custom function to save state dict by iterating key by key,
    multiplying every tensor by 2, and then saving it.
    """
    print(f"Saving {len(state_dict)} tensors with 2x multiplication...")

    for param_name, tensor in state_dict.items():
        # Multiply tensor by 2
        multiplied_tensor = tensor * 2.0

        # Save with the same key format as push_state_dict
        tensor_key = f"{key_prefix}{DELIM}{param_name}"
        await store.put(tensor_key, multiplied_tensor)

    print(f"Successfully saved {len(state_dict)} tensors multiplied by 2")


async def validate_tensors_multiplied_by_2(store, original_state_dict, key_prefix):
    """
    Custom function to validate that every tensor in store is multiplied by 2
    compared to the original state dict.
    """
    print(f"Validating {len(original_state_dict)} tensors are multiplied by 2...")

    validation_errors = []

    for param_name, original_tensor in original_state_dict.items():
        # Load tensor from store
        tensor_key = f"{key_prefix}{DELIM}{param_name}"
        stored_tensor = await store.get(tensor_key)

        # Check if stored tensor is original tensor * 2
        expected_tensor = original_tensor * 2.0

        # Use torch.allclose for floating point comparison
        if not torch.allclose(stored_tensor, expected_tensor, rtol=1e-5, atol=1e-8):
            validation_errors.append(
                f"Tensor {param_name} is not properly multiplied by 2"
            )

    if validation_errors:
        raise ValueError(f"Validation failed: {validation_errors}")

    print(
        f"Successfully validated all {len(original_state_dict)} tensors are multiplied by 2"
    )


def validate_loaded_tensors_equals_original_times_2(
    loaded_state_dict, original_state_dict, test_name="Policy"
):
    """
    Shared validation function to verify that every tensor loaded by the policy
    equals the original tensor multiplied by 2.
    """
    print("Validating that loaded tensors equal original tensors * 2...")

    validation_errors = []
    for param_name, loaded_tensor in loaded_state_dict.items():
        if param_name in original_state_dict:
            expected_tensor = original_state_dict[param_name] * 2.0
            if not torch.allclose(loaded_tensor, expected_tensor, rtol=1e-5, atol=1e-8):
                validation_errors.append(
                    f"Loaded tensor {param_name} does not equal original * 2"
                )

    if validation_errors:
        raise ValueError(f"{test_name} validation failed: {validation_errors}")

    print(
        f"Successfully validated that all {len(loaded_state_dict)} loaded tensors equal original * 2"
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
    num_layers = 32  # For Llama-8B-3.1, adjust if needed

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


async def llama3_torchstore_write():
    """
    First phase: Load Llama 3.1 8B-Instruct and write state dict to torchstore
    """
    print("=== PHASE 1: Writing Llama 3.1 8B-Instruct to TorchStore ===")

    # Use the class method create_store() which properly spawns the actors
    store = await MultiProcessStore.create_store()

    # Load from local directory instead of HuggingFace download
    model_path = "/tmp/Meta-Llama-3.1-8B-Instruct"

    try:
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

        # Write converted state dict to torchstore with 2x multiplication
        await save_state_dict_multiplied_by_2(
            store, converted_state_dict, STATE_DICT_KEY
        )
        print(
            f"Successfully wrote converted state dict (multiplied by 2) to torchstore with key: {STATE_DICT_KEY}"
        )

        return store, converted_state_dict

    except Exception as e:
        print(f"Error during model loading or processing: {e}")
        raise

    finally:
        # Clean up original model
        try:
            model_var = locals().get("model")
            if model_var is not None:
                del model_var
        except:
            pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def setup_distributed_fsdp():
    """Initialize distributed environment for FSDP with world_size=2"""
    if not dist.is_initialized():
        # Use environment variables that should already be set by multiprocessing
        rank = int(os.environ.get("RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", "2"))
        master_addr = os.environ.get("MASTER_ADDR", "localhost")
        master_port = os.environ.get("MASTER_PORT", "12356")

        try:
            # Initialize process group with timeout
            dist.init_process_group(
                backend="nccl" if torch.cuda.is_available() else "gloo",
                rank=rank,
                world_size=world_size,
                timeout=torch.distributed.timedelta(seconds=30),  # Add timeout
            )
        except Exception as e:
            raise


async def test_policy_integration_fsdp(
    store, state_dict_key, original_logits, tokenizer
):
    """
    FSDP Phase 2: Initialize Policy with tensor_parallel_size=2 and test update functionality
    """
    print(
        "\n=== FSDP PHASE 2: Testing Policy Integration with Tensor Parallel Size 2 ==="
    )

    master_addr = "localhost"
    master_port = str(get_open_port())  # Use dynamic port to avoid conflicts

    os.environ.setdefault("MASTER_ADDR", master_addr)
    os.environ["MASTER_PORT"] = master_port  # Always set a fresh port

    policy_mesh = await proc_mesh(
        gpus=2,  # 2 GPUs for tensor parallelism
        env={
            "MASTER_ADDR": master_addr,
            "MASTER_PORT": master_port,
        },
    )

    # Spawn Policy as a proper Monarch actor with tensor parallelism
    policy = await policy_mesh.spawn(
        "policy",
        Policy,
        model="meta-llama/Meta-Llama-3.1-8B-Instruct",
        tensor_parallel_size=2,  # Use tensor parallelism instead of FSDP for vLLM
        pipeline_parallel_size=1,
        enforce_eager=True,
        resources=2,  # 2 resources for 2 GPUs
        torchstore=store,
        state_dict_key=state_dict_key,
    )

    await policy.setup.call()

    # Now call update to load weights from torchstore
    print("Calling Policy.update() to load weights from torchstore...")
    await policy.update.call()


async def test_llama3_torchstore_fsdp():
    """
    Test loading a full state dict into a tensor parallel model
    """
    print("Starting tensor parallel test (load full state dict into sharded model)...")

    # Check if we have enough GPUs
    if not torch.cuda.is_available():
        print("No CUDA available for tensor parallel test")
        return False
    elif torch.cuda.device_count() < 2:
        print(
            f"Only {torch.cuda.device_count()} GPU(s) available, need 2+ for tensor parallel"
        )
        return False

    print(
        "Phase 1: Loading regular model and saving modified full state dict to torchstore..."
    )
    store, original_state_dict = await llama3_torchstore_write()

    master_addr = "localhost"
    master_port = str(get_open_port())

    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port

    print(f"Using MASTER_PORT: {master_port} for tensor parallel Policy")

    policy_mesh = await proc_mesh(
        gpus=2,  # 2 GPUs for tensor parallelism
        env={
            "MASTER_ADDR": master_addr,
            "MASTER_PORT": master_port,
        },
    )

    # Spawn Policy as a proper Monarch actor with tensor parallelism
    policy = await policy_mesh.spawn(
        "policy",
        Policy,
        model="meta-llama/Meta-Llama-3.1-8B-Instruct",
        tensor_parallel_size=2,  # Use tensor parallelism
        pipeline_parallel_size=1,
        enforce_eager=True,
        resources=2,  # 2 resources for 2 GPUs
        torchstore=store,
        state_dict_key=STATE_DICT_KEY,  # Use the key from the full model
    )

    await policy.setup.call()
    print("Tensor parallel Policy setup completed successfully!")

    # Get model state dict before update
    initial_state_dict = await policy.get_model_state_dict.call()

    # Now call update to load full weights from torchstore into sharded model
    print(
        "Calling Policy.update() to load full state dict into tensor parallel model..."
    )
    print(
        "This should automatically shard the full tensors for tensor parallel loading..."
    )

    await policy.update.call()

    # Get model state dict after update
    loaded_state_dict = await policy.get_model_state_dict.call()

    # Validate that every tensor loaded by the policy equals the original tensor * 2
    validate_loaded_tensors_equals_original_times_2(
        loaded_state_dict, original_state_dict, "FSDP Policy"
    )

    print(
        "\nTensor parallel test passed! Full state dict successfully loaded into tensor parallel model!"
    )
    return True


async def test_llama3_torchstore():
    """
    Complete test: Write to torchstore, then test Policy integration
    """

    # Phase 1: Write model to torchstore
    store, original_state_dict = await llama3_torchstore_write()

    # Phase 2: Test Policy integration
    # Set up environment variables for vLLM distributed initialization
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "12355")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")

    policy_mesh = await proc_mesh(
        gpus=1,
        env={
            "MASTER_ADDR": os.environ.get("MASTER_ADDR", "localhost"),
            "MASTER_PORT": os.environ.get("MASTER_PORT", "12355"),
        },
    )

    # Spawn Policy as a proper Monarch actor
    policy = await policy_mesh.spawn(
        "policy",
        Policy,
        model="meta-llama/Meta-Llama-3.1-8B-Instruct",
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        enforce_eager=True,
        resources=1,
        torchstore=store,
        state_dict_key=STATE_DICT_KEY,
    )

    await policy.setup.call()

    # Get model state dict before update
    initial_state_dict = await policy.get_model_state_dict.call()

    await policy.update.call()

    # Get model state dict after update
    loaded_state_dict = await policy.get_model_state_dict.call()

    # Validate that every tensor loaded by the policy equals the original tensor * 2
    validate_loaded_tensors_equals_original_times_2(
        loaded_state_dict, original_state_dict, "Single GPU Policy"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test Llama 3 8B with TorchStore and Policy integration"
    )
    parser.add_argument(
        "--test",
        choices=["single", "fsdp", "both"],
        default="single",
        help="Which test to run: single (default), fsdp, or both",
    )
    args = parser.parse_args()

    async def run_tests():
        if args.test in ["single", "both"]:
            print("Starting Llama 3 8B torchstore test (single GPU)...")
            await test_llama3_torchstore()

        if args.test in ["fsdp", "both"]:
            print("Starting Llama 3 8B FSDP torchstore test (world_size=2)...")
            await test_llama3_torchstore_fsdp()

        print("\n All requested tests completed!")

    asyncio.run(run_tests())
