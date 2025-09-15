# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Dict, Tuple

import pytest
import pytest_asyncio

import torch
import torchstore as ts
from forge.actors.policy import EngineConfig, Policy, SamplingConfig

from forge.actors.trainer import RLTrainer
from forge.controller.service import ServiceConfig, spawn_service
from forge.data.sharding import VLLMSharding

from omegaconf import DictConfig, OmegaConf
from transformers import AutoModelForCausalLM

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available",
)


logger: logging.Logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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

    sharding = VLLMSharding(tensor_parallel_size, rank)
    shard_dim, is_sharded = sharding._get_tensor_parallel_sharding_strategy(param_name)

    if not is_sharded:
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
    for param_name, loaded_tensor in loaded_state_dict.items():
        if param_name in original_state_dict:
            expected_tensor = original_state_dict[param_name]

            if tensor_parallel_size > 1:
                expected_shard = calculate_expected_shard(
                    expected_tensor,
                    param_name,
                    loaded_tensor.shape,
                    tensor_parallel_size,
                    rank,
                )
                tensor_to_compare = expected_shard.cpu().float()
            else:
                tensor_to_compare = expected_tensor.cpu().float()

            # Training trainer emitted and loaded tensors are of type bfloat16,
            # where as a HF model loaded(expected) tensor has type float16.
            if not torch.allclose(
                loaded_tensor.float(),
                tensor_to_compare,
                rtol=1e-2,
                atol=1e-3,
            ):
                logger.warning(
                    f"Loaded tensor {param_name} does not equal original.\n"
                    f"dtype = {loaded_tensor.dtype} vs {expected_tensor.dtype}\n"
                    f"shape= {loaded_tensor.shape} vs {expected_tensor.shape}\n,"
                    f"values = {loaded_tensor.copy()} vs {expected_tensor.copy()}"
                )
                raise ValueError(
                    f"Loaded tensor {param_name} does not equal original "
                    f"(shapes: loaded={loaded_tensor.shape}, expected={tensor_to_compare.shape})"
                )
            else:
                print(f"Loaded tensor {param_name} correctly validated")

    print(
        f"Successfully validated that all {len(loaded_state_dict)} loaded tensors equal original"
    )


def get_configs(worker_size: int, model_name: str) -> Tuple[Dict, ServiceConfig]:

    engine_config = EngineConfig(
        model=model_name,
        tensor_parallel_size=worker_size,
        pipeline_parallel_size=1,
        enforce_eager=True,
    )

    sampling_config = SamplingConfig(
        n=3,
        guided_decoding=True,
    )

    policy_config = {
        "engine_config": engine_config,
        "sampling_config": sampling_config,
    }
    service_config = ServiceConfig(
        procs_per_replica=worker_size, num_replicas=1, with_gpus=True
    )

    return policy_config, service_config


@pytest_asyncio.fixture(scope="session")
async def setup_test():
    """
    Pytest fixture to load Llama 3.1 8B-Instruct. We use the loaded state dict
    as the SOT for validation. Uses session scope so it's only called once
    across UT.
    """
    model_path = "meta-llama/Meta-Llama-3.1-8B-Instruct"

    # Load the model from local path - using device_map="auto" for efficient loading
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,  # Use half precision to save memory
        device_map="auto",
        trust_remote_code=True,
    )

    original_state_dict = model.state_dict()
    print(f"Original state dict has {len(original_state_dict)} parameters")
    hf_state_dict = convert_state_dict(original_state_dict)
    print(f"Converted state dict has {len(hf_state_dict)} parameters")

    return hf_state_dict


async def run_rl_trainer(worker_size) -> None:
    """
    Spawn the RL trainer
    Args:
        worker_size: Number of workers/procs.
    """
    cfg: DictConfig = OmegaConf.load("apps/rl/llama3_8b.yaml")
    rl_trainer = await spawn_service(
        ServiceConfig(procs_per_replica=worker_size, with_gpus=True, num_replicas=1),
        RLTrainer,
        **cfg.trainer,
    )
    # Push the weights to torchstore
    await rl_trainer.push_weights.choose(policy_version=0)


async def run_policy_integration(worker_size) -> Dict[str, torch.Tensor]:
    """
    Launch the policy service.

    Args:
        store: TorchStore instance
        worker_size: Number of workers/procs (2+ for tensor parallel)
    """
    print(f"=== PHASE 2: Launching Policy Engine (Workers: {worker_size}) ===")

    policy_config, service_config = get_configs(
        worker_size=worker_size, model_name="meta-llama/Llama-3.1-8B-Instruct"
    )
    policy = await spawn_service(service_config, Policy, **policy_config)

    # Policy engine start with default version 0 that gets incremented.
    print("Calling Policy.update() to load weights from torchstore...")
    await policy.update_weights.call()
    print(
        "Successfully called Policy.update_weights() to load weights from torchstore!"
    )
    results = await policy._get_model_params.call()
    assert len(results) == 1
    print("Successfully got model state dict after update")
    return results[0]


@pytest.mark.asyncio
@requires_cuda
async def test_llama3_policy_update_single(setup_test):
    """
    1. Loads weights from HF model into in-memory state-dict (source of truth)
    2. Initializes RLTrainer, make the weights available in torchstore.
    3. Initializes Policy, and calls update_weights() to load weights from torchstore.
    4. Validate the policy weights against source of truth.
    """
    logger.info("Starting Llama 3 8B torchstore test (single GPU)...")
    await ts.initialize()
    expected_state_dict = setup_test
    await run_rl_trainer(worker_size=1)
    loaded_state_dict = await run_policy_integration(worker_size=1)

    # validating for single resource case.
    validate_loaded_tensors_equals_original(
        loaded_state_dict, expected_state_dict, tensor_parallel_size=0, rank=0
    )
    logger.info(
        "Single GPU test passed! Llama 3.1 8B-Instruct model successfully loaded into Policy via TorchStore!"
    )
