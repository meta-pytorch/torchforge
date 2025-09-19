# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Callable

import pytest
import pytest_asyncio

import torch
import torchstore as ts
from forge.actors.policy import EngineConfig, Policy, SamplingConfig

from forge.actors.trainer import RLTrainer
from forge.controller.service import ServiceConfig
from forge.data.sharding import VLLMSharding

from transformers import AutoModelForCausalLM

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available",
)
from forge.actors.trainer import _qwen3_hf_to_vllm
from huggingface_hub import snapshot_download

logger: logging.Logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# Run tests: pytest tests/integration_tests/test_policy_update.py::TestWeightSync::<test_name>


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
            original_tensor = original_state_dict[param_name]

            if tensor_parallel_size > 1:
                original_shard = calculate_expected_shard(
                    original_tensor,
                    param_name,
                    tensor_parallel_size,
                    rank,
                )
                tensor_to_compare = original_shard.cpu().float()
            else:
                tensor_to_compare = original_tensor.cpu().float()

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
                    f"dtype = {loaded_tensor.dtype} vs {original_tensor.dtype}\n"
                    f"shape= {loaded_tensor.shape} vs {original_tensor.shape}\n,"
                    f"values = {loaded_tensor} vs {original_tensor}"
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


def get_configs(
    worker_size: int, tp_size: int, model_name: str
) -> tuple[dict, ServiceConfig]:
    engine_config = EngineConfig(
        model=model_name,
        tensor_parallel_size=tp_size,
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


class TestWeightSync:
    """Tests for weight sync between trainer and policy. Currently hardcoded to Qwen3-1.7B."""

    model = "Qwen/Qwen3-1.7B"
    to_vllm_fn: Callable = _qwen3_hf_to_vllm
    num_layers = 28

    @pytest_asyncio.fixture
    def trainer_cfg(self):
        cached_dir = snapshot_download(repo_id=self.model)
        return {
            "model": {
                "name": "qwen3",
                "flavor": "1.7B",
            },
            "checkpoint": {
                "enable": True,
                "folder": "/tmp/saved_checkpoints",
                "initial_load_path": cached_dir,
                "initial_load_in_hf": True,
            },
        }

    @pytest_asyncio.fixture
    def trainer_cfg_tp(self):
        # NB: TP size is set to  2.
        cached_dir = snapshot_download(repo_id=self.model)
        return {
            "model": {
                "name": "qwen3",
                "flavor": "1.7B",
            },
            "parallelism": {"tensor_parallel_degree": 2},
            "checkpoint": {
                "enable": True,
                "folder": "/tmp/saved_checkpoints",
                "initial_load_path": cached_dir,
                "initial_load_in_hf": True,
            },
        }

    @pytest_asyncio.fixture
    def expected_sd(self):
        model = AutoModelForCausalLM.from_pretrained(
            self.model,
            dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        original_state_dict = model.state_dict()
        # Hack to access through class without passing in self param
        return self.__class__.to_vllm_fn(original_state_dict, self.num_layers)

    @pytest.mark.asyncio
    @requires_cuda
    async def test_policy_update_single(self, expected_sd, trainer_cfg):
        """
        1. Loads weights from HF model into in-memory state-dict (source of truth)
        2. Initializes RLTrainer, make the weights available in torchstore.
        3. Initializes Policy, and calls update_weights() to load weights from torchstore.
        4. Validate the policy weights against source of truth.
        """
        worker_size = 1
        # 1. Initialize TS
        await ts.initialize()
        # 2. Trainer push
        rl_trainer = await RLTrainer.options(
            procs_per_replica=worker_size, with_gpus=True, num_replicas=1
        ).as_service(**trainer_cfg)

        await rl_trainer.push_weights.choose(policy_version=0)
        # 3. Policy pull weights
        policy_config, service_config = get_configs(
            worker_size=worker_size, tp_size=worker_size, model_name=self.model
        )
        policy = await Policy.options(service_config=service_config).as_service(
            **policy_config
        )
        await policy.update_weights.call()
        # 4. Validate weights
        loaded_state_dict = await policy._get_model_params.choose()
        validate_loaded_tensors_equals_original(
            loaded_state_dict, expected_sd, tensor_parallel_size=1, rank=0
        )

    @pytest.mark.asyncio
    @requires_cuda
    async def test_policy_update_tp(self, expected_sd, trainer_cfg_tp):
        """
        1. Init RLTrainer over multiple workers with TP parallelism strategy.
        2. Push weights in to torchstore.
        3. Initializes Policy over multiple workers, and calls update_weights() to load weights from torchstore.
        4. Validate the policy weights against manually loaded origina HF weights.
        """
        # test configs/paralleism
        trainer_worker_size = 2
        policy_worker_size = 2
        tp_size = 2

        if torch.cuda.device_count() < 2:
            pytest.skip(
                f"Only {torch.cuda.device_count()} GPU(s) available, need 2+ for tensor parallel"
            )
        # 1. Initialize TS
        await ts.initialize()
        # 2. Trainer push
        rl_trainer = await RLTrainer.options(
            procs_per_replica=trainer_worker_size, with_gpus=True, num_replicas=1
        ).as_service(**trainer_cfg_tp)

        await rl_trainer.push_weights.call(policy_version=0)
        # 3. Policy pull weights
        policy_config, service_config = get_configs(
            worker_size=policy_worker_size, tp_size=tp_size, model_name=self.model
        )
        policy = await Policy.options(service_config=service_config).as_service(
            **policy_config
        )
        await policy.update_weights.call()

        # validate loaded shard of each worker againt manually calculated shard (expected shard).

        # 4. Validate weight shards. We compare vLLM loades shard content with
        #    Directly loaded HF shard content.
        sharded_state_dicts = await policy._get_model_params.call()
        validate_loaded_tensors_equals_original(
            sharded_state_dicts[0][0], expected_sd, tensor_parallel_size=tp_size, rank=0
        )
        validate_loaded_tensors_equals_original(
            sharded_state_dicts[0][1], expected_sd, tensor_parallel_size=tp_size, rank=1
        )
