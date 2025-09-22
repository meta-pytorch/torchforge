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

from monarch.actor import endpoint
from torch.distributed.checkpoint._nested_dict import flatten_state_dict


requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available",
)

from huggingface_hub import snapshot_download

logger: logging.Logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# Run tests: pytest -s tests/integration_tests/test_policy_update.py::TestWeightSync::<test_name>


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
    service_config = ServiceConfig(procs=worker_size, num_replicas=1, with_gpus=True)
    return policy_config, service_config


class MockRLTrainer(RLTrainer):
    @endpoint
    async def mock_train_step(self):
        """Mock train step. This simply multiplies the model weights by 0.1."""
        sd = self.engine.checkpointer.states["model"].state_dict()
        sd, _ = flatten_state_dict(sd)
        for name, param in sd.items():
            if not torch.is_floating_point(param):
                continue
            param.copy_(sd[name] * 0.1)


class TestWeightSync:
    """Tests for weight sync between trainer and policy. Currently hardcoded to Qwen3-1.7B."""

    model = "Qwen/Qwen3-1.7B"

    @pytest_asyncio.fixture
    async def trainer_cfg(self):
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
    async def trainer_cfg_tp(self):
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

    @pytest.mark.asyncio
    @requires_cuda
    async def test_policy_update_single(self, trainer_cfg):
        """
        Test the weight synchronization process between RLTrainer and Policy.

        This test performs the following steps:
        1. Loads weights from a HuggingFace (HF) model into an in-memory state dictionary, serving as the source of truth.
        2. Initializes RLTrainer and applies a mock training step that multiplies all model weights by 0.1.
        3. Pushes the updated weights to torchstore.
        4. Initializes a Policy instance and calls update_weights() to load weights from torchstore.
        5. Validates that the policy's weights match the expected values (original weights multiplied by 0.1).
        """
        worker_size = 1
        # 1. Initialize TS
        await ts.initialize()
        # 2. Trainer push
        rl_trainer = await MockRLTrainer.options(
            procs=worker_size, with_gpus=True, num_replicas=1
        ).as_service(**trainer_cfg)

        # Mock train step multiplies everything by 0.1
        await rl_trainer.mock_train_step.call()

        await rl_trainer.push_weights.choose(policy_version=0)
        # 3. Policy pull weights
        policy_config, service_config = get_configs(
            worker_size=worker_size, tp_size=worker_size, model_name=self.model
        )
        policy = await Policy.options(service_config=service_config).as_service(
            **policy_config
        )
        await policy._test_save_model_params.call()
        await policy.update_weights.call(policy_version=0)

        # exceptions sometimes are not propogated in monarch, do it manually
        def validate_fn(prev_params, curr_model) -> Exception | None:
            try:
                for name, param in curr_model.named_parameters():
                    if not torch.is_floating_point(param):
                        continue
                    assert name in prev_params
                    assert torch.allclose(prev_params[name] * 0.1, param.cpu())
            except Exception as e:
                return e
            finally:
                return None

        all_errs = await policy._test_validate_model_params.call(validate_fn)
        for errs in all_errs:
            for _, e in errs.items():
                if e:
                    raise e

    @pytest.mark.asyncio
    @requires_cuda
    async def test_policy_update_tp(self, trainer_cfg_tp):
        """
        Test the weight synchronization process between RLTrainer and Policy.

        This test performs the following steps:
        - Initializes RLTrainer and applies a mock training step that multiplies all model weights by 0.1.
        - Pushes the updated weights to torchstore.
        - Initializes a Policy instance and calls update_weights() to load weights from torchstore.
        - Validates that the policy's weights match the expected values (original weights multiplied by 0.1).
        """
        # test configs/paralleism
        trainer_worker_size = 2
        policy_worker_size = 2
        tp_size = 2

        if torch.cuda.device_count() < 2:
            pytest.skip(
                f"Only {torch.cuda.device_count()} GPU(s) available, need 2+ for tensor parallel"
            )

        await ts.initialize()

        rl_trainer = await MockRLTrainer.options(
            procs=trainer_worker_size, with_gpus=True, num_replicas=1
        ).as_service(**trainer_cfg_tp)

        await rl_trainer.push_weights.call(policy_version=0)

        policy_config, service_config = get_configs(
            worker_size=policy_worker_size, tp_size=tp_size, model_name=self.model
        )
        policy = await Policy.options(service_config=service_config).as_service(
            **policy_config
        )

        # Mock train step multiplies everything by 0.1
        await rl_trainer.mock_train_step.call()

        await rl_trainer.push_weights.choose(policy_version=0)

        await policy._test_save_model_params.call()
        await policy.update_weights.call(policy_version=0)

        # exceptions sometimes are not propogated in monarch, do it manually
        def validate_fn(prev_params, curr_model, logger) -> Exception | None:
            verified = set()
            try:
                for name, param in curr_model.named_parameters():
                    if not torch.is_floating_point(param):
                        continue
                    assert name in prev_params
                    assert torch.allclose(prev_params[name] * 0.1, param.cpu())
                    verified.add(name)
            except Exception as e:
                return e
            finally:
                logger.info(
                    f"Successfully verified {len(verified)} parameters: {verified}"
                )
                return None

        all_errs = await policy._test_validate_model_params.call(validate_fn)
        for errs in all_errs:
            for _, e in errs.items():
                if e:
                    raise e
