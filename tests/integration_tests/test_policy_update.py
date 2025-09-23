# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import logging
from typing import Callable

import pytest
import pytest_asyncio

import torch
import torchstore as ts
from forge.actors.policy import EngineConfig, Policy, SamplingConfig

from forge.actors.trainer import RLTrainer
from forge.controller.service import ServiceConfig

from forge.controller.service.service import uuid

from monarch.actor import current_rank, endpoint
from torch.distributed.checkpoint._nested_dict import flatten_state_dict
from torch.distributed.tensor import DTensor, Replicate


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
        """Mock train step. This simply sets all model weights to zero."""
        for model_part in self.engine.model_parts:
            sd = model_part.state_dict()
            for k in sd.keys():
                if not torch.is_floating_point(sd[k]):
                    logger.info(
                        f"[MockRLTrainer] mock_train_step(): skipping non-float param {k}"
                    )
                    continue
                sd[k] *= 0.0


# exceptions sometimes are not propogated in monarch, do it manually
def validate_fn(prev_params, curr_model, logger) -> Exception | None:
    """Validate that current parameters are the same as prev_params."""
    verified = set()
    skipped = set()
    logger.info(
        f"Validating model params, all named_parameters() =  {curr_model.named_parameters()}"
    )
    errs = []
    for name, param in curr_model.named_parameters():
        if not torch.is_floating_point(param):
            logger.info(f"Skipping non-float param {name}")
            skipped.add(name)
            continue
        try:
            assert name in prev_params, f"Param {name} not found in prev_params"
            assert torch.allclose(
                prev_params[name], param.cpu(), atol=1e-3, rtol=1e-2
            ), (
                f"current param {name} does not match expected value; "
                f"previous param ({prev_params[name].size()})= {prev_params[name]}; "
                f"expected = {prev_params[name]} vs got = {param.cpu().size()} {param.cpu()}"
            )
            verified.add(name)
        except Exception as e:
            # logger.error(f"Validation failed with exception: {e}")
            errs.append((name, e))
    logger.info(f"Verified params = {verified}")
    logger.info(f"Skipped params = {skipped}")
    if errs:
        logger.error(
            f"Validation failed for the following params: {[e[0] for e in errs]}"
        )
        return AssertionError(f"Validation failed: {errs}")


# exceptions sometimes are not propogated in monarch, do it manually
def validate_fn_all_zeros(prev_params, curr_model, logger) -> Exception | None:
    """Validate all parameters are set to zero. prev_params is actually not used."""
    _ = prev_params
    verified = set()
    skipped = set()
    logger.info(
        f"Validating model params, all named_parameters() =  {curr_model.named_parameters()}"
    )
    errs = []
    for name, param in curr_model.named_parameters():
        if not torch.is_floating_point(param):
            logger.info(f"Skipping non-float param {name}")
            skipped.add(name)
            continue
        try:
            param = param.cpu()
            assert torch.allclose(
                torch.zeros_like(param), param, atol=1e-4, rtol=1e-3
            ), "param {name} is not zero."
            verified.add(name)
        except Exception as e:
            # logger.error(f"Validation failed with exception: {e}")
            errs.append((name, e))
    logger.info(f"Verified params = {verified}")
    logger.info(f"Skipped params = {skipped}")
    if errs:
        logger.error(
            f"Validation failed for the following params: {[e[0] for e in errs]}"
        )
        return AssertionError(f"Validation failed: {errs}")


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
        - Initialize trainer and push weights v0 (original huggingface ckpt)
        - Step the trainer, setting all weights to zero and push weights v1
        - Load weights v0 and check the policy has all zero weights
        - Load weights v1 and check the policy has all the weights back
        """
        trainer_worker_size = 1
        policy_worker_size = 1
        tp_size = 1

        await ts.initialize()

        policy_config, service_config = get_configs(
            worker_size=policy_worker_size, tp_size=tp_size, model_name=self.model
        )
        policy, rl_trainer = await asyncio.gather(
            *[
                Policy.options(service_config=service_config).as_service(
                    **policy_config
                ),
                MockRLTrainer.options(
                    procs=trainer_worker_size, with_gpus=True, num_replicas=1
                ).as_service(**trainer_cfg),
            ]
        )

        v0 = uuid.uuid4().int
        v1 = v0 + 1

        await rl_trainer.push_weights_hf_nonsharded.call(policy_version=v0)
        # Setting everything to zero
        await rl_trainer.mock_train_step.call()
        await rl_trainer.push_weights_hf_nonsharded.call(policy_version=v1)
        await policy._test_save_model_params.call()

        # Sanity check that before update all the tests pass
        all_errs = await policy._test_validate_model_params.call(validate_fn)
        for errs in all_errs:
            for _, e in errs.items():
                assert not e, f"Validation failed with exception: {e}"

        await policy.update_weights.call(policy_version=v1)
        all_errs = await policy._test_validate_model_params.call(validate_fn_all_zeros)
        for errs in all_errs:
            for _, e in errs.items():
                assert not e, f"Validation failed with exception: {e}"

        # Reloading v0, getting back original weights
        await policy.update_weights.call(policy_version=v0)
        all_errs = await policy._test_validate_model_params.call(validate_fn)
        for errs in all_errs:
            for _, e in errs.items():
                assert not e, f"Validation failed with exception: {e}"

        await ts.shutdown()

    @pytest.mark.asyncio
    @requires_cuda
    async def test_policy_update_tp(self, trainer_cfg_tp):
        """
        Test the weight synchronization process between RLTrainer and Policy.

        This test performs the following steps:
        - Initialize trainer and push weights v0 (original huggingface ckpt)
        - Step the trainer, setting all weights to zero and push weights v1
        - Load weights v0 and check the policy has all zero weights
        - Load weights v1 and check the policy has all the weights back
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

        policy_config, service_config = get_configs(
            worker_size=policy_worker_size, tp_size=tp_size, model_name=self.model
        )
        policy, rl_trainer = await asyncio.gather(
            *[
                Policy.options(service_config=service_config).as_service(
                    **policy_config
                ),
                MockRLTrainer.options(
                    procs=trainer_worker_size, with_gpus=True, num_replicas=1
                ).as_service(**trainer_cfg_tp),
            ]
        )

        v0 = uuid.uuid4().int
        v1 = v0 + 1

        await rl_trainer.push_weights_hf_nonsharded.call(policy_version=v0)
        # Setting everything to zero
        await rl_trainer.mock_train_step.call()
        await rl_trainer.push_weights_hf_nonsharded.call(policy_version=v1)
        await policy._test_save_model_params.call()

        # Sanity check that before update all the tests pass
        all_errs = await policy._test_validate_model_params.call(validate_fn)
        for errs in all_errs:
            for _, e in errs.items():
                assert not e, f"Validation failed with exception: {e}"

        await policy.update_weights.call(policy_version=v1)
        all_errs = await policy._test_validate_model_params.call(validate_fn_all_zeros)
        for errs in all_errs:
            for _, e in errs.items():
                assert not e, f"Validation failed with exception: {e}"
        # Reloading v0, getting back original weights
        await policy.update_weights.call(policy_version=v0)
        all_errs = await policy._test_validate_model_params.call(validate_fn)
        for errs in all_errs:
            for _, e in errs.items():
                assert not e, f"Validation failed with exception: {e}"

        await ts.shutdown()
