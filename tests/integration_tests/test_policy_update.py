# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import logging
from tempfile import TemporaryDirectory

import pytest

import torch
import torchstore as ts
from forge.actors.policy import Policy

from forge.actors.trainer import RLTrainer
from forge.cli.config import resolve_hf_hub_paths

from forge.controller.service.service import uuid
from monarch.actor import endpoint

from omegaconf import DictConfig, OmegaConf

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available",
)

from huggingface_hub import snapshot_download

logger: logging.Logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

"""
Run tests:

pytest -s tests/integration_tests/test_policy_update.py::TestWeightSync::test_sanity_check \
    --config tests/integration_tests/artifacts/qwen3_1_7b_tp.yaml --use_dcp=false

pytest -s tests/integration_tests/test_policy_update.py::TestWeightSync::test_sanity_check \
        --config apps/grpo/qwen3_8b.yaml
"""


class MockRLTrainer(RLTrainer):
    @endpoint
    async def zero_out_model_states(self):
        """This simply sets all model weights to zero."""
        for model_part in self.engine.model_parts:
            sd = model_part.state_dict()
            for k in sd.keys():
                if not torch.is_floating_point(sd[k]):
                    logger.info(
                        f"[MockRLTrainer] zero_out_model_states(): skipping non-float param {k}"
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
    """Tests for weight sync between trainer and policy."""

    def _load_config(self, config_path: str) -> DictConfig:
        cfg = None
        try:
            cfg = OmegaConf.load(config_path)
        except Exception as e:
            pytest.fail(f"Failed to load config file {config_path}: {e}")

        assert isinstance(cfg, DictConfig)

        cfg = resolve_hf_hub_paths(cfg)
        return cfg

    @pytest.mark.asyncio
    @requires_cuda
    async def test_sanity_check(self, request):
        """
        Sanity check for weight sync sharding between RLTrainer and Policy for a given model config.

        The check performs the following steps:
        - Initialize trainer and push weights v0 (original huggingface ckpt)
        - Step the trainer, setting all weights to zero and push weights v1
        - Load weights v0 and check the policy has all zero weights
        - Load weights v1 and check the policy has all the weights back

        """
        # Test setup
        config_path = request.config.getoption("--config", default=None)
        if not config_path:
            pytest.skip(
                "No config file provided. Use --config <path> to specify a YAML config file"
            )

        use_dcp_override = request.config.getoption("--use_dcp")
        cfg = self._load_config(config_path=config_path)

        trainer_proc_size = cfg.actors.trainer.procs
        policy_tp_size = cfg.policy.engine_config.tensor_parallel_size

        if policy_tp_size != cfg.services.policy.procs:
            pytest.fail(
                f"Expect policy proc = {cfg.services.policy.procs} to be equal to tensor parallel size = {policy_tp_size}"
            )

        model_card = cfg.model

        logger.info(f"Running sanity check with config: {config_path}")
        logger.info(f"Model name: {model_card}")
        logger.info(f"Trainer proc size: {trainer_proc_size}")
        logger.info(f"Policy tensor parallel size: {policy_tp_size}")

        logger.info("Downloading model checkpoint from HuggingFace Hub")
        cached_dir = snapshot_download(repo_id=model_card)
        logger.info("Finished downloading model checkpoint from HuggingFace Hub")

        await ts.initialize()
        services_policy_cfg = cfg.services.policy
        services_policy_cfg.num_replicas = 1

        trainer_cfg = cfg.trainer
        trainer_cfg.checkpoint = {
            "enable": True,
            "folder": "/tmp/saved_checkpoints",
            "initial_load_path": cached_dir,
            "initial_load_in_hf": True,
        }
        if use_dcp_override is not None:
            trainer_cfg["use_dcp"] = use_dcp_override
            logger.info(f"`trainer.use_dcp` is overriden to {use_dcp_override}")

        with TemporaryDirectory(dir="/dev/shm/") as tmpdir:
            trainer_cfg["dcp_path"] = tmpdir
            policy, rl_trainer = await asyncio.gather(
                *[
                    Policy.options(**services_policy_cfg).as_service(**cfg.policy),
                    MockRLTrainer.options(**cfg.actors.trainer).as_actor(**trainer_cfg),
                ]
            )

            # Main logic begins here
            v0 = uuid.uuid4().int
            v1 = v0 + 1

            await rl_trainer.push_weights.call(policy_version=v0)
            # Setting everything to zero
            await rl_trainer.zero_out_model_states.call()
            await rl_trainer.push_weights.call(policy_version=v1)
            await policy._test_save_model_params.fanout()

            # Sanity check that before update all the tests pass
            all_errs = await policy._test_validate_model_params.fanout(validate_fn)
            for errs in all_errs:
                for _, e in errs.items():
                    assert not e, f"Validation failed with exception: {e}"

            await policy.update_weights.fanout(policy_version=v1)
            all_errs = await policy._test_validate_model_params.fanout(
                validate_fn_all_zeros
            )
            for errs in all_errs:
                for _, e in errs.items():
                    assert not e, f"Validation failed with exception: {e}"

            # Reloading v0, getting back original weights
            await policy.update_weights.fanout(policy_version=v0)
            all_errs = await policy._test_validate_model_params.fanout(validate_fn)
            for errs in all_errs:
                for _, e in errs.items():
                    assert not e, f"Validation failed with exception: {e}"

            logger.info("✅ Weight sharding sanity check passed!")
            await ts.shutdown()
