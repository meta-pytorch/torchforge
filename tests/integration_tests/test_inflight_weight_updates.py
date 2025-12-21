# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Tests the in-flight weight update mechanism that applies updates at token
boundaries without waiting for in-progress requests to complete.

Run (uses fixture config):
    TORCHSTORE_RDMA_ENABLED=0 pytest tests/integration_tests/test_inflight_weight_updates.py -v -s \
        --config tests/integration_tests/fixtures/qwen3_0_6b_inflight.yaml
"""

import asyncio
import logging
import shutil
from pathlib import Path

import monarch
import pytest
import pytest_asyncio
import torch
import torchstore as ts

from forge.actors.generator import Generator
from forge.actors.trainer import TitanTrainer
from forge.controller.provisioner import init_provisioner
from forge.controller.service.service import uuid
from forge.types import LauncherConfig, ProvisionerConfig
from forge.util.config import resolve_hf_hub_paths
from huggingface_hub import snapshot_download
from monarch.actor import endpoint
from omegaconf import DictConfig, OmegaConf
from vllm.transformers_utils.tokenizer import get_tokenizer

# Workaround for monarch mesh shutdown exit code during teardown
monarch.actor.unhandled_fault_hook = lambda failure: None

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

requires_cuda_with_4_gpus = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 4,
    reason="CUDA not available or not enough GPUs",
)

TEST_DCP_DIR = "test_inflight_dcp_tmp"


def _format_prompt_with_chat_template(tokenizer, prompt: str) -> str:
    """Format a prompt using the model's chat template."""
    messages = [
        {"role": "user", "content": prompt},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def _load_config(config_path: str) -> DictConfig:
    """Load and validate config from YAML file."""
    cfg = None
    try:
        cfg = OmegaConf.load(config_path)
    except Exception as e:
        pytest.fail(f"Failed to load config file {config_path}: {e}")

    assert isinstance(cfg, DictConfig)
    cfg = resolve_hf_hub_paths(cfg)
    return cfg


class MockTitanTrainer(TitanTrainer):
    """Mock trainer that can modify weights for testing."""

    @endpoint
    async def zero_out_model_states(self):
        """Set all model weights to zero."""
        for model_part in self.engine.model_parts:
            sd = model_part.state_dict()
            for k in sd.keys():
                if not torch.is_floating_point(sd[k]):
                    continue
                sd[k] *= 0.0


@pytest_asyncio.fixture(scope="module")
async def setup_inflight_test(request):
    """
    Module-scoped setup fixture that creates:
    - A Generator with enable_in_flight_weight_updates=True
    - A MockTitanTrainer for pushing weights

    Using module scope to avoid re-initializing torchstore for each test.
    """
    config_path = request.config.getoption("--config", default=None)
    if not config_path:
        pytest.skip(
            "No config file provided. Use --config <path> to specify a YAML config file"
        )

    cfg = _load_config(config_path)

    model_card = cfg.model
    logger.info(f"Running in-flight weight update tests with model: {model_card}")

    # Download model
    logger.info("Downloading model checkpoint...")
    cached_dir = snapshot_download(repo_id=model_card)

    # Configure policy (Generator)
    # Use config value for num_replicas (set in YAML under services.policy.num_replicas)
    services_policy_cfg = OmegaConf.to_container(cfg.services.policy, resolve=True)

    # Configure trainer
    trainer_cfg = OmegaConf.to_container(cfg.trainer, resolve=True)
    trainer_cfg["dcp_path"] = TEST_DCP_DIR
    trainer_cfg["checkpoint"] = {
        "enable": True,
        "folder": "/tmp/test_inflight_checkpoints",
        "initial_load_path": cached_dir,
        "initial_load_in_hf": True,
    }

    actors_trainer_cfg = OmegaConf.to_container(cfg.actors.trainer, resolve=True)
    policy_cfg = OmegaConf.to_container(cfg.policy, resolve=True)

    # Initialize provisioner and torchstore
    if cfg.get("provisioner", None) is not None:
        await init_provisioner(
            ProvisionerConfig(launcher_config=LauncherConfig(**cfg.provisioner))
        )
    else:
        await init_provisioner()
    await ts.initialize(strategy=ts.ControllerStorageVolumes())

    # Create generator with in-flight updates enabled
    policy_inflight = await Generator.options(**services_policy_cfg).as_service(
        **policy_cfg,
        enable_in_flight_weight_updates=True,
    )

    # Create trainer
    trainer = await MockTitanTrainer.options(**actors_trainer_cfg).as_actor(
        **trainer_cfg
    )

    # Get tokenizer for chat template formatting
    tokenizer = get_tokenizer(model_card)

    yield {
        "generator_inflight": policy_inflight,
        "trainer": trainer,
        "config": cfg,
        "tokenizer": tokenizer,
    }

    # Teardown
    logger.info("Shutting down services...")

    # Call cleanup to destroy process group before shutdown
    # This prevents TCPStore connection errors from NCCL heartbeat threads
    await trainer.cleanup.call()

    # Shutdown sequentially to avoid race conditions
    # Note: Do NOT call provisioner.shutdown() - manual shutdowns above are sufficient
    # and calling shutdown() would try to shutdown already-stopped services again
    # Taken from test_policy_update.py
    await policy_inflight.shutdown()
    await MockTitanTrainer.shutdown(trainer)
    await ts.shutdown()

    # Cleanup DCP directory
    path = Path(TEST_DCP_DIR)
    if path.exists() and path.is_dir():
        try:
            shutil.rmtree(path)
            logger.info(f"Cleaned up {TEST_DCP_DIR}")
        except Exception as e:
            logger.warning(f"Failed to remove {TEST_DCP_DIR}: {e}")


class TestInflightWeightUpdates:
    """Tests for in-flight weight update functionality."""

    @pytest.mark.asyncio(loop_scope="module")
    @requires_cuda_with_4_gpus
    async def test_multiple_concurrent_generations_with_update(
        self, setup_inflight_test
    ):
        """Test weight update with multiple concurrent generation requests."""
        generator = setup_inflight_test["generator_inflight"]
        trainer = setup_inflight_test["trainer"]
        tokenizer = setup_inflight_test["tokenizer"]

        v0 = uuid.uuid4().int
        v1 = v0 + 1

        await trainer.push_weights.call(policy_version=v0)
        await generator.update_weights.fanout(version=v0)

        # Push new weights
        await trainer.push_weights.call(policy_version=v1)

        # Create multiple generation tasks with chat-formatted prompts
        raw_prompts = [
            "Tell me a joke about programming.",
            # "What is machine learning?",
            "Describe a beautiful sunset.",
            "How do computers work?",
            "/no_think Respond only with just one word: 'HELLO' and stop",
        ]
        prompts = [_format_prompt_with_chat_template(tokenizer, p) for p in raw_prompts]

        async def run_generations():
            tasks = [generator.generate.route(prompt) for prompt in prompts]
            return await asyncio.gather(*tasks, return_exceptions=True)

        async def run_update():
            await asyncio.sleep(3)  # Let some generations finish, especially 4th one
            await generator.update_weights.fanout(version=v1)

        # Run concurrently
        gen_results, update_result = await asyncio.gather(
            run_generations(),
            run_update(),
            return_exceptions=True,
        )

        # All generations should succeed
        assert isinstance(gen_results, list), f"Expected list, got {type(gen_results)}"
        valid_results = []
        for i, result in enumerate(gen_results):
            assert not isinstance(result, Exception), f"Generation {i} failed: {result}"
            result_list = list(result)
            assert len(result_list) > 0, f"Generation {i} returned no completions"
            valid_results.append(result_list)
            logger.info(f"Gen {i}: {result_list[0].text[:50]}...")

        # Update should succeed
        assert update_result is None or not isinstance(
            update_result, Exception
        ), f"Update failed: {update_result}"

        # Verify final version
        versions = await generator.get_version.fanout()
        version_after = versions[0]
        assert version_after == v1, f"Expected version {v1}, got {version_after}"

        # Verify metadata contains expected fields
        first_completion = valid_results[0][0]
        metadata = first_completion.metadata
        assert (
            "num_cached_tokens" in metadata
        ), "Metadata should contain num_cached_tokens"

        # Count mixed-policy completions
        mixed_count = 0
        for result_list in valid_results:
            if result_list[0].metadata.get("mixed_policy", False):
                mixed_count += 1
        logger.info(f"Mixed-policy completions: {mixed_count}/{len(prompts)}")

    @pytest.mark.asyncio(loop_scope="module")
    @requires_cuda_with_4_gpus
    async def test_sequential_weight_updates(self, setup_inflight_test):
        """Test multiple sequential weight updates work correctly."""
        generator = setup_inflight_test["generator_inflight"]
        trainer = setup_inflight_test["trainer"]
        tokenizer = setup_inflight_test["tokenizer"]

        base_version = uuid.uuid4().int
        versions = [base_version + i for i in range(3)]

        for v in versions:
            await trainer.push_weights.call(policy_version=v)
            await generator.update_weights.fanout(version=v)

            # Generate after each update
            prompt = _format_prompt_with_chat_template(tokenizer, f"Version {v} test")
            completions = await generator.generate.route(prompt)
            assert len(completions) > 0

            # Verify version
            current_versions = await generator.get_version.fanout()
            current_version = current_versions[0]
            assert current_version == v, f"Expected {v}, got {current_version}"

            logger.info(f"Updated to v{v}, generated: {completions[0].text[:30]}...")

    @pytest.mark.asyncio(loop_scope="module")
    @requires_cuda_with_4_gpus
    async def test_rapid_weight_updates(self, setup_inflight_test):
        """Test rapid successive weight updates don't cause issues."""
        generator = setup_inflight_test["generator_inflight"]
        trainer = setup_inflight_test["trainer"]
        tokenizer = setup_inflight_test["tokenizer"]

        base_version = uuid.uuid4().int

        # Push multiple versions quickly
        for i in range(5):
            v = base_version + i
            await trainer.push_weights.call(policy_version=v)

        # Rapidly request updates (they're serialized by update_lock)
        update_tasks = []
        for i in range(5):
            v = base_version + i
            update_tasks.append(generator.update_weights.fanout(version=v))

        # Wait for all updates
        await asyncio.gather(*update_tasks)

        # Final version should be the last one
        versions = await generator.get_version.fanout()
        final_version = versions[0]
        expected_final = base_version + 4
        assert (
            final_version == expected_final
        ), f"Expected final version {expected_final}, got {final_version}"

        # Verify generation still works
        prompt = _format_prompt_with_chat_template(tokenizer, "After rapid updates")
        completions = await generator.generate.route(prompt)
        assert len(completions) > 0

    @pytest.mark.asyncio(loop_scope="module")
    @requires_cuda_with_4_gpus
    async def test_mixed_policy_correctly_marks_inflight_requests(
        self, setup_inflight_test
    ):
        """
        Test that mixed_policy metadata is correctly set for requests
        that were in-flight during a weight update.

        This test uses longer generation and aggressive update timing
        to maximize the chance of catching a mixed-policy scenario.
        """
        generator = setup_inflight_test["generator_inflight"]
        trainer = setup_inflight_test["trainer"]
        tokenizer = setup_inflight_test["tokenizer"]

        v0 = uuid.uuid4().int
        v1 = v0 + 1

        # Setup initial version
        await trainer.push_weights.call(policy_version=v0)
        await generator.update_weights.fanout(version=v0)

        # Push new version
        await trainer.push_weights.call(policy_version=v1)

        # Use a very long prompt to encourage longer generation
        long_prompt = _format_prompt_with_chat_template(
            tokenizer,
            """Write a very long and detailed story about a robot learning to paint.
        Include multiple chapters, character development, and vivid descriptions of the art
        the robot creates. Make sure to describe at least 5 different paintings in detail.""",
        )

        # Start many long generations
        gen_tasks = [generator.generate.route(long_prompt) for _ in range(3)]

        # Start generation first, then update immediately
        async def do_update():
            await asyncio.sleep(0.01)
            await generator.update_weights.fanout(version=v1)

        all_tasks = [
            asyncio.gather(*gen_tasks, return_exceptions=True),
            do_update(),
        ]

        gen_results_wrapper, _ = await asyncio.gather(
            *all_tasks, return_exceptions=True
        )

        # Check results
        assert isinstance(
            gen_results_wrapper, list
        ), f"Expected list, got {type(gen_results_wrapper)}"
        gen_results = gen_results_wrapper

        mixed_policy_count = 0
        total_valid = 0
        for i, result_raw in enumerate(gen_results):
            if isinstance(result_raw, Exception):
                logger.warning(f"Generation {i} failed: {result_raw}")
                continue

            result = list(result_raw)
            total_valid += 1
            assert len(result) > 0
            is_mixed = result[0].metadata.get("mixed_policy", False)
            if is_mixed:
                mixed_policy_count += 1
            logger.info(
                f"Gen {i}: mixed_policy={is_mixed}, "
                f"tokens={len(result[0].token_ids)}"
            )

        logger.info(
            f"Total mixed-policy completions: {mixed_policy_count}/{total_valid}"
        )

        # Verify update was applied
        versions = await generator.get_version.fanout()
        final_version = versions[0]
        assert final_version == v1


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
