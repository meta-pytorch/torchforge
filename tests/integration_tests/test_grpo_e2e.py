# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""A simple smoke test that runs the GRPO loop for 3 steps.

Run this with:
PYTHONPATH=. pytest -s tests/integration_tests/test_grpo_e2e.py::test_grpo_smoke_test

"""

import logging
import shutil
from pathlib import Path

import pytest
import torch

from forge.util.config import resolve_hf_hub_paths
from omegaconf import DictConfig, OmegaConf

logger: logging.Logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available",
)

TEST_CHECKPOINT_DIR = "/tmp/grpo_test_checkpoint"


def _load_config(config_path: str) -> DictConfig:
    """Load and resolve config from YAML file."""
    cfg = None
    try:
        cfg = OmegaConf.load(config_path)
    except Exception as e:
        pytest.fail(f"Failed to load config file {config_path}: {e}")

    assert isinstance(cfg, DictConfig)
    cfg = resolve_hf_hub_paths(cfg)
    return cfg


def _cleanup_checkpoint_dir():
    """Clean up test checkpoint directory."""
    path = Path(TEST_CHECKPOINT_DIR)
    if path.exists() and path.is_dir():
        try:
            shutil.rmtree(path)
            logger.info(f"Successfully removed {TEST_CHECKPOINT_DIR}")
        except Exception as e:
            logger.error(f"Failed to remove {TEST_CHECKPOINT_DIR}: {e}")


class TestGRPOEndToEnd:
    """End-to-end integration tests for GRPO training loop."""

    @pytest.mark.asyncio
    @pytest.mark.timeout(600)  # 10 minute timeout to prevent hanging
    @requires_cuda
    async def test_grpo_smoke_test(self):
        """
        Smoke test for GRPO training loop.

        This test runs the full GRPO pipeline for 3 training steps to verify:
        - All actors and services initialize correctly
        - Rollout loop generates completions
        - Rewards are evaluated
        - Reference model computes logprobs
        - Replay buffer collects and batches experiences
        - Trainer updates weights
        - Policy receives weight updates
        - Training completes successfully
        """
        logger.info("=" * 80)
        logger.info("Starting GRPO smoke test")
        logger.info("=" * 80)

        try:
            # Load test config
            config_path = "tests/integration_tests/fixtures/grpo_smoke_test.yaml"
            cfg = _load_config(config_path)

            logger.info("Starting GRPO smoke test with config:")
            logger.info(f"  Model: {cfg.model}")
            logger.info(f"  Group size: {cfg.group_size}")
            logger.info(f"  Batch size: {cfg.local_batch_size}")
            logger.info(f"  Training steps: {cfg.trainer.training.steps}")
            logger.info(
                f"  Max req/res tokens: {cfg.max_req_tokens}/{cfg.max_res_tokens}"
            )

            # Import main here to avoid issues with module-level imports
            from apps.grpo.main import main

            logger.info("Starting main training loop...")
            # Run the main training loop
            # This should run for exactly 3 steps and then exit cleanly
            await main(cfg)

            logger.info("Main training loop completed successfully")
            logger.info("GRPO smoke test completed successfully!")

        except Exception as e:
            logger.error(f"GRPO smoke test failed with error: {e}")
            raise
        finally:
            # Cleanup
            logger.info("Cleaning up test checkpoint directory...")
            _cleanup_checkpoint_dir()
            logger.info("Cleanup complete")
            logger.info("=" * 80)
