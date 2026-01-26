# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Forge-specific MonarchExecutor with TorchStore weight sync.

This module extends the upstream-compatible MonarchExecutor with TorchStore
integration for weight synchronization in RL training loops. It provides:

- ForgeWorkerWrapper: Extends WorkerWrapper with TorchStore weight loading
- ForgeMonarchExecutor: Extends MonarchExecutor with TorchStore Controller handling

Use this executor when you need weight updates from TorchStore (e.g., GRPO training).
For inference-only workloads, use the base MonarchExecutor directly.

TODO: Add shared memory weight prefetch support (prefetch_weights_to_shm, n_fetcher_procs)
      as in v0 Generator for faster weight loading.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import os
from typing import Optional

import cloudpickle
from forge.actors._torchstore_utils import extract_param_name, get_param_prefix
from forge.actors.vllm.v1.monarch_executor import MonarchExecutor, WorkerWrapper
from monarch.actor import endpoint
from torchstore.client import LocalClient

logger = logging.getLogger(__name__)


class ForgeWorkerWrapper(WorkerWrapper):
    """Worker wrapper with TorchStore weight sync capabilities."""

    def __init__(self, vllm_config):
        super().__init__(vllm_config)
        self._torchstore_controller = None
        self._torchstore_client: Optional[LocalClient] = None

    @endpoint
    def set_torchstore_controller(self, controller) -> None:
        """Store TorchStore Controller reference for weight updates.

        Workers run in a subprocess with a different _controller_controller,
        so they can't find the Controller via get_or_spawn_controller.
        The Controller reference is passed explicitly from ForgeMonarchExecutor.
        """
        self._torchstore_controller = controller
        self._torchstore_client = None  # Reset cached client

    @endpoint
    def update_weights(self, version: int) -> int:
        """Load weights directly from torchstore.

        Args:
            version: Policy version to load from torchstore

        Returns:
            Number of parameters loaded
        """
        return asyncio.run(self._load_from_torchstore(version))

    async def _get_torchstore_client(self) -> LocalClient:
        """Get or create a LocalClient using the passed Controller reference.

        Workers can't use ts.client() directly because they're in a subprocess
        with a different _controller_controller. Instead, we create a LocalClient
        using the Controller reference passed from ForgeMonarchExecutor.
        """
        if self._torchstore_client is not None:
            return self._torchstore_client

        if self._torchstore_controller is None:
            raise RuntimeError(
                "TorchStore Controller not set. "
                "ForgeMonarchExecutor must call set_torchstore_controller before weight updates."
            )

        strategy = await self._torchstore_controller.get_controller_strategy.call_one()
        self._torchstore_client = LocalClient(
            controller=self._torchstore_controller,
            strategy=strategy,
        )
        return self._torchstore_client

    async def _load_from_torchstore(self, version: int) -> int:
        """Async helper to load from torchstore using the passed Controller."""
        client = await self._get_torchstore_client()
        prefix = get_param_prefix(version)
        matching_keys = await client.keys(prefix)
        model = self.worker.model_runner.model
        loaded_count = 0
        for key in matching_keys:
            name = extract_param_name(key)
            param = await client.get(key)
            model.load_weights([(name, param.cuda())])
            del param
            loaded_count += 1
        return loaded_count

    @endpoint
    def save_model_params(self):
        """Save model parameters before weight update, used for testing purposes only."""
        logger.info("[WorkerWrapper] save model parameters for testing.")
        if not hasattr(self, "_test_prev_params"):
            self._test_prev_params = {}
        for name, param in self.worker.model_runner.model.named_parameters():
            self._test_prev_params[name] = param.detach().cpu()
        logger.info(
            "[WorkerWrapper] finished saving model parameters, len = %d",
            len(self._test_prev_params),
        )

    @endpoint
    def validate_model_params(self, validate_fn):
        """Validate updated model params using validate_fn."""
        logger.info("[WorkerWrapper] start validating model parameters.")
        if not hasattr(self, "_test_prev_params"):
            self._test_prev_params = {}
        return validate_fn(
            self._test_prev_params, self.worker.model_runner.model, logger
        )


class ForgeMonarchExecutor(MonarchExecutor):
    """MonarchExecutor with TorchStore integration for weight sync.

    Extends the base MonarchExecutor to:
    - Deserialize TorchStore Controller from environment
    - Pass Controller to workers for direct weight loading
    - Use ForgeWorkerWrapper instead of base WorkerWrapper
    """

    worker_class = ForgeWorkerWrapper

    def _init_executor(self) -> None:
        """Initialize executor and deserialize TorchStore Controller."""
        super()._init_executor()

        controller_str = os.environ.get("VLLM_TORCHSTORE_CONTROLLER")
        if controller_str:
            logger.info(
                "[ForgeMonarchExecutor] Deserializing TorchStore Controller from environment..."
            )
            self.torchstore_controller = cloudpickle.loads(
                base64.b64decode(controller_str)
            )
            logger.info(
                f"[ForgeMonarchExecutor] TorchStore Controller deserialized: {self.torchstore_controller}"
            )
            self.workers.set_torchstore_controller.call(
                self.torchstore_controller
            ).get()

        else:
            self.torchstore_controller = None
            logger.warning(
                "[ForgeMonarchExecutor] No TorchStore Controller found in environment. "
                "Weight updates via torchstore will not work."
            )
