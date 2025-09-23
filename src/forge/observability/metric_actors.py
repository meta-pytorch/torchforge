# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import logging
from typing import Any, Dict

from monarch.actor import Actor, endpoint


logger = logging.getLogger(__name__)


class LocalFetcherActor(Actor):
    """Thin per-process actor used to trigger MetricCollector singleton
    operations without direct access. It is what GlobalLoggingActor
    uses to broadcast inits/flushes across ranks.

    GlobalLoggingActor -> per-rank LocalFetcherActor -> per-rank MetricCollector
    """

    @endpoint
    async def log_and_reset(
        self, step: int, return_state: bool = False
    ) -> Dict[str, Dict[str, Any]]:
        """Log to local logger backends (if any), reset accumulators and return metric states dict if return_state=True.

        Args:
            step (int): train step used by backends to align all metrics on the same x-axis
            return_state (bool): Used by GlobalLoggingActor for reduction across all ranks.
                If False, returns empty dict, else returns the state of all metrics collected.
        Returns:
            Dict[str, Dict[str, Any]]: Dict of {metric_key: metric_state},
                e.g., {"loss": {"reduction_type": "mean", "sum": 1.2, "count": 3}}.
        """
        from forge.observability.metrics import MetricCollector

        collector = MetricCollector()
        result = await collector.log_and_reset(step, return_state=return_state)
        return result

    @endpoint
    async def init_collector(
        self, metadata_per_primary_backend: Dict[str, Dict[str, Any]]
    ):
        from forge.observability.metrics import MetricCollector

        collector = MetricCollector()
        await collector.init_local_backends(metadata_per_primary_backend)

    @endpoint
    async def shutdown(self):
        from forge.observability.metrics import MetricCollector

        collector = MetricCollector()
        await collector.shutdown()


class GlobalLoggingActor(Actor):
    """Coordinates metric logging across all ranks for every training step.

    Supports multiple logging backends (e.g., WandB, TensorBoard, etc.),
    and per-rank and global reduction logging modes."""

    def __init__(self):
        self.fetchers: Dict[str, LocalFetcherActor] = {}
        self.config: Dict[str, Any] | None = None
        self.global_logger_backends: Dict[str, "LoggerBackend"] = {}
        self.metadata_per_primary_backend: Dict[str, Dict[str, Any]] = {}

    @endpoint
    async def initialize_backends(self, config: Dict[str, Any]):
        """
        Sets config on global actor and inits backends; broadcasts to registered per-rank fetchers.

        - Validates unique backend classes;
        - Extracts metadata from a primary logger to be shared with secondary loggers (e.g., shared run IDs) for per-rank modes.
        - Eagerly inits metric collectors on fetchers.

        Args:
            config (Dict[str, Any]): Config for metric logging
        """
        self.config = config

        # Validate unique classes
        classes = [b["class"] for b in config["backends"]]
        if len(set(classes)) != len(classes):
            raise ValueError("Duplicate logger_backend classes in config")

        # Init global logger_backends and states where needed
        from forge.observability.metrics import get_logger_backend_class

        for backend_config in config["backends"]:
            cls_name = backend_config["class"]
            backend = get_logger_backend_class(cls_name)(backend_config)
            await backend.init(role="global")

            # Extract metadata from primary logger to be shared with secondary loggers
            # and store it
            log_per_rank = backend_config.get("log_per_rank", True)
            if log_per_rank:
                primary_backend_metadata = (
                    backend.get_metadata_for_secondary_ranks() or {}
                )
                self.metadata_per_primary_backend[cls_name] = primary_backend_metadata

            # Store global logger backends
            if not log_per_rank:
                self.global_logger_backends[cls_name] = backend

        # Eager init collectors on all registered fetchers in parallel, passing primary states
        if self.fetchers:
            tasks = [
                fetcher.init_collector.call(self.metadata_per_primary_backend)
                for fetcher in self.fetchers.values()
            ]
            await asyncio.gather(*tasks, return_exceptions=True)

    @endpoint
    def get_config(self) -> Dict[str, Any] | None:
        """
        Returns the stored logging config for MetricCollector to query during init,
        so local backends can be initialized in their own process.
        """
        return self.config

    @endpoint
    async def register_fetcher(self, fetcher: LocalFetcherActor, name: str):
        """Registers a fetcher with the global actor. Each key represents a process mesh.
        If there are 2 processes, each with 2 replicas with N gpus, we would
        have 4 keys, i.e. 2 proces meshes, each with 2 replicas."""
        self.fetchers[name] = fetcher

    @endpoint
    async def deregister_fetcher(self, name: str):
        del self.fetchers[name]

    @endpoint
    async def flush(self, step: int):
        """
        Triggers parallel flush/reset on all registered fetchers. Per-rank MetricCollectors
        log to local backends and return states if needed for cross-rank reduction.

        Args:
            step (int): Global step for logging.
        """
        if not self.fetchers:
            return

        config = self.config
        has_reduce = any(not b.get("log_per_rank", True) for b in config["backends"])

        logger.debug(f"Global flush for step {step}: {len(self.fetchers)} fetchers")

        # Broadcast log_and_reset to all fetchers
        results = await asyncio.gather(
            *[
                f.log_and_reset.call(step, return_state=has_reduce)
                for f in self.fetchers.values()
            ],
            return_exceptions=True,
        )

        if has_reduce:
            # Handle exceptions and extract values from ValueMesh results
            all_local_states = []
            for res in results:
                if isinstance(res, Exception):
                    logger.warning(f"Flush failed on a fetcher: {res}")
                    continue
                # res is a ValueMesh. TODO: use public API (.items()), but need to parse metadata
                res = res._values
                if isinstance(res, list):
                    all_local_states.extend(
                        [r for r in res if isinstance(r, dict) and r]
                    )
                elif isinstance(res, dict) and res:
                    all_local_states.append(res)

            if not all_local_states:
                logger.warning(f"No states to reduce for step {step}")
                return

            # Reduce
            from forge.observability.metrics import reduce_metrics_states

            reduced_metrics = reduce_metrics_states(all_local_states)

            # Log to each global logger_backend
            for (
                logger_backend_name,
                logger_backend,
            ) in self.global_logger_backends.items():
                await logger_backend.log(reduced_metrics, step)

    @endpoint
    async def shutdown(self):
        # Finish per-rank logger_backends via fetchers
        if self.fetchers:
            tasks = [fetcher.shutdown.call() for fetcher in self.fetchers.values()]
            await asyncio.gather(*tasks, return_exceptions=True)
        # Finish global logger_backends
        for logger_backend_name, logger_backend in self.global_logger_backends.items():
            await logger_backend.finish()
