# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import logging
from typing import Any, Dict, Optional

from monarch.actor import Actor, endpoint


logger = logging.getLogger(__name__)


class LocalFetcherActor(Actor):
    """Thin per-process actor used to trigger MetricCollector singleton
    operations without direct access. It is what GlobalLoggingActor
    uses to broadcast inits/flushes across ranks.

    GlobalLoggingActor -> per-rank LocalFetcherActor -> per-rank MetricCollector
    """

    def __init__(self, global_logger: Optional["GlobalLoggingActor"] = None) -> None:
        self.global_logger = global_logger

    @endpoint
    async def flush(
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
        result = await collector.flush(step, return_state=return_state)
        return result

    @endpoint
    async def init_backends(
        self,
        metadata_per_primary_backend: Dict[str, Dict[str, Any]],
        config: Dict[str, Any],
    ):
        """Init local (per-rank) logger backends and MetricCollector."""
        from forge.observability.metrics import MetricCollector

        collector = MetricCollector()
        await collector.init_backends(metadata_per_primary_backend, config)

    @endpoint
    async def shutdown(self):
        from forge.observability.metrics import MetricCollector

        collector = MetricCollector()
        await collector.shutdown()


class GlobalLoggingActor(Actor):
    """Coordinates metric logging across all ranks for every training step.

    Supports multiple logging backends (e.g., WandB, TensorBoard, etc.),
    for per-rank and/or global reduction logging modes.

    If a backend config has flag `reduce_across_ranks=False`, an instance of the backend
    is initialized per-rank, otherwise it is done once globally.

    This GlobalLoggingActor should be spawned once in the controller. A LocalFetcherActor
    is automatically spawned per-rank in `forge.controller.provisioner.py` and registered
    with this actor. The LocalFetcherActor is responsible for instantiating
    the per-rank MetricCollector.

    In summary, the flow is:
    - GlobalLoggingActor init_backends() -> LocalFetcherActor init_backends() -> per-rank MetricCollector
    - GlobalLoggingActor flush() -> LocalFetcherActor flush() -> per-rank MetricCollector flush
    """

    def __init__(self):
        self.fetchers: Dict[str, LocalFetcherActor] = {}
        self.config: Dict[str, Any] | None = None
        self.global_logger_backends: Dict[str, "LoggerBackend"] = {}
        self.metadata_per_primary_backend: Dict[str, Dict[str, Any]] = {}

    @endpoint
    async def init_backends(self, config: Dict[str, Any]):
        """
        Sets config in global actor, so other actors can get it, then eagerly initializes backend and MetricCollectors
        in all registered fetchers.

        A backend is always initialized in the controller (primary backend) and can be used as a logger or as a source
        for metadata to be shared with per-rank backends, e.g. shared run IDs for wandb.

        The backend instantiation is controlled by the backend config flag `reduce_across_ranks`: if False,
        a per-rank backend is initialized, i.e. if there are 2 ranks, each will have its own backend,
        and will log independently, i.e. each rank will have its own run in wandb.

        Else, if True, the GlobalLoggingActor will fetch all local metrics collectors to get their states
        and reduce them to a single value, which will be logged by the primary backend in this controller.

        Args:
            config (Dict[str, Any]): Config for metric logging where keys are backend names,
                e.g. {"console": {"log_per_rank": True}, "wandb": {"log_per_rank": False}}
        """
        self.config = config

        # Init global logger_backends and states where needed
        from forge.observability.metrics import get_logger_backend_class

        for backend_name, backend_config in config.items():
            backend = get_logger_backend_class(backend_name)(backend_config)
            await backend.init(role="global")

            # Extract metadata from primary logger to be shared with secondary loggers
            # and store it
            reduce_across_ranks = backend_config.get("reduce_across_ranks", True)
            if not reduce_across_ranks:
                primary_backend_metadata = (
                    backend.get_metadata_for_secondary_ranks() or {}
                )
                self.metadata_per_primary_backend[
                    backend_name
                ] = primary_backend_metadata

            # Store global logger backends
            if reduce_across_ranks:
                self.global_logger_backends[backend_name] = backend

        # Eager init collectors on all registered fetchers in parallel, passing primary states and config
        if self.fetchers:
            tasks = [
                fetcher.init_backends.call(
                    self.metadata_per_primary_backend, self.config
                )
                for fetcher in self.fetchers.values()
            ]
            await asyncio.gather(*tasks, return_exceptions=True)

    @endpoint
    async def register_fetcher(self, fetcher: LocalFetcherActor, name: str):
        """Registers a fetcher with the global actor. Each key represents a process mesh.
        If there are 2 processes, each with 2 replicas with N gpus, we would
        have 4 keys, i.e. 2 proces meshes, each with 2 replicas."""
        self.fetchers[name] = fetcher

        # Self-init for respawned actors
        if self.config:
            logger.debug(f"Initializing new LocalFetchActor {name}")
            await fetcher.init_backends.call(
                self.metadata_per_primary_backend, self.config
            )

    @endpoint
    async def deregister_fetcher(self, name: str):
        if name not in self.fetchers:
            logger.warning(
                f"Fetcher {name} not registered in GlobalLoggingActor. Cannot deregister."
            )
            return
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
        # if reduce_across_ranks=True, we need to reduce the states from all ranks
        # and log with the primary backend
        requires_reduce = any(
            backend_config.get("reduce_across_ranks", True)
            for backend_config in config.values()
        )

        logger.debug(f"Global flush for step {step}: {len(self.fetchers)} fetchers")

        # Broadcast flush to all fetchers
        results = await asyncio.gather(
            *[
                f.flush.call(step, return_state=requires_reduce)
                for f in self.fetchers.values()
            ],
            return_exceptions=True,
        )

        if requires_reduce:
            # Handle exceptions and extract values from ValueMesh results
            all_local_states = []
            for result in results:
                if isinstance(result, Exception):
                    logger.warning(f"Flush failed on a fetcher: {result}")
                    continue

                # result is a generator that outputs a pair [{'gpus': i/N}, {metric_key1: metric_state1, ...}}]
                for gpu_info, local_metric_state in result.items():
                    if isinstance(local_metric_state, dict):
                        all_local_states.append(local_metric_state)
                    else:
                        logger.warning(
                            f"Unexpected result from fetcher. {gpu_info=}, {local_metric_state=}"
                        )

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
    def get_fetcher_count(self) -> int:
        return len(self.fetchers)

    @endpoint
    async def shutdown(self):
        # Finish per-rank logger_backends via fetchers
        if self.fetchers:
            tasks = [fetcher.shutdown.call() for fetcher in self.fetchers.values()]
            await asyncio.gather(*tasks, return_exceptions=True)
        # Finish global logger_backends
        for logger_backend_name, logger_backend in self.global_logger_backends.items():
            await logger_backend.finish()
