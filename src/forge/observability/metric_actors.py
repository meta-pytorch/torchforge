# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import logging
import os
from typing import Any, Dict, Optional

from monarch.actor import Actor, endpoint, get_or_spawn_controller, ProcMesh, this_proc

from forge.env_constants import FORGE_DISABLE_METRICS
from forge.observability.metrics import (
    BackendRole,
    get_logger_backend_class,
    LoggerBackend,
    MetricCollector,
    reduce_metrics_states,
)

logger = logging.getLogger(__name__)

_global_logger = None


async def get_or_create_metric_logger(
    proc_mesh: ProcMesh | None = None,
) -> "GlobalLoggingActor":
    """Initializes a LocalFetcherActor in the specified process mesh (or current process if None),
    if not already initialized, registers it with the GlobalLoggingActor and returns the
    GlobalLoggingActor instance.

    There are primarily two ways to use this function:
    1. In the main process, call `get_or_create_metric_logger()` to get the global logger.
    2. In service processes, call `get_or_create_metric_logger(proc_mesh)` to register the
       local fetcher with the global logger.

    Args:
        proc_mesh: Optional ProcMesh to spawn LocalFetcherActor on. If None,
            uses `monarch.actor.this_proc()`.

    Returns:
        GlobalLoggingActor: The global logging controller.

    Raises:
        ValueError: If the logging state is inconsistent, i.e. the fetcher is already
            registered, but only in the process or the global logger.

    Example:
        from forge.observability.metric_actors import get_or_create_metric_logger
        from forge.observability.metrics import record_metric

        # Main process setup
        mlogger = await get_or_create_metric_logger()

        # Initialize logging backends
        await mlogger.init_backends({
            "console": {"reduce_across_ranks": True},
            "wandb": {"project": "my_project", "reduce_across_ranks": False}
        })

        # Initialize services...
        policy = await Policy.as_service(...)

        # Training loop
        for step in range(max_steps):
            record_metric("loss", 1.2, step, reduction_type=Reduce.MEAN)
            # ... training code with record_metric() calls ...
            await mlogger.flush(step)  # Log metrics for this step

        # Shutdown
        await mlogger.shutdown()
    """
    # Get or create the singleton global logger
    global _global_logger
    if _global_logger is None:
        _global_logger = await get_or_spawn_controller(
            "global_logger", GlobalLoggingActor
        )
    global_logger = _global_logger

    # Determine process context
    proc = proc_mesh if proc_mesh is not None else this_proc()

    # Check current state for consistency
    proc_has_local_fetcher = hasattr(proc, "_local_fetcher")
    global_logger_has_local_fetcher = await global_logger.has_fetcher.call_one(proc)

    # Consistency check: both should be in sync
    if proc_has_local_fetcher != global_logger_has_local_fetcher:
        raise ValueError(
            f"Inconsistent logging state for proc {proc}: "
            f"proc has _local_fetcher={proc_has_local_fetcher}, "
            f"but global_logger has registration={global_logger_has_local_fetcher}. "
            f"This indicates a bug in logging setup/teardown. "
            f"Both should be True (already setup) or both False (needs setup)."
        )

    # Setup local_fetcher_actor if needed (unless disabled by environment flag)
    if (
        not proc_has_local_fetcher
        and os.getenv(FORGE_DISABLE_METRICS, "false").lower() != "true"
    ):
        local_fetcher_actor = proc.spawn(
            "local_fetcher_actor", LocalFetcherActor, global_logger
        )
        await global_logger.register_fetcher.call_one(local_fetcher_actor, proc)
        proc._local_fetcher = local_fetcher_actor  # pyre-ignore

    return global_logger


class LocalFetcherActor(Actor):
    """Thin per-process actor used to trigger MetricCollector singleton
    operations without direct access. It is what GlobalLoggingActor
    uses to broadcast inits/flushes across ranks.

    GlobalLoggingActor -> per-rank LocalFetcherActor -> per-rank MetricCollector
    """

    def __init__(self, global_logger: Optional["GlobalLoggingActor"] = None) -> None:
        self.global_logger = global_logger
        _is_initialized = False

    @endpoint
    async def flush(
        self, global_step: int, return_state: bool = False
    ) -> Dict[str, Dict[str, Any]]:
        """Log to local logger backends (if any), reset accumulators and return metric states dict if return_state=True.
        This should only ever be called by the global logger.

        Args:
            global_step (int): step used by backends to align all metrics on the same x-axis
            return_state (bool): Used by GlobalLoggingActor for reduction across all ranks.
                If False, returns empty dict, else returns the state of all metrics collected.
        Returns:
            Dict[str, Dict[str, Any]]: Dict of {metric_key: metric_state},
                e.g., {"loss": {"reduction_type": "mean", "sum": 1.2, "count": 3}}.
        """
        collector = MetricCollector()
        result = await collector.flush(global_step, return_state=return_state)
        return result

    @endpoint
    async def init_backends(
        self,
        metadata_per_primary_backend: Dict[str, Dict[str, Any]],
        config: Dict[str, Any],
    ) -> None:
        """Init local (per-rank) logger backends and MetricCollector."""
        collector = MetricCollector()
        await collector.init_backends(metadata_per_primary_backend, config)

    @endpoint
    async def shutdown(self) -> None:
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
        self.global_logger_backends: Dict[str, LoggerBackend] = {}
        self.metadata_per_primary_backend: Dict[str, Dict[str, Any]] = {}

    @endpoint
    async def init_backends(self, config: Dict[str, Any]) -> None:
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
                e.g. {"console": {"reduce_across_ranks": True}, "wandb": {"reduce_across_ranks": False}}
        """
        self.config = config

        for backend_name, backend_config in config.items():
            backend = get_logger_backend_class(backend_name)(backend_config)
            await backend.init(role=BackendRole.GLOBAL)

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
    async def register_fetcher(
        self, fetcher: LocalFetcherActor, name: str | ProcMesh
    ) -> None:
        """Registers a fetcher with the global actor. Each key represents a process mesh.
        If there are 2 processes, each with 2 replicas with N gpus, we would
        have 4 keys, i.e. 2 proces meshes, each with 2 replicas."""
        self.fetchers[name] = fetcher  # pyre-ignore

        # Self-init for respawned actors
        if self.config:
            logger.debug(f"Initializing new LocalFetcherActor {name}")
            await fetcher.init_backends.call(
                self.metadata_per_primary_backend, self.config
            )

    @endpoint
    async def deregister_fetcher(self, name: str | ProcMesh) -> None:
        if name not in self.fetchers:
            logger.warning(
                f"Fetcher {name} not registered in GlobalLoggingActor. Cannot deregister."
                f"Available fetchers: {self.fetchers.keys()}"
            )
            return
        del self.fetchers[name]

    @endpoint
    async def flush(self, global_step: int) -> None:
        """
        Triggers parallel flush/reset on all registered fetchers. Per-rank MetricCollectors
        log to local backends and return states if needed for cross-rank reduction.

        Args:
            global_step (int): step for logging.
        """
        if not self.fetchers:
            return

        config = self.config
        if config is None:
            logger.warning(
                "GlobalLoggingActor flush() called before init_backends(). "
                "No backends will be flushed."
            )
            return
        # if reduce_across_ranks=True, we need to reduce the states from all ranks
        # and log with the primary backend
        requires_reduce = any(
            backend_config.get("reduce_across_ranks", True)
            for backend_config in config.values()
        )

        logger.debug(
            f"Global flush for global_step {global_step}: {len(self.fetchers)} fetchers"
        )

        # Broadcast flush to all fetchers
        results = await asyncio.gather(
            *[
                f.flush.call(global_step, return_state=requires_reduce)
                for f in self.fetchers.values()
            ],
            return_exceptions=True,
        )

        if requires_reduce:
            # Handle exceptions and extract values from ValueMesh results
            all_local_states = []
            for result in results:
                if isinstance(result, BaseException):
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
                logger.warning(f"No states to reduce for global_step {global_step}")
                return

            # Reduce metrics from states
            reduced_metrics = reduce_metrics_states(all_local_states)

            # Log to each global logger_backend
            for (
                logger_backend_name,
                logger_backend,
            ) in self.global_logger_backends.items():
                await logger_backend.log(reduced_metrics, global_step)

    @endpoint
    def has_fetcher(self, name: str | ProcMesh) -> bool:
        """Check if a fetcher is registered with the given name."""
        return name in self.fetchers

    @endpoint
    def get_fetcher_count(self) -> int:
        return len(self.fetchers)

    @endpoint
    async def shutdown(self) -> None:
        # Finish per-rank logger_backends via fetchers
        if self.fetchers:
            tasks = [fetcher.shutdown.call() for fetcher in self.fetchers.values()]
            await asyncio.gather(*tasks, return_exceptions=True)
        # Finish global logger_backends
        for logger_backend_name, logger_backend in self.global_logger_backends.items():
            await logger_backend.finish()
