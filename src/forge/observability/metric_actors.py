# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import logging
from typing import Any, Dict, Optional

from monarch.actor import Actor, endpoint, get_or_spawn_controller, ProcMesh, this_proc

from forge.observability.metrics import (
    get_logger_backend_class,
    LoggerBackend,
    LoggingMode,
    MetricCollector,
    reduce_metrics_states,
)

from forge.observability.utils import detect_actor_name_from_call_stack

logger = logging.getLogger(__name__)

_global_logger = None


async def get_or_create_metric_logger(
    proc_mesh: ProcMesh | None = None,
    process_name: str | None = None,
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
        process_name: Optional meaningful process name (e.g., "TrainActor", "GeneratorActor") for logging.
            If None, will auto-detect from call stack or default to "UnknownActor" if not found.

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
            "console": {"logging_mode": "global_reduce"},
            "wandb": {"project": "my_project", "logging_mode": "per_rank_no_reduce"}
        })

        # Initialize services...
        policy = await Policy.as_service(...)

        # Training loop
        for step in range(max_steps):
            record_metric("loss", 1.2, reduction_type=Reduce.MEAN)
            # ... training code with record_metric() calls ...
            await mlogger.flush(step)  # Log metrics for this step

        # Shutdown
        await mlogger.shutdown()
    """

    if process_name is None:
        process_name = detect_actor_name_from_call_stack()

    # Get or create the singleton global logger
    global _global_logger
    if _global_logger is None:
        _global_logger = await get_or_spawn_controller(
            "global_logger", GlobalLoggingActor
        )
    global_logger = _global_logger

    # Sanity check that if we already have a LocalFetcherActor,
    # it is registered with the global logger
    proc = proc_mesh if proc_mesh is not None else this_proc()
    proc_has_local_fetcher = hasattr(proc, "_local_fetcher")
    global_logger_has_local_fetcher = await global_logger.has_fetcher.call_one(proc)
    if proc_has_local_fetcher != global_logger_has_local_fetcher:
        raise ValueError(
            f"Inconsistent logging state for proc {proc}: "
            f"proc has _local_fetcher={proc_has_local_fetcher}, "
            f"but global_logger has registration={global_logger_has_local_fetcher}. "
            f"This indicates a bug in logging setup/teardown. "
            f"Both should be True (already setup) or both False (needs setup)."
        )

    # Setup local_fetcher_actor if needed
    if not proc_has_local_fetcher:
        local_fetcher_actor = proc.spawn(
            "local_fetcher_actor", LocalFetcherActor, global_logger, process_name
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

    def __init__(
        self,
        global_logger: Optional["GlobalLoggingActor"] = None,
        process_name: str | None = None,
    ) -> None:
        self.global_logger = global_logger
        self.process_name = process_name  # Passed MetricCollector for logging

    @endpoint
    async def flush(
        self, step: int, return_state: bool = False
    ) -> Dict[str, Dict[str, Any]]:
        """Log to local logger backends (if any), reset accumulators and return metric states dict if return_state=True.
        This should only ever be called by the global logger.

        Args:
            step (int): step used by backends to align all metrics on the same x-axis
            return_state (bool): Used by GlobalLoggingActor for reduction across all ranks.
                If False, returns empty dict, else returns the state of all metrics collected.
        Returns:
            Dict[str, Dict[str, Any]]: Dict of {metric_key: metric_state},
                e.g., {"loss": {"reduction_type": "mean", "sum": 1.2, "count": 3}}.
        """
        collector = MetricCollector()
        result = await collector.flush(step, return_state=return_state)
        return result

    @endpoint
    async def init_backends(
        self,
        metadata_per_primary_backend: Dict[str, Dict[str, Any]],
        config: Dict[str, Any],
        step: int = 0,
    ):
        """Init local (per-rank) logger backends and MetricCollector.

        Args:
            metadata_per_primary_backend: Metadata from primary backends for shared state.
            config: Backend configurations with logging modes and settings.
            step: Initial step for metrics.
        """
        collector = MetricCollector()
        await collector.init_backends(
            metadata_per_primary_backend, config, step, actor_name=self.process_name
        )

    @endpoint
    async def shutdown(self):

        collector = MetricCollector()
        await collector.shutdown()


class GlobalLoggingActor(Actor):
    """Coordinates metric logging across all ranks for every step.

    Supports multiple logging backends (e.g., WandB, TensorBoard, etc.),
    for per-rank and/or global reduction logging modes.

    If a backend config has flag `reduce_across_ranks=False`, an instance of the backend
    is initialized per-rank, otherwise it is done once globally.

    This GlobalLoggingActor should be spawned once in the controller. A LocalFetcherActor
    is automatically spawned per-rank in `forge.controller.provisioner.py` and registered
    with this actor. The LocalFetcherActor is responsible for instantiating
    the per-rank MetricCollector.

    In summary, the flow is:
    - GlobalLoggingActor.init_backends() -> LocalFetcherActor.init_backends() -> per-rank MetricCollector.init_backends()
    - GlobalLoggingActor.flush() -> LocalFetcherActor.flush() -> per-rank MetricCollector.flush
    """

    def __init__(self):
        self.fetchers: Dict[str, LocalFetcherActor] = {}
        self.config: Dict[str, Any] | None = None
        self.global_logger_backends: Dict[str, LoggerBackend] = {}
        self.metadata_per_primary_backend: Dict[str, Dict[str, Any]] = {}

    def _validate_backend_config(
        self, backend_name: str, config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate and normalize backend configuration."""
        if "logging_mode" not in config:
            logger.debug(
                f"logging_mode not provided for backend {backend_name}. Defaulting to global_reduce."
            )

        mode_str = config.get("logging_mode", "global_reduce")
        mode = LoggingMode(mode_str)

        # Validate per_rank_share_run configuration
        share_run = config.get("per_rank_share_run", False)
        if mode == LoggingMode.GLOBAL_REDUCE and share_run:
            logger.warning(
                f"{backend_name}: per_rank_share_run ignored in {mode.value} mode."
                "Set it to False or change logging_mode to per rank."
            )

        return {
            **config,
            "logging_mode": mode,
        }

    @endpoint
    async def init_backends(self, config: Dict[str, Any]):
        """Sets config in global actor, initializes primary backends and eagerly initializes MetricCollectors
        in all registered fetchers.

        The backend instantiation is controlled by the logging_mode field. Primary backends
        (instantiated in the controller) can provide metadata to be shared with secondary backends on ranks,
        e.g. shared run IDs for WandB. For details on logging modes, see `forge.observability.metrics.LoggingMode`.

        Args:
            config (Dict[str, Any]): Config for metric logging where keys are backend names.
                Examples:
                - {"console": {"logging_mode": "global_reduce"}}
                - {"wandb": {"logging_mode": "per_rank_no_reduce", "project": "my_project", "per_rank_share_run": True}}

        Raises:
            ValueError: If backend config is invalid or missing required fields.
        """
        self.config = {}

        # Validate and normalize each backend config
        for backend_name, backend_config in config.items():
            self.config[backend_name] = self._validate_backend_config(
                backend_name, backend_config
            )

        # Initialize backends based on logging mode
        for backend_name, backend_config in self.config.items():
            mode = backend_config["logging_mode"]

            backend = get_logger_backend_class(backend_name)(backend_config)
            await backend.init(role="global")

            # Extract metadata for per-rank shared modes
            if mode != LoggingMode.GLOBAL_REDUCE:
                primary_metadata = backend.get_metadata_for_secondary_ranks() or {}
                self.metadata_per_primary_backend[backend_name] = primary_metadata

            # Store global backends for later flush
            if mode == LoggingMode.GLOBAL_REDUCE:
                self.global_logger_backends[backend_name] = backend

        # Initialize per rank fetchers
        if self.fetchers:
            tasks = [
                fetcher.init_backends.call(
                    self.metadata_per_primary_backend, self.config
                )
                for fetcher in self.fetchers.values()
            ]
            await asyncio.gather(*tasks, return_exceptions=True)

    @endpoint
    async def register_fetcher(self, fetcher: LocalFetcherActor, name: str | ProcMesh):
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
    async def deregister_fetcher(self, name: str | ProcMesh):
        if name not in self.fetchers:
            logger.warning(
                f"Fetcher {name} not registered in GlobalLoggingActor. Cannot deregister."
                f"Available fetchers: {self.fetchers.keys()}"
            )
            return
        del self.fetchers[name]

    @endpoint
    async def flush(self, step: int):
        """
        Triggers parallel flush/reset on all registered fetchers. Per-rank MetricCollectors
        log to local backends and return states if needed for cross-rank reduction.

        Args:
            step (int): step for logging.
        """
        if not self.fetchers:
            return

        config = self.config
        if config is None:
            logger.warning(
                "Cannot flush collected metrics. GlobalLoggingActor.flush() called before init_backends()."
                " No backends will be flushed. Please call in your main file:\n"
                "`mlogger = await get_or_create_metric_logger(process_name='Controller')`\n"
                "`await mlogger.init_backends.call_one(logging_config)`\n"
            )
            return

        # Check if we need states for GLOBAL_REDUCE backends
        requires_reduce = any(
            backend_config["logging_mode"] == LoggingMode.GLOBAL_REDUCE
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

            def extract_values_from_valuemesh(results):
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
                return all_local_states

            all_local_states = extract_values_from_valuemesh(results)

            if not all_local_states:
                logger.warning(f"No states to reduce for step {step}")
                return

            # Reduce metrics from states
            reduced_metrics = reduce_metrics_states(all_local_states)

            # Log to global backends
            for backend_name, backend in self.global_logger_backends.items():
                await backend.log_batch(reduced_metrics, step)

    @endpoint
    def has_fetcher(self, name: str | ProcMesh) -> bool:
        """Check if a fetcher is registered with the given name."""
        return name in self.fetchers

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
