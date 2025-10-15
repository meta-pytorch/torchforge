# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import logging
from typing import Any, Union

from monarch.actor import (
    Actor,
    context,
    endpoint,
    get_or_spawn_controller,
    ProcMesh,
    this_proc,
)

from forge.env import FORGE_DISABLE_METRICS
from forge.observability.metrics import (
    BackendRole,
    get_logger_backend_class,
    LoggerBackend,
    LoggingMode,
    MetricCollector,
    reduce_metrics_states,
)


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
        process_name: Optional process name (e.g., "TrainActor", "GeneratorActor") for logging.
            If None, will be auto-detected from the mesh_name provided during actor initialization or
                a generic mesh name if one was not provided.

    Returns:
        GlobalLoggingActor: The global logging controller.

    Raises:
        ValueError: If the logging state is inconsistent, i.e. the fetcher is already
            registered, but only in the process or the global logger.

    Example:
        from forge.observability.metric_actors import get_or_create_metric_logger
        from forge.observability.metrics import record_metric

        # Main process setup
        mlogger = await get_or_create_metric_logger(process_name="Controller")

        # Initialize logging backends
        await mlogger.init_backends({
            "console": {"logging_mode": "global_reduce"},
            "wandb": {"project": "my_project", "logging_mode": "per_rank_reduce"}
        })

        # Initialize services...
        policy = await Generator.as_service(...)

        # Training loop
        for step in range(max_steps):
            record_metric("loss", 1.2, reduction_type=Reduce.MEAN)
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

    # Auto-detect process_name from proc mesh if not provided
    if process_name is None:
        ctx = context()
        process_name = ctx.actor_instance.actor_id.actor_name

    # Check current state for consistency
    proc_has_local_fetcher = hasattr(proc, "_local_fetcher")
    global_logger_has_local_fetcher = await global_logger.has_fetcher.call_one(proc)

    # Consistency check: both should be in sync
    if proc_has_local_fetcher != global_logger_has_local_fetcher:
        raise ValueError(
            f"Inconsistent logging state for {proc=} with {process_name=}: "
            f"proc has _local_fetcher={proc_has_local_fetcher}, "
            f"but global_logger has registration={global_logger_has_local_fetcher}. "
            f"This indicates a bug in logging setup/teardown. "
            f"Both should be True (already setup) or both False (needs setup)."
        )

    # Setup local_fetcher_actor if needed (unless disabled by environment flag)
    if not proc_has_local_fetcher and not FORGE_DISABLE_METRICS.get_value():
        local_fetcher_actor = proc.spawn(
            "local_fetcher_actor", LocalFetcherActor, global_logger, process_name
        )
        await global_logger.register_fetcher.call_one(local_fetcher_actor, proc)
        proc._local_fetcher = local_fetcher_actor  # pyre-ignore

    return global_logger


class LocalFetcherActor(Actor):
    """Thin per-rank actor used to trigger MetricCollector singleton
    operations without direct access. It is what GlobalLoggingActor
    uses to broadcast inits/flushes across ranks.

    GlobalLoggingActor -> per-rank LocalFetcherActor -> per-rank MetricCollector
    """

    def __init__(
        self,
        global_logger: Union["GlobalLoggingActor", None] = None,
        process_name: str | None = None,
    ) -> None:
        self.global_logger = global_logger
        self.process_name = process_name  # Passed to MetricCollector for logging
        _is_initialized = False

    @endpoint
    async def flush(
        self, global_step: int, return_state: bool = False
    ) -> dict[str, dict[str, Any]]:
        """Log to local logger backends (if any), reset accumulators and return metric states dict if return_state=True.
        This should only ever be called by the global logger.

        Args:
            global_step (int): step used by backends to align all metrics on the same x-axis
            return_state (bool): Used by GlobalLoggingActor for reduction across all ranks.
                If False, returns empty dict, else returns the state of all metrics collected.
        Returns:
            dict[str, dict[str, Any]]: of {metric_key: metric_state},
                e.g., {"loss": {"reduction_type": "mean", "sum": 1.2, "count": 3}}.
        """
        collector = MetricCollector()
        result = await collector.flush(global_step, return_state=return_state)
        return result

    @endpoint
    async def init_backends(
        self,
        metadata_per_controller_backend: dict[str, dict[str, Any]],
        config: dict[str, Any],
        global_step: int = 0,
    ) -> None:
        """Init local (per-rank) logger backends and MetricCollector.

        Args:
            metadata_per_controller_backend (dict[str, dict[str, Any]]): Metadata from controller backends for shared state.
            config (dict[str, Any]): Backend configurations with logging modes and settings.
            global_step (int): Initial step for metrics.
        """
        collector = MetricCollector()
        await collector.init_backends(
            metadata_per_controller_backend,
            config,
            global_step,
            process_name=self.process_name,
        )

    @endpoint
    async def shutdown(self) -> None:
        collector = MetricCollector()
        await collector.shutdown()


class GlobalLoggingActor(Actor):
    """Coordinates metric logging across all ranks for every global step.

    Supports multiple logging backends (e.g., WandB, TensorBoard, etc.),
    for per-rank and/or global reduction logging modes.

    This GlobalLoggingActor should be spawned once in the controller. A LocalFetcherActor
    is automatically spawned per-rank in `forge.controller.provisioner.py` and registered
    with this actor. The LocalFetcherActor is responsible for instantiating
    the per-rank MetricCollector and working as a bridge between GlobalLoggingActor and processes.

    In summary, the flow is:
    - GlobalLoggingActor.init_backends() -> LocalFetcherActor.init_backends() -> per-rank MetricCollector.init_backends()
    - GlobalLoggingActor.flush() -> LocalFetcherActor.flush() -> per-rank MetricCollector.flush
    """

    def __init__(self):
        self.fetchers: dict[str, LocalFetcherActor] = {}
        self.config: dict[str, Any] | None = None
        self.global_logger_backends: dict[str, LoggerBackend] = {}
        self.metadata_per_controller_backend: dict[str, dict[str, Any]] = {}

    def _validate_backend_config(
        self, backend_name: str, config: dict[str, Any]
    ) -> dict[str, Any]:
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
                f"{backend_name}: per_rank_share_run=True is ignored in {mode.value} mode. "
                "Setting it to False."
            )
            share_run = False

        # WandB-specific warning for suboptimal configuration
        if (
            backend_name == "wandb"
            and mode == LoggingMode.PER_RANK_REDUCE
            and share_run
        ):
            logger.warning(
                "WandB: Using 'per_rank_reduce' with 'per_rank_share_run=True' is not recommended. "
                "This configuration can lead to confusing metrics where reduced values from multiple ranks "
                "are written to the same run/step, displaying only one of them. Consider either:\n"
                "  1. Set 'per_rank_share_run=False' to create separate runs per rank, OR\n"
                "  2. Use 'per_rank_no_reduce' for real-time streaming to a shared run"
            )

        return {
            **config,
            "logging_mode": mode,
            "per_rank_share_run": share_run,
        }

    @endpoint
    async def init_backends(self, config: dict[str, Any]) -> None:
        """Sets config in global actor, initializes controller backends and eagerly initializes MetricCollectors
        in all registered fetchers.

        The backend instantiation is controlled by the logging_mode field. Controller backends
        (instantiated in the controller) can provide metadata to be shared with rank backends,
        e.g. shared run IDs for WandB. For details on logging modes, see `forge.observability.metrics.LoggingMode`.

        Args:
            config (dict[str, Any]): Config for metric logging where keys are backend names.
                Each backend config supports:
                - logging_mode (str | LoggingMode, default "global_reduce"): One of "global_reduce",
                  "per_rank_reduce", or "per_rank_no_reduce". Can be specified as a string or LoggingMode enum.
                - per_rank_share_run (bool, default False): For per-rank modes only. Whether ranks
                  share a single run/logger instance. Ignored for "global_reduce" mode.
                - Additional backend-specific options (e.g., "project" for WandB)

                Example:
                {
                    "console": {"logging_mode": "global_reduce"},
                    "wandb": {
                        "project": "my_project",
                        "logging_mode": "per_rank_no_reduce",
                        "per_rank_share_run": True
                    }
                }

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
            await backend.init(role=BackendRole.GLOBAL)

            # Extract metadata from controller logger to be shared with per-rank loggers
            if mode != LoggingMode.GLOBAL_REDUCE:
                controller_metadata = backend.get_metadata_for_secondary_ranks() or {}
                self.metadata_per_controller_backend[backend_name] = controller_metadata

            # Store global logger backends for later flush
            if mode == LoggingMode.GLOBAL_REDUCE:
                self.global_logger_backends[backend_name] = backend

        # Eager init collectors on all registered fetchers in parallel, passing controller states and config
        if self.fetchers:
            tasks = [
                fetcher.init_backends.call(
                    self.metadata_per_controller_backend, self.config
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
                self.metadata_per_controller_backend, self.config
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
                "Cannot flush collected metrics. GlobalLoggingActor.flush() called before init_backends()."
                " No backends will be flushed. Please call in your main file:\n"
                "`mlogger = await get_or_create_metric_logger(process_name='Controller')`\n"
                "`await mlogger.init_backends.call_one(logging_config)`\n"
            )
            return

        # Check if need to do reduce and retrieve states from fetchers
        requires_reduce = any(
            backend_config["logging_mode"] == LoggingMode.GLOBAL_REDUCE
            for backend_config in config.values()
        )

        logger.debug(
            f"Global flush for global step {global_step}: {len(self.fetchers)} fetchers"
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

            def extract_values_from_valuemesh(results) -> list[dict[str, Any]]:
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
                logger.warning(f"No states to reduce for global_step {global_step}")
                return

            # Reduce metrics from states
            reduced_metrics = reduce_metrics_states(all_local_states)

            # Log to global backends
            for backend_name, backend in self.global_logger_backends.items():
                await backend.log_batch(reduced_metrics, global_step)

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
