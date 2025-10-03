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
    Metric,
    MetricCollector,
    Reduce,
    reduce_metrics_states,
)

from forge.observability.utils import get_actor_name_with_rank

logger = logging.getLogger(__name__)

_global_logger = None


async def get_or_create_metric_logger(
    proc_mesh: ProcMesh | None = None,
    actor_name: str | None = None,
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
        actor_name: Optional meaningful actor name (e.g., "TrainActor", "GeneratorActor") for logging.
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
    # Auto-detect actor name if not provided - get_actor_name_with_rank will extract just the actor name part
    # Auto-detect actor name if not provided
    if actor_name is None:
        # Extract just the actor name from "ActorName_replicaId_rRank" format
        full_name = get_actor_name_with_rank()
        actor_name = full_name.split("_")[0] if "_" in full_name else full_name

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

    # Setup local_fetcher_actor if needed
    if not proc_has_local_fetcher:
        local_fetcher_actor = proc.spawn(
            "local_fetcher_actor", LocalFetcherActor, global_logger, actor_name
        )
        await global_logger.register_fetcher.call_one(local_fetcher_actor, proc)
        proc._local_fetcher = local_fetcher_actor

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
        actor_name: str | None = None,
    ) -> None:
        self.global_logger = global_logger
        self.actor_name = actor_name  # Store the meaningful actor name
        _is_initialized = False

    @endpoint
    async def flush(
        self, step: int, return_state: bool = False
    ) -> Dict[str, Dict[str, Any]]:
        """Log to local logger backends (if any), reset accumulators and return metric states dict if return_state=True.
        This should only ever be called by the global logger.

        Args:
            step (int): train step used by backends to align all metrics on the same x-axis
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
        train_step: int = 0,
    ):
        """Init local (per-rank) logger backends and MetricCollector.

        Args:
            metadata_per_primary_backend: Metadata from primary backends for shared state.
            config: Backend configurations with logging modes and settings.
            train_step: Initial training step for metrics.
        """
        collector = MetricCollector()
        await collector.init_backends(
            metadata_per_primary_backend, config, train_step, actor_name=self.actor_name
        )

    @endpoint
    async def shutdown(self):

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

    def _validate_backend_config(
        self, backend_name: str, config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate and normalize backend configuration."""
        # Validate logging_mode is provided and valid
        if "logging_mode" not in config:
            raise ValueError(
                f"Backend '{backend_name}' missing required 'logging_mode' field"
            )

        mode_str = config["logging_mode"]
        mode = LoggingMode(mode_str)

        # Validate per_rank_share_run configuration
        share_run = config.get("per_rank_share_run", False)
        if mode == LoggingMode.GLOBAL_REDUCE and share_run:
            logger.warning(
                f"{backend_name}: per_rank_share_run ignored in {mode.value} mode."
            )

        return {
            **config,
            "logging_mode": mode,
        }

    @endpoint
    async def init_backends(self, config: Dict[str, Any]):
        """Sets config in global actor, initializes primary backends and eagerly initializes MetricCollectors
        in all registered fetchers.

        A backend is categorized by its logging_mode configuration:
        - GLOBAL_REDUCE: Backend instantiated only in the controller (this actor). Local ranks
          accumulate metrics and send states for global reduction. Final reduced metrics are logged
          only by the controller every train_step.
        - PER_RANK_REDUCE: Backend instantiated per-rank. Each rank accumulates metrics locally
          and logs aggregated values on flush(). No cross-rank reduction.
        - PER_RANK_NO_REDUCE: Backend instantiated per-rank. Each rank logs raw metric values
          immediately on each record_metric() call. Reduce type is ignored. Great alternative for
          analyzing metrics per time stamp instead of per train step.

        The backend instantiation is controlled by the logging_mode field. Primary backends
        (instantiated in the controller) can provide metadata to be shared with secondary backends on ranks,
        e.g. shared run IDs for WandB.

        Args:
            config (Dict[str, Any]): Config for metric logging where keys are backend names.
                Each backend must specify logging_mode field.
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

            # Extract metadata for shared modes
            if mode != LoggingMode.GLOBAL_REDUCE:
                primary_metadata = backend.get_metadata_for_secondary_ranks() or {}
                self.metadata_per_primary_backend[backend_name] = primary_metadata

            # Store global backends (only GLOBAL_REDUCE uses global logging)
            if mode == LoggingMode.GLOBAL_REDUCE:
                self.global_logger_backends[backend_name] = backend

        # Initialize local collectors
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
            step (int): Global step for logging.
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
                logger.warning(f"No states to reduce for step {step}")
                return

            # Reduce metrics from states
            reduced_metrics_dict = reduce_metrics_states(all_local_states)

            # Convert to Metric objects for backend logging
            reduced_metrics = []
            for key, value in reduced_metrics_dict.items():
                # Get reduction type from first state that has this key
                reduction_type = None
                for state in all_local_states:
                    if key in state and "reduction_type" in state[key]:
                        reduction_type = Reduce(state[key]["reduction_type"])
                        break

                if reduction_type is None:
                    reduction_type = Reduce.MEAN  # fallback

                metric = Metric(
                    key=key,
                    value=value,
                    reduction=reduction_type,
                )
                reduced_metrics.append(metric)

            # Log to global backends
            for backend_name, backend in self.global_logger_backends.items():
                await backend.log(reduced_metrics, step)

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
