# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

import pytz
from monarch.actor import context, current_rank

from forge.util.logging import log_once

logger = logging.getLogger(__name__)


class BackendRole(Enum):
    """Backend role constants for metric logging actors.

    Defines whether an actor operates as a local (per-rank) or global (controller) role
    in the distributed metrics collection system.
    """

    LOCAL = "local"
    GLOBAL = "global"


class Reduce(Enum):
    MEAN = "mean"
    SUM = "sum"
    MAX = "max"
    MIN = "min"
    STD = "std"

    @property
    def accumulator_class(self):
        mapping = {
            Reduce.MEAN: MeanAccumulator,
            Reduce.SUM: SumAccumulator,
            Reduce.MAX: MaxAccumulator,
            Reduce.MIN: MinAccumulator,
            Reduce.STD: StdAccumulator,
        }
        return mapping[self]


@dataclass
class Metric:
    """Container for metric data including key, value, reduction type, and timestamp.

    Timestamp is automatically set to current EST time if not provided.
    """

    key: str
    value: Any
    reduction: Reduce
    timestamp: Optional[float] = None

    def __post_init__(self):
        if self.timestamp is None:
            # Always record in UTC timezone
            self.timestamp = datetime.now(pytz.UTC).timestamp()


def get_actor_name_with_rank() -> str:
    """
    Extracts actor information from Monarch context to form a logging name.

    Returns:
        str: Format "ActorName_replicaId_rLocalRank" (e.g., "TrainActor_abcd_r0").
             Falls back to "UnknownActor" if context unavailable.
    """
    # Add more defensive checks
    ctx = context()
    if ctx is None or ctx.actor_instance is None:
        logger.warning("Context unavailable, using fallback actor name for logging.")
        return "UnknownActor"

    actor_instance = ctx.actor_instance
    rank = current_rank()

    actor_id_full = str(actor_instance.actor_id)

    # Parse the actor_id
    parts = actor_id_full.split(".")
    rank_name = "UnknownActor"  # fallback
    if len(parts) >= 2:
        world_part = parts[0]  # e.g., "_1rjutFUXQrEJ[0]"
        actor_part = parts[1]  # e.g., "TestActorConfigured[0]"

        # Extract world ID and proc rank
        world_id = world_part.split("[")[0] if "[" in world_part else world_part

        # Extract clean actor name (remove "Configured" suffix if present)
        if "[" in actor_part:
            actor_name = actor_part.split("[")[0]  # e.g., "TestActorConfigured"
            if actor_name.endswith("Configured"):
                actor_name = actor_name[:-10]  # Remove "Configured"
        else:
            actor_name = actor_part

        # Use last 4 characters of world_id as replica identifier
        # This is deterministic, readable, and works for any number of replicas
        replica_id = world_id[-4:] if len(world_id) >= 4 else world_id

        # Use current_rank().rank as the local rank within the replica
        local_rank = rank.rank

        rank_name = f"{actor_name}_{replica_id}_r{local_rank}"

    return rank_name


def record_metric(key: str, value: Any, reduction: Reduce = Reduce.MEAN) -> None:
    """Thin wrapper to send metrics to per-rank local MetricCollectors.

    Relies on a per-rank MetricCollector singleton for ease of use, i.e.
    call `record_metric` anywhere in the code without moving the
    collector from function to function.

    The collector methods are triggered per-rank by a
    `forge.observability.metric_actors.LocalFetcherActor`, instantiated
    during actor initialization.

    Records are flushed when `forge.observability.metric_actors.GlobalLoggingActor.flush()`
    is called, typically triggered by the training loop at regular intervals.

    Can be disabled globally by setting the environment variable `FORGE_DISABLE_METRICS=true`.
    """
    # Skip metrics collection
    if os.getenv("FORGE_DISABLE_METRICS", "false").lower() == "true":
        return

    # timestamp is added automatically by the Metric class
    metric = Metric(key=key, value=value, reduction=reduction)
    collector = MetricCollector()
    collector.push(metric)


def reduce_metrics_states(states: List[Dict[str, Dict[str, Any]]]) -> List[Metric]:
    """Reduce metric accumulators states to a list of metrics.

    Can be used when reducing metrics across ranks or services, as merging
    states is more precise than merging locally reduced metrics.

    Args:
        states (List[Dict[str, Dict[str, Any]]]): List of state of one or more metrics,
            normally retrieved using `forge.observability.metrics.MetricAccumulator.get_state()`.

    Returns:
        List[Metric]: List of reduced metrics

    Example:
        states = [
            {"loss": {"count": 5, "sum": 14, "reduction_type": Reduce.MEAN}},
            {"loss": {"count": 10, "sum": 16, "reduction_type": Reduce.MEAN}},
        ]
        reduce_metrics_states(states)
        >>> [Metric(key="loss", value=2.0, reduction=Reduce.MEAN)]

    Raises:
        ValueError: on mismatched reduction types for the same metric key.
    """
    if not states:
        return []

    # Collect unique keys across all
    all_keys = set(k for state in states for k in state)

    reduced_metrics = []
    for key in all_keys:
        metric_states = [state.get(key) for state in states if key in state]
        if not metric_states:
            continue

        first_reduction_type = metric_states[0]["reduction_type"]  # pyre-ignore

        # Check consistency
        for state in metric_states:
            if state is None:
                continue
            if state["reduction_type"] != first_reduction_type:
                raise ValueError(
                    f"Mismatched reduction types for key '{key}': {first_reduction_type} vs {state['reduction_type']}"
                )

        metric_accumulator = Reduce(first_reduction_type).accumulator_class
        reduced_value = metric_accumulator.get_reduced_value_from_states(metric_states)

        # Create Metric object with reduced value
        metric = Metric(
            key=key,
            value=reduced_value,
            reduction=Reduce(first_reduction_type),
        )
        reduced_metrics.append(metric)

    return reduced_metrics


################
# Accumulators #
################


class MetricAccumulator(ABC):
    """Every metric maps to a MetricAccumulator, which accumulates values and optionally reduces them."""

    def __init__(self, reduction: Reduce) -> None:
        self.reduction_type = reduction

    @abstractmethod
    def append(self, value: Any) -> None:
        """Updates accumulator with new value (e.g., adds to sum and count for MEAN)."""
        pass

    @abstractmethod
    def get_value(self) -> Any:
        """Returns locally reduced value (e.g., sum/count for MEAN)."""
        pass

    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        """Returns serializable state for cross-rank merge (e.g., {'sum': 10.0, 'count': 5})."""
        pass

    @classmethod
    @abstractmethod
    def get_reduced_value_from_states(cls, states: List[Dict[str, Any]]) -> Any:
        """Merges states from multiple ranks into single reduced value (e.g., total_sum/total_count for MEAN)."""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Clears for next accumulation cycle (e.g., sum=0, count=0 for MEAN)."""
        pass


class MeanAccumulator(MetricAccumulator):
    def __init__(self, reduction: Reduce) -> None:
        super().__init__(reduction)
        self.sum = 0.0
        self.count = 0

    def append(self, value: Any) -> None:
        v = float(value.item() if hasattr(value, "item") else value)
        self.sum += v
        self.count += 1

    def get_value(self) -> float:
        return self.sum / self.count if self.count > 0 else 0.0

    def get_state(self) -> Dict[str, Any]:
        return {
            "reduction_type": self.reduction_type.value,
            "sum": self.sum,
            "count": self.count,
        }

    @classmethod
    def get_reduced_value_from_states(cls, states: List[Dict[str, Any]]) -> float:
        total_sum = sum(s["sum"] for s in states)
        total_count = sum(s["count"] for s in states)
        return total_sum / total_count if total_count > 0 else 0.0

    def reset(self) -> None:
        self.sum = 0.0
        self.count = 0


class SumAccumulator(MetricAccumulator):
    def __init__(self, reduction: Reduce) -> None:
        super().__init__(reduction)
        self.total = 0.0

    def append(self, value: Any) -> None:
        v = float(value.item() if hasattr(value, "item") else value)
        self.total += v

    def get_value(self) -> float:
        return self.total

    def get_state(self) -> Dict[str, Any]:
        return {"reduction_type": self.reduction_type.value, "total": self.total}

    @classmethod
    def get_reduced_value_from_states(cls, states: List[Dict[str, Any]]) -> float:
        return sum(s["total"] for s in states)

    def reset(self) -> None:
        self.total = 0.0


class MaxAccumulator(MetricAccumulator):
    def __init__(self, reduction: Reduce) -> None:
        super().__init__(reduction)
        self.max_val = float("-inf")

    def append(self, value: Any) -> None:
        v = float(value.item() if hasattr(value, "item") else value)
        self.max_val = max(self.max_val, v)

    def get_value(self) -> float:
        return self.max_val

    def get_state(self) -> Dict[str, Any]:
        return {"reduction_type": self.reduction_type.value, "max_val": self.max_val}

    @classmethod
    def get_reduced_value_from_states(cls, states: List[Dict[str, Any]]) -> float:
        return max(s["max_val"] for s in states)

    def reset(self) -> None:
        self.max_val = float("-inf")


class MinAccumulator(MetricAccumulator):
    def __init__(self, reduction: Reduce) -> None:
        super().__init__(reduction)
        self.min_val = float("inf")

    def append(self, value: Any) -> None:
        v = float(value.item() if hasattr(value, "item") else value)
        self.min_val = min(self.min_val, v)

    def get_value(self) -> float:
        return self.min_val

    def get_state(self) -> Dict[str, Any]:
        return {"reduction_type": self.reduction_type.value, "min_val": self.min_val}

    @classmethod
    def get_reduced_value_from_states(cls, states: List[Dict[str, Any]]) -> float:
        return min(s["min_val"] for s in states)

    def reset(self) -> None:
        self.min_val = float("inf")


class StdAccumulator(MetricAccumulator):
    def __init__(self, reduction: Reduce) -> None:
        super().__init__(reduction)
        self.sum = 0.0
        self.sum_sq = 0.0
        self.count = 0

    def append(self, value: Any) -> None:
        v = float(value.item() if hasattr(value, "item") else value)
        self.sum += v
        self.sum_sq += v * v
        self.count += 1

    def get_value(self) -> float:
        if self.count == 0:
            return 0.0
        if self.count == 1:
            return 0.0
        mean = self.sum / self.count
        variance = (self.sum_sq / self.count) - (mean * mean)
        return max(0.0, variance) ** 0.5

    def get_state(self) -> Dict[str, Any]:
        return {
            "reduction_type": self.reduction_type.value,
            "sum": self.sum,
            "sum_sq": self.sum_sq,
            "count": self.count,
        }

    @classmethod
    def get_reduced_value_from_states(cls, states: List[Dict[str, Any]]) -> float:
        total_sum = sum(s["sum"] for s in states)
        total_sum_sq = sum(s["sum_sq"] for s in states)
        total_count = sum(s["count"] for s in states)
        if total_count == 0:
            return 0.0
        if total_count == 1:
            return 0.0
        mean = total_sum / total_count
        variance = (total_sum_sq / total_count) - (mean * mean)
        return max(0.0, variance) ** 0.5

    def reset(self) -> None:
        self.sum = 0.0
        self.sum_sq = 0.0
        self.count = 0


#############
# Collector #
#############


class MetricCollector:
    """Per-rank singleton for accumulating, retrieving and flushing metrics to backends.

    A logger is represented by a backend, i.e. wandb backend. If reduce_across_ranks=False,
    the backend is instantiated per-rank, in the MetricCollector, otherwise it is instantiated once globally,
    in the GlobalLoggingActor.

    - Ensures one instance per process; actors call record_metric() which delegates here.
    - Init via GlobalLoggingActor -> LocalFetcherActor -> per-rank MetricCollector;
    - GlobalLoggingActor flushes trigger reductions and log for any locally setup backend. Can optionally also
    return non-reduced states for global aggregation. This can be different for each backend.
    - Resets accumulators post-flush to avoid leaks across train steps;
    """

    _instances: Dict[int, "MetricCollector"] = {}
    _singleton_rank: int

    def __new__(cls):
        """Singleton per-rank, ensures one instance per process."""
        rank = current_rank().rank

        if rank not in cls._instances:
            inst = super().__new__(cls)
            cls._instances[rank] = inst
            inst._singleton_rank = rank
        else:
            inst = cls._instances[rank]
            if inst._singleton_rank != rank:
                raise ValueError(
                    f"Singleton expected rank {inst._singleton_rank}, but saw {rank}"
                )
        return inst

    def __init__(self) -> None:
        if hasattr(self, "_is_initialized"):
            return

        self.accumulators: Dict[str, MetricAccumulator] = {}
        self.rank = current_rank().rank
        self.logger_backends: List[LoggerBackend] = []
        self._is_initialized = False

    async def init_backends(
        self,
        metadata_per_primary_backend: Optional[Dict[str, Dict[str, Any]]],
        config: Dict[str, Any],
    ) -> None:
        """A logger is represented by a backend, i.e. wandb backend. If reduce_across_ranks=False,
        the backend is instantiated per-rank, in the MetricCollector, otherwise it is only instantiated
        once globally.

        Args:
            metadata_per_primary_backend (Optional[Dict[str, Dict[str, Any]]]): Metadata from primary
                logger backend, e.g., {"wandb": {"run_id": "abc123"}}.
            config (Dict[str, Any]): Logger backend configuration, e.g. {"wandb": {"project": "my_project"}}.
        """
        if self._is_initialized:
            logger.debug(f"Rank {self.rank}: MetricCollector already initialized")
            return

        # instantiate local backends if any
        for backend_name, backend_config in config.items():
            if backend_config.get("reduce_across_ranks", True):
                continue  # Skip local backend instantiation and use global instead

            # get metadata from primary backend if any
            primary_metadata = {}
            if metadata_per_primary_backend:
                primary_metadata = metadata_per_primary_backend.get(backend_name, {})

            # instantiate local backend
            logger_backend = get_logger_backend_class(backend_name)(backend_config)
            await logger_backend.init(
                role=BackendRole.LOCAL, primary_logger_metadata=primary_metadata
            )
            self.logger_backends.append(logger_backend)

        self._is_initialized = True

    def push(self, metric: Metric) -> None:
        """Process a metric according to configured logging modes.

        Args:
            metric: Metric dataclass containing key, value, reduction type, and timestamp.

        Raises:
            TypeError: If metric is not a Metric object.

        Example:
            collector = MetricCollector()
            metric = Metric("loss", 0.5, Reduce.MEAN)
            collector.push(metric)
        """
        if not self._is_initialized:
            log_once(
                logger,
                level=logging.WARNING,
                msg=(
                    "Skipping metric collection. Metric logging backends (e.g. wandb) were not initialized."
                    " This happens when you try to use `record_metric` before calling `init_backends`."
                    " To disable this warning, please call in your main file:\n"
                    "`mlogger = await get_or_create_metric_logger()`\n"
                    "`await mlogger.init_backends.call_one(logging_config)`\n"
                    "or set env variable `FORGE_DISABLE_METRICS=True`"
                ),
            )
            return

        # Validate metric object
        if not isinstance(metric, Metric):
            raise TypeError(f"Expected {Metric} object, got {type(metric)}")

        # Always accumulate for reduction and state return
        key = metric.key
        if key not in self.accumulators:
            self.accumulators[key] = metric.reduction.accumulator_class(
                metric.reduction
            )
        self.accumulators[key].append(metric.value)

    async def flush(
        self, global_step: int, return_state: bool = False
    ) -> Dict[str, Dict[str, Any]]:
        """Log to local logger backends (if any), reset accumulators and return metric states dict if return_state=True.

        Args:
            global_step (int): step used by backends to align metrics on the same x-axis
            return_state (bool): Used by GlobalLoggingActor for reduction across all ranks.
                If False, returns empty dict, else returns the state of all metrics collected.
        Returns:
            Dict[str, Dict[str, Dict[str, Any]]]: Dict of {metric_key: metric_state},
                e.g., {"loss": {"reduction_type": "mean", "sum": 1.2, "count": 3}}.
        """
        if not self._is_initialized:
            log_once(
                logger,
                level=logging.WARNING,
                msg="Cannot flush collected metrics. MetricCollector.flush() called before init_backends()."
                "\nPlease call in your main file:\n"
                "`mlogger = await get_or_create_metric_logger()`\n"
                "`await mlogger.init_backends.call_one(logging_config)`\n"
                "before calling `flush`",
            )
            return {}

        if not self.accumulators:
            logger.debug(
                f"Collector rank {get_actor_name_with_rank()}: No metrics to flush for global_step {global_step}"
            )
            return {}

        # Snapshot states and reset immediately
        states = {}
        for key, acc in self.accumulators.items():
            states[key] = acc.get_state()
            acc.reset()

        # Reduce metrics from states for logging if any per-rank backend
        if self.logger_backends:
            # Use reduce_metrics_states for consistency
            reduced_metrics = reduce_metrics_states([states])

            # Log to local logger_backends
            for logger_backend in self.logger_backends:
                await logger_backend.log(reduced_metrics, global_step)

        return states if return_state else {}

    async def shutdown(self):
        """Shutdown logger_backends if initialized."""
        if not self._is_initialized:
            logger.debug(
                f"Collector for {get_actor_name_with_rank()} not initialized. Skipping shutdown"
            )
            return

        for logger_backend in self.logger_backends:
            await logger_backend.finish()


###########
# Backends #
###########


class LoggerBackend(ABC):
    """Abstract logger_backend for metric logging, e.g. wandb, jsonl, etc."""

    def __init__(self, logger_backend_config: Dict[str, Any]) -> None:
        self.logger_backend_config = logger_backend_config

    @abstractmethod
    async def init(
        self,
        role: BackendRole,
        primary_logger_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initializes backend, e.g. wandb.run.init().

        Args:
            role (BackendRole): BackendRole.GLOBAL (controller/primary) or BackendRole.LOCAL (per-rank/secondary).
                Can be used to behave differently for primary vs secondary roles.
            primary_logger_metadata (Optional[Dict[str, Any]]): From global backend for
                backend that required shared info, e.g. {"shared_run_id": "abc123"}.

        Raises: ValueError if missing metadata for shared local init.
        """
        pass

    @abstractmethod
    async def log(self, metrics: List[Metric], global_step: int) -> None:
        """
        Log a batch of metrics to the backend.

        Args:
            metrics: List of Metric objects to log.
            global_step: Step number for x-axis alignment across metrics.
        """
        pass

    async def finish(self) -> None:
        pass

    def get_metadata_for_secondary_ranks(self) -> Optional[Dict[str, Any]]:
        """Return sharable state after primary init (e.g., for shared modes). Called only on globals."""
        return None


class ConsoleBackend(LoggerBackend):
    """Simple console logging of metrics."""

    def __init__(self, logger_backend_config: Dict[str, Any]) -> None:
        super().__init__(logger_backend_config)

    async def init(
        self,
        role: BackendRole,
        primary_logger_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.prefix = (
            get_actor_name_with_rank()
            if self.logger_backend_config.get("reduce_across_ranks", True)
            else "Controller"
        )

    async def log(self, metrics: List[Metric], global_step: int) -> None:
        metrics_str = "\n".join(
            f"  {metric.key}: {metric.value}"
            for metric in sorted(metrics, key=lambda m: m.key)
        )
        logger.info(
            f"=== [{self.prefix}] - METRICS STEP {global_step} ===\n{metrics_str}\n==============================\n"
        )

    async def finish(self) -> None:
        pass


class WandbBackend(LoggerBackend):
    """
    Weights & Biases logging backend for distributed training.

    Supports 3 types of modes as described in https://docs.wandb.ai/guides/track/log/distributed-training/:
    Track a single process: reduce_across_ranks=True
    Track each process separately: reduce_across_ranks=False, share_run_id=False
    Track all processes to a single run: reduce_across_ranks=False, share_run_id=True

    Configuration:
        reduce_across_ranks (bool, default True): If True, log reduced metrics only from controller (global mode).
            If False, enables per-rank logging; then use share_run_id to pick mode.
        share_run_id (bool, default False): Only used if reduce_across_ranks=False.
            True -> shared run across ranks; False -> separate runs per rank.
        project (str): WandB project name
        group (str, optional): WandB group name for organizing runs. Defaults to "experiment_group"
    """

    def __init__(self, logger_backend_config: Dict[str, Any]) -> None:
        super().__init__(logger_backend_config)
        self.project = logger_backend_config["project"]
        self.group = logger_backend_config.get("group", "experiment_group")
        self.name = None
        self.run = None
        self.reduce_across_ranks = logger_backend_config.get(
            "reduce_across_ranks", True
        )
        self.share_run_id = logger_backend_config.get("share_run_id", False)

    async def init(
        self,
        role: BackendRole,
        primary_logger_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:

        if primary_logger_metadata is None:
            primary_logger_metadata = {}

        self.name = (
            get_actor_name_with_rank()
            if role == BackendRole.LOCAL
            else "global_controller"
        )

        # Default global mode: only inits on controller
        if self.reduce_across_ranks:
            if role != BackendRole.GLOBAL:
                logger.debug(
                    f"Skipped init for global mode (reduce_across_ranks=True) and {role} role."
                )
                return
            await self._init_global()

        # Per-rank modes based on share_run_id bool
        elif role == BackendRole.GLOBAL and self.share_run_id:
            await self._init_shared_global()

        elif role == BackendRole.LOCAL:
            if self.share_run_id:
                await self._init_shared_local(primary_logger_metadata)
            else:
                await self._init_per_rank()

    async def _init_global(self):
        import wandb

        self.run = wandb.init(project=self.project, group=self.group)

    async def _init_per_rank(self):
        import wandb

        self.run = wandb.init(project=self.project, group=self.group, name=self.name)

    async def _init_shared_global(self):
        import wandb

        settings = wandb.Settings(
            mode="shared", x_primary=True, x_label="controller_primary"
        )
        self.run = wandb.init(project=self.project, group=self.group, settings=settings)

    async def _init_shared_local(self, primary_metadata: Dict[str, Any]):
        import wandb

        shared_id = primary_metadata.get("shared_run_id")
        if shared_id is None:
            raise ValueError(
                f"Shared ID required but not provided for {self.name} backend init"
            )

        # Clear any stale service tokens that might be pointing to dead processes
        # In multiprocessing environments, WandB service tokens can become stale and point
        # to dead service processes. This causes wandb.init() to hang indefinitely trying
        # to connect to non-existent services. Clearing forces fresh service connection.
        from wandb.sdk.lib.service import service_token

        service_token.clear_service_in_env()

        settings = wandb.Settings(mode="shared", x_primary=False, x_label=self.name)
        self.run = wandb.init(
            id=shared_id,
            project=self.project,
            group=self.group,
            settings=settings,
        )

    async def log(self, metrics: List[Metric], global_step: int) -> None:
        if self.run:
            # Convert metrics to WandB log format
            log_data = {"global_step": global_step}
            for metric in metrics:
                log_data[metric.key] = metric.value

            self.run.log(log_data)
            logger.info(
                f"WandbBackend: Logged {len(metrics)} metrics at global_step {global_step}"
            )
        else:
            logger.debug(f"WandbBackend: No run started, skipping log for {self.name}")

    def get_metadata_for_secondary_ranks(self) -> Dict[str, Any]:
        if self.run and not self.reduce_across_ranks and self.share_run_id:
            return {"shared_run_id": self.run.id}
        return {}

    async def finish(self) -> None:
        if self.run:
            self.run.finish()
            logger.info(f"WandbBackend {self.name}: Finished run")


def get_logger_backend_class(cls_name: str) -> type[LoggerBackend]:
    """Simple mapping between logger_backend type and its class

    Factory for backend classes from config; returns uninitialized class for role-based init.
    """
    if cls_name == "console":
        return ConsoleBackend
    elif cls_name == "wandb":
        return WandbBackend
    else:
        raise ValueError(f"Unknown logger backend type: {cls_name}")
