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

from monarch.actor import current_rank

from forge.observability.utils import get_actor_name_with_rank

logger = logging.getLogger(__name__)


@dataclass
class Role:
    """Role identifier for metric logging actors.

    Defines whether an actor operates as a local (per-rank) or global (controller) role
    in the distributed metrics collection system.
    """

    LOCAL: str = "local"
    GLOBAL: str = "global"


class LoggingMode(Enum):
    """Metric logging mode.

    A backend is categorized by its logging_mode configuration:
    - GLOBAL_REDUCE: Backend is instantiated only in the controller (GlobalLoggingActor). Local ranks
        accumulate metrics and send states for global reduction. Final reduced metrics are logged
        only by the controller every step.

    - PER_RANK_REDUCE: Backend is instantiated per-rank. Each rank accumulates metrics locally
        and logs aggregated values on flush(). No cross-rank reduction.

    - PER_RANK_NO_REDUCE: Backend is instantiated per-rank. Each rank logs raw metric values
        immediately on each record_metric() call. Reduce type is **ignored**. Great alternative for
        analyzing metrics per time stamp instead of per step.
    """

    GLOBAL_REDUCE = "global_reduce"
    PER_RANK_REDUCE = "per_rank_reduce"
    PER_RANK_NO_REDUCE = "per_rank_no_reduce"


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
            # Always record in EST timezone
            est = pytz.timezone("US/Eastern")
            self.timestamp = datetime.now(est).timestamp()


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

    def __init__(self, reduction: Reduce):
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
    def __init__(self, reduction: Reduce):
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
    def __init__(self, reduction: Reduce):
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
    def __init__(self, reduction: Reduce):
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
    def __init__(self, reduction: Reduce):
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
    def __init__(self, reduction: Reduce):
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
    - Resets accumulators post-flush to avoid leaks across steps;
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

    def __init__(self):
        if hasattr(self, "_is_initialized"):
            return

        self.accumulators: Dict[str, MetricAccumulator] = {}
        self.rank = current_rank().rank
        self.per_rank_reduce_backends: List[LoggerBackend] = []
        self.per_rank_no_reduce_backends: List[LoggerBackend] = []
        self.step: int = 0  # Updated on flush
        self._is_initialized = False

    async def init_backends(
        self,
        metadata_per_primary_backend: Optional[Dict[str, Dict[str, Any]]],
        config: Dict[str, Any],
        step: int = 0,
        actor_name: str | None = None,
    ) -> None:
        """Initialize per-rank logger backends and MetricCollector state.

        A logger backend is represented by a backend class (e.g. WandBBackend, ConsoleBackend).
        Backends are categorized by their logging_mode. For details, see `forge.observability.metrics.LoggingMode`.

        Args:
            metadata_per_primary_backend (Optional[Dict[str, Dict[str, Any]]]): Metadata from primary
                logger backends for backends that require shared state, e.g.,
                {"wandb": {"shared_run_id": "abc123"}} for shared WandB runs across ranks.
            config (Dict[str, Any]): Backend configurations where each key is a backend name
                and value contains logging_mode and backend-specific settings.
                e.g., {"wandb": {"logging_mode": "per_rank_no_reduce", "project": "my_proj"}}
            step (int, default 0): Initial step for immediate logging. This allows
                restarting from checkpoints with correct step numbering.
            actor_name (str | None): The meaningful actor name for logging.
        """
        if self._is_initialized:
            logger.debug(f"Rank {self.rank}: MetricCollector already initialized")
            return

        # Initialize step tracking for immediate logging
        self.step = step

        self.per_rank_reduce_backends: List[LoggerBackend] = []
        self.per_rank_no_reduce_backends: List[LoggerBackend] = []

        # Initialize backends based on logging mode
        for backend_name, backend_config in config.items():
            mode = LoggingMode(backend_config["logging_mode"])

            # Skip local instantiation for GLOBAL_REDUCE
            # Backend will be instantiated in GlobalLoggingActor
            if mode == LoggingMode.GLOBAL_REDUCE:
                continue

            # Get primary metadata if needed
            primary_metadata = {}
            if metadata_per_primary_backend:
                primary_metadata = metadata_per_primary_backend.get(backend_name, {})

            # Instantiate backend
            backend = get_logger_backend_class(backend_name)(backend_config)
            await backend.init(
                role=Role.LOCAL,
                primary_logger_metadata=primary_metadata,
                actor_name=actor_name,
            )

            # Categorize by logging mode
            if mode == LoggingMode.PER_RANK_NO_REDUCE:
                self.per_rank_no_reduce_backends.append(backend)
            else:
                self.per_rank_reduce_backends.append(backend)

        self._is_initialized = True

    def push(self, metric: Metric) -> None:
        """Immediately log metrics to backends marked as "no_reduce" and adds metrics to accumulators for reduction
        and later logging."""
        if not self._is_initialized:
            raise ValueError(
                "MetricCollector was not initialized. This happens when you try to use `record_metric` "
                "before you have initialized any logging backends. Please call in your main file:\n"
                "`mlogger = await get_or_create_metric_logger(actor_name='Controller')`\n"
                "`await mlogger.init_backends.call_one(logging_config)`\n"
                "or, to disable metric logging globally, set env variable `FORGE_DISABLE_METRICS=True`"
            )

        # Validate metric object
        if not isinstance(metric, Metric):
            raise TypeError(f"Expected {Metric} object, got {metric}")

        # Always accumulate for deferred logging and state return
        key = metric.key
        if key not in self.accumulators:
            self.accumulators[key] = metric.reduction.accumulator_class(
                metric.reduction
            )
        self.accumulators[key].append(metric.value)

        # For PER_RANK_NO_REDUCE backends: log immediately
        for backend in self.per_rank_no_reduce_backends:
            backend.log_immediate(metric=metric, step=self.step)

    async def flush(
        self, step: int, return_state: bool = False
    ) -> Dict[str, Dict[str, Any]]:
        """Log to local logger backends (if any), reset accumulators and return metric states dict if return_state=True.

        Args:
            step (int): step used by backends to align metrics on the same x-axis
            return_state (bool): Used by GlobalLoggingActor for reduction across all ranks.
                If False, returns empty dict, else returns the state of all metrics collected.
        Returns:
            Dict[str, Dict[str, Dict[str, Any]]]: Dict of {metric_key: metric_state},
                e.g., {"loss": {"reduction_type": "mean", "sum": 1.2, "count": 3}}.
        """

        if not self._is_initialized:
            logger.debug(
                f"Collector not yet initialized for {get_actor_name_with_rank()}. Call init_backends first."
            )
            return {}

        if not self.accumulators:
            logger.debug(
                f"Collector rank {get_actor_name_with_rank()}: No metrics to flush for step {step}"
            )
            return {}

        # Snapshot states and reset immediately
        states = {}
        for key, acc in self.accumulators.items():
            states[key] = acc.get_state()
            acc.reset()

        # Update step (used by NO_REDUCE backends in push)
        self.step = step

        # Log to PER_RANK_REDUCE backends only (NO_REDUCE already logged in push)
        if self.per_rank_reduce_backends:
            metrics_for_backends = reduce_metrics_states([states])

            # Log to PER_RANK_REDUCE backends
            for backend in self.per_rank_reduce_backends:
                await backend.log(metrics_for_backends, step)

        return states if return_state else {}

    async def shutdown(self):
        """Shutdown logger_backends if initialized."""

        if not self._is_initialized:
            logger.debug(
                f"Collector for {get_actor_name_with_rank()} not initialized. Skipping shutdown"
            )
            return

        for backend in self.per_rank_reduce_backends + self.per_rank_no_reduce_backends:
            await backend.finish()


###########
# Backends #
###########


class LoggerBackend(ABC):
    """Abstract logger_backend for metric logging, e.g. wandb, jsonl, etc."""

    def __init__(self, logger_backend_config: Dict[str, Any]):
        self.logger_backend_config = logger_backend_config

    @abstractmethod
    async def init(
        self,
        role: str,
        primary_logger_metadata: Optional[Dict[str, Any]] = None,
        actor_name: str | None = None,
    ) -> None:
        """
        Initializes backend, e.g. wandb.run.init().

        Args:
            role (str): "global" (controller/primary) or "local" (per-rank/secondary).
                Can be used to behave differently for primary vs secondary roles.
            primary_logger_metadata (Optional[Dict[str, Any]]): From global backend for
                backend that required shared info, e.g. {"shared_run_id": "abc123"}.

        Raises: ValueError if missing metadata for shared local init.
        """
        if primary_logger_metadata is None:
            primary_logger_metadata = {}
        pass

    async def log(self, metrics: List[Metric], step: int, *args, **kwargs) -> None:
        """Log list of metrics to backend. Meant to log in bulk, e.g. on flush."""
        pass

    def log_immediate(self, metric: Metric, step: int, *args, **kwargs) -> None:
        """Log single metric to backend. Meant to log metric as soon as collected.
        Backend implementation can decide to buffer/flush as needed."""
        pass

    async def finish(self) -> None:
        pass

    def get_metadata_for_secondary_ranks(self) -> Optional[Dict[str, Any]]:
        """Return sharable state after primary init (e.g., for shared modes). Called only on globals."""
        return None


class ConsoleBackend(LoggerBackend):
    """Simple console logging of metrics."""

    def __init__(self, logger_backend_config: Dict[str, Any]):
        super().__init__(logger_backend_config)

    async def init(
        self,
        role: str,
        primary_logger_metadata: Optional[Dict[str, Any]] = None,
        actor_name: str | None = None,
    ) -> None:
        pass

    async def log(self, metrics: List[Metric], step: int, *args, **kwargs) -> None:
        metrics_str = "\n".join(f"  {metric.key}: {metric.value}" for metric in metrics)
        logger.info(
            f"=== [METRICS STEP {step} ===\n{metrics_str}\n==============================\n"
        )

    def log_immediate(self, metric: Metric, step: int, *args, **kwargs) -> None:
        """Log metric immediately to console with timestamp."""
        logger.info(f"{metric.key}: {metric.value}")

    async def finish(self) -> None:
        pass


class WandbBackend(LoggerBackend):
    """
    Weights & Biases logging backend.

    For logging mode details, see `forge.observability.metrics.LoggingMode` documentation.

    More details on wandb distributed logging here: https://docs.wandb.ai/guides/track/log/distributed-training/

    Configuration:
        logging_mode (LoggingMode): Determines logging behavior
        per_rank_share_run (bool, default False): For per-rank modes, whether to share run ID across ranks.
            If true, then a single wandb is created and all ranks log to it. Its particularly useful if
            logging with no_reduce to capture a time based stream of information. Not recommended if reducing values.
        project (str): WandB project name
        group (str, optional): WandB group name for organizing runs. Defaults to "experiment_group"
    """

    def __init__(self, logger_backend_config: Dict[str, Any]):
        super().__init__(logger_backend_config)
        self.project = logger_backend_config["project"]
        self.group = logger_backend_config.get("group", "experiment_group")
        self.name = None
        self.run = None
        self.logging_mode = LoggingMode(logger_backend_config["logging_mode"])
        self.per_rank_share_run = logger_backend_config.get("per_rank_share_run", False)

    async def init(
        self,
        role: str,
        primary_logger_metadata: Optional[Dict[str, Any]] = None,
        actor_name: str | None = None,
    ) -> None:

        if primary_logger_metadata is None:
            primary_logger_metadata = {}

        if role not in [Role.GLOBAL, Role.LOCAL]:
            raise ValueError(
                f"Invalid role {role} for WandbBackend init. Must be '{Role.GLOBAL}' or '{Role.LOCAL}'."
            )

        self.name = (
            get_actor_name_with_rank(actor_name)
            if role == Role.LOCAL
            else "global_controller"
        )

        # GLOBAL_REDUCE mode: only inits on controller
        if self.logging_mode == LoggingMode.GLOBAL_REDUCE:
            if role != Role.GLOBAL:
                logger.warning(f"Skipped init for GLOBAL_REDUCE mode and {role} role.")
                return
            await self._init_global()

        # Per-rank modes based on per_rank_share_run bool
        elif role == Role.GLOBAL and self.per_rank_share_run:
            await self._init_shared_global()

        elif role == Role.LOCAL:
            if self.per_rank_share_run:
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
        settings = wandb.Settings(mode="shared", x_primary=False, x_label=self.name)
        self.run = wandb.init(
            id=shared_id,
            project=self.project,
            group=self.group,
            settings=settings,
        )

    async def log(self, metrics: List[Metric], step: int, *args, **kwargs) -> None:
        if not self.run:
            logger.debug(f"WandbBackend: No run started, skipping log for {self.name}")
            return

        # Convert metrics to WandB log format
        log_data = {"step": step}
        for metric in metrics:
            log_data[metric.key] = metric.value

        self.run.log(log_data)
        logger.info(f"WandbBackend: Logged {len(metrics)} metrics at step {step}")

    def log_immediate(self, metric: Metric, step: int, *args, **kwargs) -> None:
        """Log metric immediately to WandB with both step and timestamp."""
        if not self.run:
            return

        # Log with both step and timestamp - users can choose x-axis in WandB UI
        log_data = {
            metric.key: metric.value,
            "step": step,
            "_timestamp": metric.timestamp,
        }
        self.run.log(log_data)

    def get_metadata_for_secondary_ranks(self) -> Dict[str, Any]:
        if self.run and self.per_rank_share_run:
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
