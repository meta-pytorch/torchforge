# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional

from monarch.actor import context, current_rank


logger = logging.getLogger(__name__)


# Reduction Types
class ReductionType(Enum):
    MEAN = "mean"
    SUM = "sum"
    MAX = "max"
    MIN = "min"
    STD = "std"

    @property
    def accumulator_class(self):
        mapping = {
            ReductionType.MEAN: MeanAccumulator,
            ReductionType.SUM: SumAccumulator,
            ReductionType.MAX: MaxAccumulator,
            ReductionType.MIN: MinAccumulator,
            ReductionType.STD: StdAccumulator,
        }
        return mapping[self]


def get_actor_name_with_rank() -> str:
    """
    Extracts actor information from Monarch context to form a logging name.

    Returns a string like "TrainActor_abcd_r0" (actor type + replica ID suffix + local rank).
    Relies on parsing actor_id string; fallback to "UnknownActor" if context unavailable.

    # TODO: Replace string parsing with structured actor_id access once Monarch exposes it.
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


async def record_metric(
    key: str, value: Any, reduction: ReductionType = ReductionType.MEAN
) -> None:
    """
    Records a metric value for later reduction and logging.

    Relies on a per-rank MetricCollector singleton for ease of use, i.e.
    call `record_metric` anywhere in the code without moving the
    collector from function to function.

    The collector methods are triggered per-rank by a
    `forge.observability.metric_actors.LocalFetcherActor`, instantiated
    during actor initialization.

    Records are flushed after every N train steps, triggered by
    `forge.observability.metric_actors.GlobalLoggingActor`
    """
    collector = MetricCollector()
    await collector.push(key, value, reduction)


def reduce_metrics_states(states: List[Dict[str, Dict[str, Any]]]) -> Dict[str, Any]:
    """Reduce metric accumulators states to a single value per metric.

    Can be used when reducing metrics across ranks or services, as merging
    states is more precise than merging locally reduced metrics.

    Args:
        states (List[Dict[str, Dict[str, Any]]]): List of state of one or more metrics,
            normally retrieved using `forge.observability.metrics.MetricAccumulator.get_state()`.

    Returns:
        Dict[str, Any]: Dictionary with format {metric_key: reduced_value}

    Example:
        states = [
            {"loss": {"count": 5, "sum": 14, "reduction_type": ReductionType.MEAN}},
            {"loss": {"count": 10, "sum": 16, "reduction_type": ReductionType.MEAN}},
        ]
        reduce_metrics_states(states)
        >>> {"loss": 2.0}

    Raises:
        ValueError: on mismatched reduction types for the same metric key.
    """
    if not states:
        return {}

    # Collect unique keys across all
    all_keys = set(k for state in states for k in state)

    reduced_metrics = {}
    for key in all_keys:
        metric_states = [state.get(key) for state in states if key in state]
        if not metric_states:
            continue

        first_reduction_type = metric_states[0]["reduction_type"]

        # Check consistency
        for state in metric_states:
            if state["reduction_type"] != first_reduction_type:
                raise ValueError(
                    f"Mismatched reduction types for key '{key}': {first_reduction_type} vs {state['reduction_type']}"
                )

        metric_accumulator = ReductionType(first_reduction_type).accumulator_class
        reduced_value = metric_accumulator.get_reduced_value_from_states(metric_states)
        reduced_metrics[key] = reduced_value

    return reduced_metrics


class MetricAccumulator(ABC):
    # TODO: add docstring for every method, explaining when/why this is used
    def __init__(self, reduction: ReductionType):
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
    def __init__(self, reduction: ReductionType):
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
    def __init__(self, reduction: ReductionType):
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
    def __init__(self, reduction: ReductionType):
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
    def __init__(self, reduction: ReductionType):
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


class MetricCollector:
    """
    Per-rank singleton for accumulating, retrieving or flushing metrics to backends.

    - Ensures one instance per process (rank-enforced); actors call record_metric() which delegates here.
    - Init via GlobalLoggingActor -> LocalFetcherActor -> per-rank MetricCollector;
    - GlobalLoggingActor flushes trigger reductions and log for any locally setup backend. Can optionally also
    return non-reduced states for global aggregation.
    - Resets accumulators post-flush to avoid leaks across train steps;
    """

    _instances: Dict[int, "MetricCollector"] = {}

    def __new__(cls):
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
        if hasattr(self, "_initialized_sync"):
            return
        self._initialized_sync = True
        self.accumulators: Dict[str, MetricAccumulator] = {}
        self.logger_backends: List[LoggerBackend] = []
        self._initialized_async = False
        self.rank = current_rank().rank

    async def init_local_backends(
        self,
        metadata_per_primary_backend: Dict[str, Dict[str, Any]],
        global_logger=None,
    ) -> None:
        """Initializes collector with logger_backends from global config.

        Queries global logger for config; sets up local logger_backends only if log_per_rank=True.
        Called once per-rank by LocalFetcherActor.
        """
        if self._initialized_async:
            logger.debug(f"Rank {self.rank}: MetricCollector already initialized")
            return

        # Allow passing global_logger directly to avoid controller context issues
        if global_logger is None:
            logger.debug(
                f"Rank {self.rank}: No global_logger provided, spawning controller"
            )
            from monarch.actor import get_or_spawn_controller

            from forge.observability.metric_actors import GlobalLoggingActor

            global_logger = await get_or_spawn_controller(
                "global_logger", GlobalLoggingActor
            )
        else:
            logger.debug(f"Rank {self.rank}: Using provided global_logger")

        config = await global_logger.get_config.call_one()
        if config is None:
            raise ValueError(f"Rank {self.rank}: Config not set—call init_config first")

        # Init local logger_backends only if log_per_rank=True, inject state from
        # the primary logger, which may have shared info for all secondary local loggers.
        for logger_backend_config in config["backends"]:
            if not logger_backend_config.get("log_per_rank", True):
                continue  # Skip globals/reduce
            cls_name = logger_backend_config["class"]
            primary_state = metadata_per_primary_backend.get(cls_name, {})
            logger_backend = get_logger_backend_class(cls_name)(logger_backend_config)
            await logger_backend.init(
                role="local", primary_logger_backend_metadata=primary_state
            )
            self.logger_backends.append(logger_backend)

        self._initialized_async = True

    async def push(
        self, key: str, value: Any, reduction: ReductionType = ReductionType.MEAN
    ) -> None:
        if not self._initialized_async:
            raise ValueError("Collector not initialized—call init first")

        if key not in self.accumulators:
            self.accumulators[key] = reduction.accumulator_class(reduction)

        self.accumulators[key].append(value)

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
        if not self._initialized_async:
            raise ValueError("Collector not initialized—call init first")

        if not self.accumulators:
            logger.debug(
                f"Collector rank {self.rank}: No metrics to flush for step {step}"
            )
            return {}

        # Snapshot states and reset immediately
        states = {}
        for key, acc in self.accumulators.items():
            states[key] = acc.get_state()
            acc.reset()

        # Reduce metrics from states for logging if any per-rank backend
        if self.logger_backends:
            metrics = {}
            for key, state in states.items():
                acc_class = ReductionType(state["reduction_type"]).accumulator_class
                metrics[key] = acc_class.get_reduced_value_from_states([state])

            # Log to local logger_backends
            for logger_backend in self.logger_backends:
                await logger_backend.log(metrics, step)

        return states if return_state else {}

    async def shutdown(self):
        """Shutdown logger_backends if initialized."""
        if not self._initialized_async:
            logger.debug(
                f"Collector rank {self.rank}: Not initialized, skipping shutdown"
            )
            return

        for logger_backend in self.logger_backends:
            await logger_backend.finish()


###########
# Backends #
###########


class LoggerBackend(ABC):
    """Abstract logger_backend for metric logging, e.g. wandb, jsonl, etc.

    #TODO: improve docstrings. Say how they are used/when/why/what they should do. Keep it short
    but informative. For example, it should behave differently if logging per rank or reducing.
    how global actor can call get_metadata_for_secondary_ranks from the primary run so it can share with the others
    during initialize.
    """

    def __init__(self, logger_backend_config: Dict[str, Any]):
        self.logger_backend_config = logger_backend_config

    @abstractmethod
    async def init(
        self,
        role: str,
        primary_logger_backend_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initializes backend for role in distributed logging flow.

        Called by GlobalLoggingActor: globals first, then broadcasts metadata to locals via fetchers.

        Args:
            role (str): "global" (controller/primary) or "local" (per-rank/secondary).
            primary_metadata (Optional[Dict[str, Any]]): From global backend for
                backend that required shared info, e.g. {"shared_run_id": "abc123"}.

        Raises: ValueError if missing metadata for shared local init.
        """
        if primary_logger_backend_metadata is None:
            primary_logger_backend_metadata = {}
        pass

    async def log(self, metrics: Dict[str, Any], step: int) -> None:
        pass

    async def finish(self) -> None:
        pass

    def get_metadata_for_secondary_ranks(self) -> Optional[Dict[str, Any]]:
        """Return sharable state after primary init (e.g., for shared modes). Called only on globals."""
        return None


class ConsoleBackend(LoggerBackend):
    def __init__(self, logger_backend_config: Dict[str, Any]):
        super().__init__(logger_backend_config)

    async def init(
        self,
        role: str,
        primary_logger_backend_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        pass

    async def log(self, metrics: Dict[str, Any], step: int) -> None:
        prefix = (
            get_actor_name_with_rank()
            if self.logger_backend_config.get("log_per_rank", True)
            else "GLOBAL"
        )
        logger.info(f"=== {prefix} METRICS STEP {step} ===")

        # TODO: Improve display. Maybe pprint? Currently requires loglevel == info
        for key, value in metrics.items():
            logger.info(f"  {key}: {value}")
        logger.info("==============================\n")

    async def finish(self) -> None:
        pass


class WandbBackend(LoggerBackend):
    """Reference: docs.wandb.ai/guides/track/log/distributed-training

    #TODO: give this better names
    #TODO: most likely delete wandb_rank_0_log_all
    valid_modes = [
            "wandb_all_log_all", # Track multiple processes
            "wandb_rank_0_log_all", #Track all processes to a single run
            "wandb_rank_0_reduce_all", # Track a single process
        ]
    """

    def __init__(self, logger_backend_config: Dict[str, Any]):
        super().__init__(logger_backend_config)
        self.project = logger_backend_config["project"]
        self.group = logger_backend_config.get("group", "experiment_group")
        self.name = None
        self.run = None
        self.mode = logger_backend_config.get("mode", "wandb_all_log_all")
        valid_modes = [
            "wandb_all_log_all",
            "wandb_rank_0_log_all",
            "wandb_rank_0_reduce_all",
        ]
        if self.mode not in valid_modes:
            raise ValueError(
                f"Invalid WandbBackend mode '{self.mode}'. Must be one of {valid_modes}."
            )

    async def init(
        self,
        role: str,
        primary_logger_backend_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        import wandb

        if primary_logger_backend_metadata is None:
            primary_logger_backend_metadata = {}
        self.name = (
            get_actor_name_with_rank() if role == "local" else "global_controller"
        )

        if self.mode == "wandb_all_log_all" and role == "local":
            self.run = wandb.init(
                project=self.project, group=self.group, name=self.name
            )
        elif self.mode == "wandb_rank_0_log_all":
            if role == "global":
                # Primary
                settings = wandb.Settings(
                    mode="shared", x_primary=True, x_label="controller_primary"
                )
                self.run = wandb.init(
                    project=self.project, group=self.group, settings=settings
                )
                # TODO: Make metric definitions automatic or configurable via logger_backend config
                self.run.define_metric("global_step")
                self.run.define_metric("train/loss", step_metric="global_step")
                self.run.define_metric("generate/tokens", step_metric="global_step")
            elif role == "local":
                # Secondary: Use shared_run_id from primary_logger_backend_metadata
                shared_id = primary_logger_backend_metadata.get("shared_run_id")
                if shared_id is None:
                    local_rank = current_rank().rank
                    raise ValueError(
                        f"Rank {local_rank}: Shared ID required but not provided"
                    )
                settings = wandb.Settings(
                    mode="shared", x_primary=False, x_label=self.name
                )
                self.run = wandb.init(
                    id=shared_id,
                    project=self.project,
                    group=self.group,
                    settings=settings,
                )
        elif self.mode == "wandb_rank_0_reduce_all" and role == "global":
            self.run = wandb.init(project=self.project, group=self.group)
            # self.run.define_metric("global_step")
            # self.run.define_metric("train/loss", step_metric="global_step")
            # self.run.define_metric("generate/tokens", step_metric="global_step")
        else:
            logger.debug(f"Skipped init for {self.mode} mode and {role} role")

    async def log(self, metrics: Dict[str, Any], step: int) -> None:
        if self.run:
            log_data = {**metrics, "global_step": step}
            self.run.log(log_data)
            logger.info(f"WandbBackend: Logged {len(metrics)} metrics at step {step}")
        else:
            logger.debug(f"WandbBackend: No run, skipping log for {self.name}")

    def get_metadata_for_secondary_ranks(self) -> Optional[Dict[str, Any]]:
        if self.run and self.mode == "wandb_rank_0_log_all":
            return {"shared_run_id": self.run.id}
        return None  # {} for others

    async def finish(self) -> None:
        if self.run:
            self.run.finish()
            logger.info(f"WandbBackend {self.name}: Finished run")


class StdAccumulator(MetricAccumulator):
    def __init__(self, reduction: ReductionType):
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
        return max(0.0, variance) ** 0.5  # sqrt, avoid negative due to floating point

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
