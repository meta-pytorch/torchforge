# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional

import wandb
from monarch.actor import context, current_rank


# Reduction Types
class ReductionType(Enum):
    MEAN = "mean"
    SUM = "sum"
    MAX = "max"
    MIN = "min"
    COUNT = "count"

    @property
    def accumulator_class(self):

        mapping = {
            ReductionType.MEAN: MeanAccumulator,
            ReductionType.SUM: SumAccumulator,
            ReductionType.MAX: MaxAccumulator,
            ReductionType.MIN: MinAccumulator,
            ReductionType.COUNT: CountAccumulator,
        }
        return mapping[self]


def get_actor_name_for_logging():
    """
    Extract actor information from Monarch context and return formatted name for logging.
    Returns string like "{actor_type}_{replica_id[-6:]}_r{local_rank_int}"

    #TODO: this is flaky as it currently relies on string parsing.
    """

    # Add more defensive checks
    ctx = context()
    if ctx is None:
        print("⚠️ Warning: context() returned None")
        return "UnknownActor_r0_l0"

    actor_instance = ctx.actor_instance
    if actor_instance is None:
        print("⚠️ Warning: actor_instance is None")
        return "UnknownActor_r0_l0"

    rank = current_rank()
    if rank is None:
        print("⚠️ Warning: current_rank() returned None")
        return "UnknownActor_r0_l0"

    actor_id_full = str(actor_instance.actor_id)

    # Parse the actor_id
    parts = actor_id_full.split(".")
    rank_name = "UnknownActor_r0_l0"  # fallback
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


# Simple push
async def push_metrics(
    key: str, value: Any, reduction: ReductionType = ReductionType.MEAN
) -> None:
    collector = MetricCollector()
    await collector.push(key, value, reduction)


def reduce_across_ranks(
    all_local_states: List[Dict[str, Dict[str, Any]]]
) -> Dict[str, Any]:
    """Reduce states across ranks per key."""
    if not all_local_states:
        return {}

    # Collect unique keys across all
    all_keys = set(k for states in all_local_states for k in states)
    print(f"🔧 Reduce: Unique keys: {list(all_keys)}")

    global_metrics = {}
    for key in all_keys:
        metric_states = [
            states.get(key) for states in all_local_states if key in states
        ]
        if not metric_states:
            continue

        first_red_type = metric_states[0]["reduction_type"]
        # Check consistency
        for state in metric_states[1:]:
            if state["reduction_type"] != first_red_type:
                raise ValueError(
                    f"Mismatched reduction types for key '{key}': {first_red_type} vs {state['reduction_type']}"
                )

        red_enum = ReductionType(first_red_type)
        acc_class = red_enum.accumulator_class
        reduced_value = acc_class.merge_states(metric_states)
        global_metrics[key] = reduced_value

    return global_metrics


# Backend ABC
class Backend(ABC):
    async def setup(
        self,
        config: Dict[str, Any],
        role: str,
        primary_states: Optional[Dict[str, Any]] = None,
    ) -> None:
        if primary_states is None:
            primary_states = {}
        pass

    async def log(self, metrics: Dict[str, Any], step: int) -> None:
        pass

    async def finish(self) -> None:
        pass

    def get_primary_state(self) -> Optional[Dict[str, Any]]:
        """Return sharable state after primary init (e.g., for shared modes). Called only on globals."""
        return None


class ConsoleBackend(Backend):
    async def setup(
        self,
        config: Dict[str, Any],
        role: str,
        primary_states: Optional[Dict[str, Any]] = None,
    ) -> None:
        if primary_states is None:
            primary_states = {}
        print("ConsoleBackend: Initialized")

    async def log(self, metrics: Dict[str, Any], step: int) -> None:
        try:
            rank = current_rank().rank
            rank_str = f"RANK {rank}"
        except Exception:
            rank_str = "GLOBAL"
        print(f"\n=== {rank_str} METRICS STEP {step} ===")
        for key, value in metrics.items():
            print(f"  {key}: {value}")
        print("==============================\n")

    async def finish(self) -> None:
        print("ConsoleBackend: Finished")


class WandbBackend(Backend):
    def __init__(self, backend_config: Dict[str, Any]):
        self.backend_config = backend_config
        self.project = backend_config["project"]
        self.group = backend_config.get("group", "experiment_group")
        self.name = None
        self.run = None
        self.mode = backend_config.get("mode", "wandb_all_log_all")

    async def setup(
        self,
        config: Dict[str, Any],
        role: str,
        primary_states: Optional[Dict[str, Any]] = None,
    ) -> None:
        if primary_states is None:
            primary_states = {}
        self.name = (
            get_actor_name_for_logging() if role == "local" else "global_controller"
        )

        if self.mode == "wandb_rank_0_reduce_all" and role == "local":
            # Should not init locals for reduce
            print("WandbBackend: Skipped local init for reduce mode")
            return

        if self.mode == "wandb_all_log_all" and role == "global":
            print("WandbBackend: Skipped global init for all_log_all mode")
            return

        if self.mode == "wandb_all_log_all":
            self.run = wandb.init(
                project=self.project, group=self.group, name=self.name
            )
            print(f"WandbBackend: Separate run '{self.name}' in group '{self.group}'")
        elif self.mode == "wandb_rank_0_log_all":
            if role == "global":
                # Primary
                settings = wandb.Settings(
                    mode="shared", x_primary=True, x_label="controller_primary"
                )
                self.run = wandb.init(
                    project=self.project, group=self.group, settings=settings
                )
                self.run.define_metric("global_step")
                self.run.define_metric("train/loss", step_metric="global_step")
                self.run.define_metric("generate/tokens", step_metric="global_step")
                print("🌍 Global: Defined metrics with global_step axis for shared mode")
            elif role == "local":
                # Secondary: Use shared_run_id from primary_states
                shared_id = primary_states.get("shared_run_id")
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
                print(
                    f"WandbBackend: Joined shared run '{shared_id}' as secondary with label '{self.name}'"
                )
        elif self.mode == "wandb_rank_0_reduce_all" and role == "global":
            self.run = wandb.init(project=self.project, group=self.group)
            self.run.define_metric("global_step")
            self.run.define_metric("train/loss", step_metric="global_step")
            self.run.define_metric("generate/tokens", step_metric="global_step")
            print("🌍 Global: Initialized single run for reduce mode")

    async def log(self, metrics: Dict[str, Any], step: int) -> None:
        if self.run:
            log_data = {**metrics, "global_step": step}
            print(f"WandbBackend: About to log data: {log_data} at step {step}")
            self.run.log(log_data)
            print(
                f"WandbBackend: Successfully logged {len(metrics)} metrics at step {step}"
            )
        else:
            print(f"WandbBackend: No run, skipping log for {self.name}")

    def get_primary_state(self) -> Optional[Dict[str, Any]]:
        if self.run and self.mode == "wandb_rank_0_log_all":
            return {"shared_run_id": self.run.id}
        return None  # {} for others

    async def finish(self) -> None:
        if self.run:
            self.run.finish()
            print(f"WandbBackend {self.name}: Finished run")


def create_backend(backend_config: Dict[str, Any]) -> Backend:
    backend_type = backend_config["class"]
    if backend_type == "console":
        return ConsoleBackend()
    elif backend_type == "wandb":
        return WandbBackend(backend_config)
    else:
        raise ValueError(f"Unknown backend type: {backend_type}")


class MetricAccumulator(ABC):
    def __init__(self, reduction: ReductionType):
        self.reduction_type = reduction

    @abstractmethod
    def append(self, value: Any) -> None:
        pass

    @abstractmethod
    def get_reduced_value(self) -> Any:
        pass

    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        pass

    @classmethod
    @abstractmethod
    def merge_states(cls, states: List[Dict[str, Any]]) -> Any:
        pass

    @abstractmethod
    def reset(self) -> None:
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

    def get_reduced_value(self) -> float:
        return self.sum / self.count if self.count > 0 else 0.0

    def get_state(self) -> Dict[str, Any]:
        return {
            "reduction_type": self.reduction_type.value,
            "sum": self.sum,
            "count": self.count,
        }

    @classmethod
    def merge_states(cls, states: List[Dict[str, Any]]) -> float:
        if not states:
            return 0.0
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

    def get_reduced_value(self) -> float:
        return self.total

    def get_state(self) -> Dict[str, Any]:
        return {"reduction_type": self.reduction_type.value, "total": self.total}

    @classmethod
    def merge_states(cls, states: List[Dict[str, Any]]) -> float:
        if not states:
            return 0.0
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

    def get_reduced_value(self) -> float:
        return self.max_val

    def get_state(self) -> Dict[str, Any]:
        return {"reduction_type": self.reduction_type.value, "max_val": self.max_val}

    @classmethod
    def merge_states(cls, states: List[Dict[str, Any]]) -> float:
        if not states:
            return float("-inf")
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

    def get_reduced_value(self) -> float:
        return self.min_val

    def get_state(self) -> Dict[str, Any]:
        return {"reduction_type": self.reduction_type.value, "min_val": self.min_val}

    @classmethod
    def merge_states(cls, states: List[Dict[str, Any]]) -> float:
        if not states:
            return float("inf")
        return min(s["min_val"] for s in states)

    def reset(self) -> None:
        self.min_val = float("inf")


class CountAccumulator(MetricAccumulator):
    def __init__(self, reduction: ReductionType):
        super().__init__(reduction)
        self.count = 0

    def append(self, value: Any) -> None:
        self.count += 1

    def get_reduced_value(self) -> int:
        return self.count

    def get_state(self) -> Dict[str, Any]:
        return {"reduction_type": self.reduction_type.value, "count": self.count}

    @classmethod
    def merge_states(cls, states: List[Dict[str, Any]]) -> int:
        if not states:
            return 0
        return sum(s["count"] for s in states)

    def reset(self) -> None:
        self.count = 0


def create_accumulator(reduction: ReductionType) -> MetricAccumulator:
    acc_class = reduction.accumulator_class
    return acc_class(reduction)


class MetricCollector:
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
        self.backends: List[Backend] = []
        self._initialized_async = False
        self.rank = current_rank().rank
        print(f"🔧 MetricCollector rank {self.rank}: Singleton initialized (unique)")

    async def _init(self, primary_backend_states: Dict[str, Dict[str, Any]]):
        if self._initialized_async:
            return

        from monarch.actor import get_or_spawn_controller

        from forge.controller.v3.metric_actors import GlobalLoggingActor

        global_logger = await get_or_spawn_controller(
            "global_logger", GlobalLoggingActor
        )

        config = await global_logger.get_metric_logger_cfg.call_one()
        if config is None:
            raise ValueError(f"Rank {self.rank}: Config not set—call init_config first")

        # Init local backends only if log_per_rank=True, inject primary states
        for backend_config in config["backends"]:
            if not backend_config.get("log_per_rank", True):
                continue  # Skip globals/reduce
            cls_name = backend_config["class"]
            primary_state = primary_backend_states.get(cls_name, {})
            backend = create_backend(backend_config)
            await backend.setup(config, role="local", primary_states=primary_state)
            self.backends.append(backend)
            print(f"🔧 Collector rank {self.rank}: Initialized local backend {cls_name}")

        self._initialized_async = True
        print(f"🔧 MetricCollector rank {self.rank}: Async initialization complete")

    async def push(
        self, key: str, value: Any, reduction: ReductionType = ReductionType.MEAN
    ):
        # Assume eager init; fallback to lazy
        if not self._initialized_async:
            raise ValueError("Collector not initialized—call init first")
        if key not in self.accumulators:
            self.accumulators[key] = create_accumulator(reduction)
        self.accumulators[key].append(value)
        print(
            f"🔧 Collector rank {self.rank}: Pushed {key}={value} (reduction={reduction.value})"
        )

    async def log_and_reset(
        self, step: int, return_state: bool = False
    ) -> Optional[Dict[str, Dict[str, Any]]]:
        """Log to local backends (if any), optionally return states, and reset."""
        if not self._initialized_async:
            raise ValueError("Collector not initialized—call init first")
        if not self.accumulators:
            print(f"🔧 Collector rank {self.rank}: No metrics to flush for step {step}")
            return {} if return_state else None

        print(
            f"🔧 Collector rank {self.rank}: Accumulators before flush: {list(self.accumulators.keys())}"
        )

        # Snapshot states and reset immediately
        states = {key: acc.get_state() for key, acc in self.accumulators.items()}
        for acc in list(self.accumulators.values()):
            acc.reset()
        self.accumulators.clear()

        # Derive metrics from states if needed
        if self.backends:
            metrics = {}
            for key, state in states.items():
                red_enum = ReductionType(state["reduction_type"])
                acc_class = red_enum.accumulator_class
                metrics[key] = acc_class.merge_states([state])
            print(f"🔧 Collector rank {self.rank}: Metrics: {metrics}")

            # Log to local backends
            for backend in self.backends:
                await backend.log(metrics, step)

        if return_state:
            print(f"🔧 Collector rank {self.rank}: States: {list(states.keys())}")

        print(f"🔧 Collector rank {self.rank}: Flushed and reset for step {step}")
        return states if return_state else None

    async def shutdown(self):
        """Shutdown backends if initialized."""
        if not self._initialized_async:
            print(f"🔧 Collector rank {self.rank}: Not initialized, skipping shutdown")
            return
        for backend in self.backends:
            await backend.finish()
        print(f"🔧 Collector rank {self.rank}: Shutdown complete")
