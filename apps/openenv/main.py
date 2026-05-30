# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
OpenEnv GRPO Training Script using GenericEnvClient.

This version uses GenericEnvClient and GenericAction to work with ANY
OpenEnv environment without requiring environment-specific packages.

Usage:
    python -m apps.openenv.main --config apps/openenv/llama3_8b_julia.yaml
    python -m apps.openenv.main --config apps/openenv/llama3_8b_coding.yaml
"""

from __future__ import annotations

# CRITICAL: Set CUDA allocator config BEFORE any PyTorch imports
# This enables expandable segments which:
# 1. Reduces GPU memory fragmentation
# 2. Enables GPU Direct RDMA for faster weight updates (~4s vs ~10s)
# 3. Prevents OOM errors when storage volume uses GPU memory
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import asyncio
import importlib
import logging
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, TYPE_CHECKING

# Type-only imports to avoid runtime import of openenv (which pulls in fastmcp/docket
# and conflicts with monarch's OpenTelemetry meter provider)
if TYPE_CHECKING:
    from openenv import GenericAction
    from openenv.core.client_types import StepResult

# CRITICAL: Add openenv directory to sys.path at module level
_appdir = Path(__file__).parent
if str(_appdir) not in sys.path:
    sys.path.insert(0, str(_appdir))

import torch
import torch.nn.functional as F
import torchstore as ts
import yaml
from datasets import load_dataset
from forge.actors.generator import Generator
from forge.actors.openenv import OpenEnvActor
# find_available_port is used by OpenEnvActor internally
from forge.actors.reference_model import ReferenceModel
from forge.actors.replay_buffer import ReplayBuffer
from forge.actors.trainer import TitanTrainer
from forge.controller.actor import ForgeActor
from forge.controller.provisioner import init_provisioner, shutdown
from forge.data_models.completion import Completion
from forge.observability.metric_actors import get_or_create_metric_logger
from forge.observability.metrics import record_metric, Reduce
from forge.observability.perf_tracker import Tracer
from forge.rl.loss import GRPOLoss, DAPOLoss
from forge.types import LauncherConfig, ProvisionerConfig, TrainBatch
from forge.util.checkpoint import drop_weights
from forge.util.config import parse
from monarch.actor import endpoint
from omegaconf import DictConfig, ListConfig
from vllm.transformers_utils.tokenizer import get_tokenizer


# Set up module logger
logger = logging.getLogger(__name__)
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level, logging.INFO),
    format="[%(levelname)s %(name)s] %(message)s",
)


@dataclass
class Episode:
    episode_id: str
    pad_id: int
    request_len: int
    response_len: int
    target: Any | None = None
    completion: Completion | None = None
    ref_logprobs: torch.Tensor | None = None
    generator_logprobs: torch.Tensor | None = None  # For GRPOLoss
    loss_mask: torch.Tensor | None = None  # For GRPOLoss
    reward: float | None = None
    advantage: float | None = None

    @property
    def policy_version(self) -> int | None:
        return self.completion.generator_version if self.completion else None

    @property
    def stop_reason(self) -> str | None:
        """Get stop reason from completion for truncation detection."""
        return self.completion.stop_reason if self.completion else None

    @property
    def request_tensor(self) -> torch.Tensor:
        tensor: torch.Tensor = self.completion.prompt_ids.to(torch.long)
        if tensor.shape[0] > self.request_len:  # truncate from left (keep end)
            tensor = tensor[-self.request_len :]
        elif tensor.shape[0] < self.request_len:  # left pad
            diff = self.request_len - tensor.shape[0]
            tensor = F.pad(tensor, (diff, 0), value=self.pad_id)
        return tensor

    @property
    def response_tensor(self) -> torch.Tensor:
        tensor: torch.Tensor = self.completion.token_ids.to(torch.long)
        if tensor.shape[0] > self.response_len:  # truncate from right (keep beginning)
            tensor = tensor[: self.response_len]
        elif tensor.shape[0] < self.response_len:  # right pad
            diff = self.response_len - tensor.shape[0]
            tensor = F.pad(tensor, (0, diff), value=self.pad_id)
        return tensor


Group = list[Episode]
Policy = Generator


def load_function_from_string(func_ref: str) -> Callable:
    """Load a function from a string reference like 'module.function_name'."""
    openenv_dir = Path(__file__).parent
    if str(openenv_dir) not in sys.path:
        sys.path.insert(0, str(openenv_dir))

    module_name, func_name = func_ref.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, func_name)


def function_constructor(loader, node):
    """YAML constructor for !function tag."""
    value = loader.construct_scalar(node)
    return ("!function", value)


yaml.add_constructor("!function", function_constructor, Loader=yaml.SafeLoader)


def collate(
    batches: list[Group],
) -> list[TrainBatch]:
    """Collates a list of batches into TrainBatch objects.

    Supports both GRPOLoss (requires generator_logprobs, loss_mask) and DAPOLoss.
    """
    result = []
    for batch_idx, batch in enumerate(batches):
        logger.debug(f"collate Processing batch {batch_idx}, len={len(batch)}")

        request = [e.request_tensor for e in batch]
        request = torch.stack(request)

        response = [e.response_tensor for e in batch]
        response = torch.stack(response)

        input_ids = torch.cat([request, response], dim=1)
        seq_len = input_ids.shape[1]

        # ref_logprobs is optional - only stack if all episodes have it
        ref_logprobs = None
        if all(e.ref_logprobs is not None for e in batch):
            ref_logprobs = torch.stack([e.ref_logprobs for e in batch])

        advantages = [e.advantage for e in batch]
        advantages = torch.tensor(advantages).unsqueeze(-1)  # [b x 1]
        advantages = advantages.expand(-1, seq_len)  # [b x s]

        generator_logprobs = torch.stack([e.generator_logprobs for e in batch])
        loss_mask = torch.stack([e.loss_mask for e in batch])

        loss_inputs = {
            "generator_logprobs": generator_logprobs,
            "loss_mask": loss_mask,
            "advantages": advantages,
        }
        # Include ref_logprobs for GRPOLoss (uses it for KL penalty when beta > 0)
        if ref_logprobs is not None:
            loss_inputs["ref_logprobs"] = ref_logprobs

        result.append(
            TrainBatch(
                model_inputs={"tokens": input_ids},
                loss_inputs=loss_inputs,
            )
        )
    return result


def make_loss(cfg: DictConfig):
    """Factory function to create loss based on config.

    Supports both GRPOLoss and DAPOLoss based on `loss_type` config.

    Args:
        cfg: Configuration dict containing `grpo` section with:
            - loss_type: "grpo" or "dapo" (default: "dapo")
            - beta: KL penalty coefficient (for GRPOLoss only)
            - clip_eps_low / clip_low: Lower clipping bound
            - clip_eps_high / clip_high: Upper clipping bound
            - agg_type: Aggregation type
            - dual_clip_c: Dual-clip constant (for DAPOLoss only)

    Returns:
        Loss function (GRPOLoss or DAPOLoss instance)
    """
    grpo_cfg = cfg.get("grpo", {})
    loss_type = grpo_cfg.get("loss_type", "dapo").lower()

    # Support both naming conventions
    clip_low = grpo_cfg.get("clip_eps_low", grpo_cfg.get("clip_low", 0.2))
    clip_high = grpo_cfg.get("clip_eps_high", grpo_cfg.get("clip_high", 0.28))

    if loss_type == "grpo":
        beta = grpo_cfg.get("beta", 0.1)
        agg_type = grpo_cfg.get("agg_type", "fixed_horizon")
        logger.info(
            f"Using GRPOLoss with clip_low={clip_low}, clip_high={clip_high}, "
            f"beta={beta}, agg_type={agg_type}"
        )
        return GRPOLoss(
            clip_low=clip_low,
            clip_high=clip_high,
            beta=beta,
            agg_type=agg_type,
        )
    elif loss_type == "dapo":
        dual_clip_c = grpo_cfg.get("dual_clip_c", 3.0)
        agg_type = grpo_cfg.get("agg_type", "token_mean")
        logger.info(
            f"Using DAPOLoss with clip_low={clip_low}, clip_high={clip_high}, "
            f"dual_clip_c={dual_clip_c}, agg_type={agg_type}"
        )
        return DAPOLoss(
            clip_low=clip_low,
            clip_high=clip_high,
            dual_clip_c=dual_clip_c,
            agg_type=agg_type,
        )
    else:
        raise ValueError(
            f"Unknown loss_type: {loss_type}. Supported: 'grpo', 'dapo'"
        )


@dataclass
class GenericRewardActor(ForgeActor):
    """Generic reward actor that uses GenericEnvClient and GenericAction.

    Supports multiple env_actors for parallel evaluation across different
    WebSocket connections. Includes circuit breaker pattern to detect and
    restart unhealthy containers.
    """

    env_actors: list  # List of OpenEnvActor instances
    build_action_fn: Callable[[str, Dict[str, Any]], GenericAction]
    evaluate_response_fn: Callable[[StepResult, str, Dict[str, Any]], float]
    evaluation_timeout_s: float = 60.0

    # Circuit breaker configuration
    circuit_breaker_threshold: int = 10  # Timeouts before marking unhealthy
    circuit_breaker_window_s: float = 60.0  # Time window for counting timeouts
    circuit_breaker_cooldown_s: float = 30.0  # Cooldown before retrying unhealthy actor

    _request_counter: int = 0  # For round-robin distribution

    # Circuit breaker state (initialized in setup using field defaults for safety)
    _actor_timeout_counts: list = field(default_factory=list)  # Timeout count per actor
    _actor_timeout_timestamps: list = field(default_factory=list)  # Recent timeout timestamps per actor
    _actor_healthy: list = field(default_factory=list)  # Health status per actor
    _actor_cooldown_until: list = field(default_factory=list)  # Cooldown end time per actor
    _restart_in_progress: list = field(default_factory=list)  # Restart lock per actor
    _restart_tasks: list = field(default_factory=list)  # Track restart tasks for cleanup

    @endpoint
    async def setup(self):
        """Ensure the openenv directory is in sys.path for imports."""
        logger.debug("GenericRewardActor.setup Starting setup...")
        openenv_dir = Path(__file__).parent
        if str(openenv_dir) not in sys.path:
            sys.path.insert(0, str(openenv_dir))

        # Initialize circuit breaker state
        num_actors = len(self.env_actors)
        self._actor_timeout_counts = [0] * num_actors
        self._actor_timeout_timestamps = [[] for _ in range(num_actors)]
        self._actor_healthy = [True] * num_actors
        self._actor_cooldown_until = [0.0] * num_actors
        self._restart_in_progress = [False] * num_actors

        logger.debug(
            f"GenericRewardActor.setup Timeout set to {self.evaluation_timeout_s}s"
        )
        logger.debug(f"GenericRewardActor.setup Using {num_actors} env_actors for parallel evaluation")
        logger.debug(
            f"GenericRewardActor.setup Circuit breaker: threshold={self.circuit_breaker_threshold}, "
            f"window={self.circuit_breaker_window_s}s"
        )
        logger.debug("GenericRewardActor.setup Setup complete!")

    def _get_healthy_actor_idx(self) -> int:
        """Get the next healthy actor index using round-robin with health awareness.

        Returns:
            Index of a healthy actor, or the least-bad unhealthy actor if all are unhealthy
        """
        num_actors = len(self.env_actors)
        current_time = time.time()

        # Try to find a healthy actor using round-robin
        for _ in range(num_actors):
            idx = self._request_counter % num_actors
            self._request_counter += 1

            if self._actor_healthy[idx]:
                return idx

            # Check if cooldown has expired for unhealthy actor
            if current_time >= self._actor_cooldown_until[idx]:
                # Cooldown expired, give it another chance
                logger.info(f"Circuit breaker: Actor {idx} cooldown expired, retrying")
                self._actor_healthy[idx] = True
                self._actor_timeout_counts[idx] = 0
                self._actor_timeout_timestamps[idx] = []
                return idx

        # All actors are unhealthy and in cooldown - use the one with earliest cooldown end
        earliest_idx = min(range(num_actors), key=lambda i: self._actor_cooldown_until[i])
        logger.warning(f"Circuit breaker: All actors unhealthy, using actor {earliest_idx}")
        return earliest_idx

    def _record_timeout(self, actor_idx: int):
        """Record a timeout for an actor and check if circuit should trip."""
        current_time = time.time()

        # Add timestamp to recent timeouts
        self._actor_timeout_timestamps[actor_idx].append(current_time)

        # Remove old timestamps outside the window
        window_start = current_time - self.circuit_breaker_window_s
        self._actor_timeout_timestamps[actor_idx] = [
            ts for ts in self._actor_timeout_timestamps[actor_idx]
            if ts >= window_start
        ]

        # Count recent timeouts
        recent_timeouts = len(self._actor_timeout_timestamps[actor_idx])
        self._actor_timeout_counts[actor_idx] = recent_timeouts

        record_metric(f"circuit_breaker/actor_{actor_idx}/timeout_count", recent_timeouts, Reduce.MAX)

        # Check if circuit should trip
        if recent_timeouts >= self.circuit_breaker_threshold and self._actor_healthy[actor_idx]:
            logger.error(
                f"Circuit breaker TRIPPED for actor {actor_idx}: "
                f"{recent_timeouts} timeouts in {self.circuit_breaker_window_s}s"
            )
            self._actor_healthy[actor_idx] = False
            self._actor_cooldown_until[actor_idx] = current_time + self.circuit_breaker_cooldown_s
            record_metric(f"circuit_breaker/actor_{actor_idx}/tripped", 1, Reduce.SUM)

            # Trigger async restart with error handling callback
            def _restart_done_callback(task, idx=actor_idx):
                try:
                    exc = task.exception()
                    if exc is not None:
                        logger.error(f"Circuit breaker: Restart task for actor {idx} failed: {exc}")
                except asyncio.CancelledError:
                    pass  # Task was cancelled, not an error
                # Remove completed task from tracking list
                if task in self._restart_tasks:
                    self._restart_tasks.remove(task)

            restart_task = asyncio.create_task(self._restart_actor(actor_idx))
            restart_task.add_done_callback(_restart_done_callback)
            self._restart_tasks.append(restart_task)  # Track for cleanup during shutdown

    def _record_success(self, actor_idx: int):
        """Record a successful execution for an actor."""
        # Successful execution reduces timeout pressure
        # We don't clear all timeouts, but the window will naturally expire them

    async def _restart_actor(self, actor_idx: int):
        """Restart an unhealthy actor's container pool."""
        if self._restart_in_progress[actor_idx]:
            logger.debug(f"Restart already in progress for actor {actor_idx}")
            return

        self._restart_in_progress[actor_idx] = True
        record_metric(f"circuit_breaker/actor_{actor_idx}/restart_attempt", 1, Reduce.SUM)

        try:
            logger.warning(f"Circuit breaker: Initiating FULL POOL restart for actor {actor_idx}")

            env_actor = self.env_actors[actor_idx]
            result = await env_actor.restart_container.call_one()

            if result.get("success"):
                logger.info(
                    f"Circuit breaker: Actor {actor_idx} pool restarted successfully - "
                    f"{result.get('num_containers')} containers, {result.get('num_connections')} connections"
                )
                self._actor_healthy[actor_idx] = True
                self._actor_timeout_counts[actor_idx] = 0
                self._actor_timeout_timestamps[actor_idx] = []
                self._actor_cooldown_until[actor_idx] = 0.0
                record_metric(f"circuit_breaker/actor_{actor_idx}/restart_success", 1, Reduce.SUM)
            else:
                logger.error(
                    f"Circuit breaker: Actor {actor_idx} restart failed: {result.get('error')}"
                )
                # Extend cooldown on failure
                self._actor_cooldown_until[actor_idx] = time.time() + self.circuit_breaker_cooldown_s * 2
                record_metric(f"circuit_breaker/actor_{actor_idx}/restart_failure", 1, Reduce.SUM)

        except Exception as e:
            logger.error(f"Circuit breaker: Exception during restart of actor {actor_idx}: {e}")
            import traceback
            traceback.print_exc()
            self._actor_cooldown_until[actor_idx] = time.time() + self.circuit_breaker_cooldown_s * 2
            record_metric(f"circuit_breaker/actor_{actor_idx}/restart_failure", 1, Reduce.SUM)

        finally:
            self._restart_in_progress[actor_idx] = False

    @endpoint
    async def evaluate_response(self, prompt: str, response: str, target: Any) -> float:
        """
        Evaluate response using task-specific functions with timeout protection.

        Uses health-aware round-robin distribution across env_actors with
        circuit breaker pattern to detect and restart unhealthy containers.

        Args:
            prompt: The problem description
            response: The model's generated response
            target: The target/test data from dataset

        Returns:
            Reward score (0.0 if timeout or error)
        """
        # Initialize actor index for error logging (may be updated below)
        env_actor_idx = -1

        try:
            # Build action using task-specific function (returns GenericAction)
            sample = {"target": target}
            action = self.build_action_fn(response, sample)

            # Get healthy actor using circuit breaker logic
            env_actor_idx = self._get_healthy_actor_idx()
            env_actor = self.env_actors[env_actor_idx]

            # Execute in environment with timeout protection
            result = await asyncio.wait_for(
                env_actor.execute.call_one(
                    dict(action)
                ),  # Convert to dict for serialization
                timeout=self.evaluation_timeout_s,
            )

            # Record success
            self._record_success(env_actor_idx)

            # Evaluate result using task-specific function
            reward = self.evaluate_response_fn(result, response, sample)

            record_metric("reward/evaluate_response/sum_reward", reward, Reduce.SUM)
            record_metric("reward/evaluate_response/avg_reward", reward, Reduce.MEAN)
            record_metric("reward/evaluate_response/count_calls", 1, Reduce.SUM)

            return reward

        except asyncio.TimeoutError:
            logger.warning(
                f"Evaluation timeout after {self.evaluation_timeout_s}s on actor {env_actor_idx} "
                f"- likely infinite loop in generated code"
            )
            # Record timeout for circuit breaker
            self._record_timeout(env_actor_idx)
            record_metric("reward/evaluate_response/timeout_count", 1, Reduce.SUM)
            return 0.0

        except Exception as e:
            logger.error(f"Evaluation error on actor {env_actor_idx}: {e}")
            # Connection errors also count towards circuit breaker
            # Only record timeout if we actually got an actor (env_actor_idx >= 0)
            # to avoid recording against wrong actor when build_action_fn fails
            if env_actor_idx >= 0 and ("connection" in str(e).lower() or "websocket" in str(e).lower()):
                self._record_timeout(env_actor_idx)
            record_metric("reward/evaluate_response/error_count", 1, Reduce.SUM)
            return 0.0

    @endpoint
    async def cancel_restart_tasks(self) -> int:
        """Cancel all pending restart tasks during shutdown.

        Returns:
            Number of tasks that were cancelled
        """
        cancelled_count = 0
        for task in self._restart_tasks:
            if not task.done():
                task.cancel()
                cancelled_count += 1
        # Wait for all tasks to complete cancellation
        if self._restart_tasks:
            await asyncio.gather(*self._restart_tasks, return_exceptions=True)
        self._restart_tasks.clear()
        return cancelled_count

    @endpoint
    async def get_health_status(self) -> Dict[str, Any]:
        """Get health status of all env_actors for monitoring."""
        return {
            "actors": [
                {
                    "index": i,
                    "healthy": self._actor_healthy[i],
                    "timeout_count": self._actor_timeout_counts[i],
                    "cooldown_remaining": max(0, self._actor_cooldown_until[i] - time.time()),
                    "restart_in_progress": self._restart_in_progress[i],
                }
                for i in range(len(self.env_actors))
            ],
            "healthy_count": sum(self._actor_healthy),
            "total_count": len(self.env_actors),
        }


@dataclass
class ComputeAdvantages(ForgeActor):
    @endpoint
    async def setup(self):
        logger.debug("ComputeAdvantages.setup Setup complete!")

    @endpoint
    async def compute(self, group: Group) -> list[float]:
        rewards = torch.tensor([[e.reward for e in group]], dtype=torch.float32)
        mean = rewards.mean(1, keepdim=True)
        std = rewards.std(1, keepdim=True)
        advantages = (rewards - mean) / (std + 1e-4)
        return advantages.squeeze(0).tolist()


@dataclass
class GenericDatasetActor(ForgeActor):
    """Generic dataset actor that uses task-specific transformation function."""

    path: str
    revision: str = "main"
    data_split: str = "train"
    streaming: bool = False
    model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    transform_sample_fn: Callable | None = None

    @endpoint
    async def setup(self):
        openenv_dir = Path(__file__).parent
        if str(openenv_dir) not in sys.path:
            sys.path.insert(0, str(openenv_dir))

        self._tokenizer = get_tokenizer(self.model)

        logger.info(f"Loading dataset from: {self.path}")
        if os.path.isfile(self.path):
            if self.path.endswith(".parquet"):
                ds = load_dataset(
                    "parquet",
                    data_files={"train": self.path},
                    split=self.data_split,
                )
            elif self.path.endswith(".json"):
                ds = load_dataset(
                    "json",
                    data_files={"train": self.path},
                    split=self.data_split,
                )
            else:
                raise ValueError(f"Unsupported file format: {self.path}")
        else:
            ds = load_dataset(
                self.path,
                split=self.data_split,
                streaming=self.streaming,
                revision=self.revision,
            )

        if len(ds) == 0:
            raise ValueError(f"Dataset is empty after loading from {self.path}.")

        logger.info(f"Dataset loaded, size: {len(ds)}")

        if self.transform_sample_fn:
            def transform_wrapper(sample):
                return self.transform_sample_fn(sample, self._tokenizer)

            original_size = len(ds)
            ds = ds.filter(lambda x: transform_wrapper(x) is not None)
            filtered_size = len(ds)
            logger.info(f"Dataset filtered: {original_size} -> {filtered_size}")

            if filtered_size == 0:
                raise ValueError(
                    f"Dataset transform filtered out ALL {original_size} samples!"
                )

            ds = ds.map(transform_wrapper)

        ds = ds.shuffle()

        if len(ds) == 0:
            raise ValueError("Dataset is empty after all transformations!")

        self._dataset = ds  # Keep reference for looping
        self._iterator = iter(ds)
        self._loop_count = 0
        logger.info(f"Dataset setup complete! Size: {len(ds)}")

    @endpoint
    async def sample(self) -> dict[str, str] | None:
        try:
            sample = next(self._iterator)
            record_metric("dataset/sample/count_samples_generated", 1, Reduce.SUM)
            if "request" in sample:
                record_metric(
                    "dataset/sample/avg_sample_len",
                    len(sample["request"]),
                    Reduce.MEAN,
                )
            return sample
        except StopIteration:
            # Loop the dataset instead of returning None
            self._loop_count += 1
            logger.info(f"[DATASET] Completed epoch {self._loop_count}, reshuffling and restarting...")
            record_metric("dataset/epoch_completed", 1, Reduce.SUM)
            self._dataset = self._dataset.shuffle()
            self._iterator = iter(self._dataset)
            # Return first sample from new epoch
            sample = next(self._iterator)
            record_metric("dataset/sample/count_samples_generated", 1, Reduce.SUM)
            return sample

    @endpoint
    async def pad_token(self):
        if self._tokenizer.pad_token_id is not None:
            return self._tokenizer.pad_token_id
        return self._tokenizer.eos_token_id


async def main(cfg: DictConfig):
    """Main GRPO training loop using GenericEnvClient."""
    group_size = cfg.group_size
    max_req_tokens = cfg.max_req_tokens
    max_res_tokens = cfg.max_res_tokens

    # Load task-specific functions
    logger.debug("main Loading task-specific functions...")
    task_config = cfg.task

    build_action_fn = None
    evaluate_response_fn = None
    transform_sample_fn = None

    if (
        isinstance(task_config.build_action, (tuple, list, ListConfig))
        and len(task_config.build_action) == 2
        and task_config.build_action[0] == "!function"
    ):
        build_action_fn = load_function_from_string(task_config.build_action[1])

    if (
        isinstance(task_config.evaluate_response, (tuple, list, ListConfig))
        and len(task_config.evaluate_response) == 2
        and task_config.evaluate_response[0] == "!function"
    ):
        evaluate_response_fn = load_function_from_string(
            task_config.evaluate_response[1]
        )

    if hasattr(task_config, "transform_sample"):
        if (
            isinstance(task_config.transform_sample, (tuple, list, ListConfig))
            and len(task_config.transform_sample) == 2
            and task_config.transform_sample[0] == "!function"
        ):
            transform_sample_fn = load_function_from_string(
                task_config.transform_sample[1]
            )

    logger.debug("main All task-specific functions loaded successfully")

    # Global setups
    provisioner = None
    if cfg.get("provisioner", None) is not None:
        provisioner = await init_provisioner(
            ProvisionerConfig(launcher_config=LauncherConfig(**cfg.provisioner))
        )
    else:
        provisioner = await init_provisioner()

    metric_logging_cfg = cfg.get("metric_logging", {})
    mlogger = await get_or_create_metric_logger(process_name="Controller")
    await mlogger.init_backends.call_one(metric_logging_cfg)

    # ---- Setup loss function ---- #
    loss_fn = make_loss(cfg)

    # Fail-fast: Check loss/ref_model compatibility before spawning actors
    uses_ref_model = cfg.get("services", {}).get("ref_model") is not None
    if uses_ref_model and not isinstance(loss_fn, GRPOLoss):
        logger.warning(
            f"ref_model is configured but {type(loss_fn).__name__} does not use ref_logprobs. "
            "Consider removing the ref_model service config to save GPU resources."
        )
    if isinstance(loss_fn, GRPOLoss) and loss_fn.beta > 0 and not uses_ref_model:
        raise ValueError(
            f"GRPOLoss with beta={loss_fn.beta} requires ref_logprobs, but ref_model is not configured. "
            "Either add ref_model to services config or set beta=0."
        )

    # Setup OpenEnvActor - works with ANY OpenEnv Docker image
    openenv_config = cfg.get("openenv_config", {})
    docker_image = openenv_config.get("docker_image")
    env_vars = openenv_config.get("env_vars", {})
    container_timeout_s = openenv_config.get("container_timeout_s", 180.0)
    request_timeout_s = openenv_config.get("request_timeout_s", 120.0)
    container_memory_gb = openenv_config.get("container_memory_gb", 4)

    # Set environment variables from config
    if "PORT" not in env_vars:
        env_vars["PORT"] = str(openenv_config.get("port", 8000))
    if "NUM_WORKER" not in env_vars:
        env_vars["NUM_WORKER"] = str(openenv_config.get("num_worker", 4))

    # Get env_name for actor mesh naming and logging paths
    env_name = openenv_config.get("env_name", task_config.get("env_name", "generic"))

    logger.debug(
        f"main Initializing OpenEnvActor with image={docker_image}..."
    )

    # Smart container allocation: Create one actor per concurrent evaluation needed
    # Each actor manages its own container(s) with connection pooling

    num_env_actors = openenv_config.get("num_env_actors", cfg.get("group_size", 8))
    num_containers_per_actor = openenv_config.get("num_containers", 1)
    num_connections_per_container = openenv_config.get("num_connections", 1)

    logger.info(
        f"Creating {num_env_actors} env_actors, each with {num_containers_per_actor} containers "
        f"and {num_connections_per_container} connections per container"
    )

    # Create env_actors
    env_actors = []
    base_port = openenv_config.get("port", 8000)

    for i in range(num_env_actors):
        actor_env_vars = env_vars.copy()
        # Each actor starts from a different port range to avoid conflicts
        actor_port = base_port - (i * num_containers_per_actor * 2)

        logger.debug(
            f"Creating env_actor {i + 1}/{num_env_actors} starting at port {actor_port}"
        )

        env_actor = await OpenEnvActor.options(
            **cfg.actors.get(f"{env_name}_env", cfg.actors.get("env", {}))
        ).as_actor(
            docker_image=docker_image,
            env_name=env_name,
            env_vars=actor_env_vars,
            container_timeout_s=container_timeout_s,
            request_timeout_s=request_timeout_s,
            container_memory_gb=container_memory_gb,
            port=actor_port,
            num_containers=num_containers_per_actor,
            num_connections=num_connections_per_container,
        )
        env_actors.append(env_actor)

    total_containers = num_env_actors * num_containers_per_actor
    logger.info(
        f"All {num_env_actors} env_actors initialized successfully "
        f"({total_containers} total containers)"
    )

    # Create all other actors
    async def noop():
        return None

    (
        dataloader,
        policy,
        trainer,
        replay_buffer,
        compute_advantages,
        ref_model,
        reward_actor,
    ) = await asyncio.gather(
        GenericDatasetActor.options(**cfg.actors.dataset).as_actor(
            path=cfg.dataset.path,
            revision=cfg.dataset.get("revision", "main"),
            data_split=cfg.dataset.get("data_split", "train"),
            streaming=cfg.dataset.get("streaming", False),
            model=cfg.model,
            transform_sample_fn=transform_sample_fn,
        ),
        Policy.options(**cfg.services.policy).as_service(**cfg.policy),
        TitanTrainer.options(**cfg.actors.trainer).as_actor(
            **cfg.trainer,
            loss=loss_fn,
        ),
        ReplayBuffer.options(**cfg.actors.replay_buffer).as_actor(
            **cfg.replay_buffer, collate=collate
        ),
        ComputeAdvantages.options(**cfg.actors.compute_advantages).as_actor(),
        (
            ReferenceModel.options(**cfg.services.ref_model).as_service(**cfg.ref_model)
            if uses_ref_model
            else noop()
        ),
        GenericRewardActor.options(**cfg.services.reward_actor).as_service(
            env_actors=env_actors,
            build_action_fn=build_action_fn,
            evaluate_response_fn=evaluate_response_fn,
            evaluation_timeout_s=cfg.get("evaluation_timeout_s", 60.0),
            # Circuit breaker configuration
            circuit_breaker_threshold=cfg.get("circuit_breaker", {}).get("threshold", 10),
            circuit_breaker_window_s=cfg.get("circuit_breaker", {}).get("window_s", 60.0),
            circuit_breaker_cooldown_s=cfg.get("circuit_breaker", {}).get("cooldown_s", 30.0),
        ),
    )
    logger.debug("main asyncio.gather completed successfully!")

    max_steps = cfg.trainer.training.steps or -1

    print("All services initialized successfully!")
    shutdown_event = asyncio.Event()

    # Initialize torchstore
    trainer_num_procs = cfg.actors.trainer["procs"]
    trainer_host_mesh_name = cfg.actors.trainer["mesh_name"]
    trainer_hosts = await provisioner.get_host_mesh(trainer_host_mesh_name)
    await ts.initialize(
        mesh=trainer_hosts.spawn_procs(per_host={"procs": trainer_num_procs}),
        strategy=ts.LocalRankStrategy(),
    )
    print("Torchstore successfully initialized with local rank strategy")

    # Episode dropout configuration
    dropout_cfg = cfg.get("episode_dropout", {})
    enable_variance_dropout = dropout_cfg.get("enable_variance_dropout", True)
    enable_truncation_dropout = dropout_cfg.get("enable_truncation_dropout", True)
    variance_threshold = dropout_cfg.get("variance_threshold", 1e-3)

    # Core RL loops
    async def continuous_rollouts():
        try:
            rollout_count = 0
            consecutive_errors = 0
            max_consecutive_errors = int(os.environ.get("FORGE_MAX_ROLLOUT_ERRORS", "50"))
            rollout_timeout_s = float(os.environ.get("FORGE_ROLLOUT_TIMEOUT_S", "300"))

            pad_id = await dataloader.pad_token.call_one()

            # Rollout-side backpressure settings
            # Only produce new episodes when buffer needs them (prevents sample waste)
            batch_size = cfg.batch_size
            episodes_per_step = batch_size * group_size
            # Buffer target: enough for N training steps (configurable via env var)
            buffer_target_steps = int(os.environ.get("FORGE_BUFFER_TARGET_STEPS", "4"))
            max_buffer_episodes = episodes_per_step * buffer_target_steps
            backpressure_check_interval = float(os.environ.get("FORGE_BACKPRESSURE_CHECK_INTERVAL", "0.5"))

            while not shutdown_event.is_set():
                try:
                    t = Tracer("main_perf/continuous_rollouts")
                    t.start()

                    # ROLLOUT BACKPRESSURE: Check if buffer needs more episodes
                    # This prevents overproduction and sample waste
                    try:
                        buffer_size = await replay_buffer._numel.call_one()
                        if buffer_size >= max_buffer_episodes:
                            # Buffer is full enough, wait before producing more
                            record_metric("rollout/backpressure/paused", 1, Reduce.SUM)
                            record_metric("rollout/backpressure/buffer_size", buffer_size, Reduce.MAX)
                            await asyncio.sleep(backpressure_check_interval)
                            t.stop()  # Don't count this as a rollout iteration
                            continue
                    except Exception as e:
                        # If buffer check fails, continue with rollout
                        logger.debug(f"Buffer size check failed: {e}")

                    t.step("backpressure_check")

                    # Timeout on data loading
                    try:
                        sample = await asyncio.wait_for(
                            dataloader.sample.call_one(),
                            timeout=30.0,
                        )
                    except asyncio.TimeoutError:
                        logger.warning("[ROLLOUT] Timeout waiting for dataloader sample")
                        record_metric("main/continuous_rollouts/dataloader_timeout", 1, Reduce.SUM)
                        continue

                    if sample is None:
                        print("Dataloader is empty, exiting continuous rollout")
                        return

                    t.step("data_loading")

                    prompt, target = sample["request"], sample["target"]

                    # Timeout on policy generation
                    try:
                        responses: list[Completion] = await asyncio.wait_for(
                            policy.generate.route(prompt),
                            timeout=rollout_timeout_s,
                        )
                    except asyncio.TimeoutError as timeout_err:
                        logger.error(
                            f"[ROLLOUT] Timeout after {rollout_timeout_s}s waiting for policy.generate(). "
                            f"Generator may be stuck during weight update."
                        )
                        record_metric("main/continuous_rollouts/generation_timeout", 1, Reduce.SUM)
                        consecutive_errors += 1
                        if consecutive_errors >= max_consecutive_errors:
                            raise RuntimeError(
                                f"[ROLLOUT FAILURE] {consecutive_errors} consecutive rollout errors. "
                                f"Generator appears to be unresponsive."
                            ) from timeout_err
                        continue

                    t.step("policy_generation")

                    episodes = []
                    input_ids = torch.ones(
                        (group_size, max_req_tokens + max_res_tokens),
                        dtype=torch.long,
                    )
                    seq_len = max_req_tokens + max_res_tokens

                    # Track evaluation errors for circuit breaker
                    eval_errors_this_batch = 0

                    # Create episodes first
                    for i, response in enumerate(responses):
                        # Both GRPOLoss and DAPOLoss need generator_logprobs and loss_mask
                        # Validate logprobs exist
                        if response.logprobs is None:
                            raise ValueError(
                                "Completion.logprobs is None. "
                                "Ensure Generator returns logprobs by setting 'logprobs: 1' in sampling_params config."
                            )

                        # Prepare generator_logprobs (shifted for next-token prediction)
                        actual_response_len = response.token_ids.shape[0]
                        generator_logprobs = torch.zeros(seq_len, dtype=response.logprobs.dtype)
                        generator_logprobs[
                            max_req_tokens : max_req_tokens + actual_response_len
                        ] = response.logprobs
                        generator_logprobs = torch.roll(generator_logprobs, shifts=-1, dims=0)
                        generator_logprobs[-1] = 0.0

                        # Prepare loss_mask
                        response_mask = torch.zeros(seq_len, dtype=torch.float32)
                        response_mask[max_req_tokens : max_req_tokens + actual_response_len] = 1.0
                        loss_mask = torch.roll(response_mask, shifts=-1, dims=0)
                        loss_mask[-1] = 0.0

                        episode = Episode(
                            episode_id=str(uuid.uuid4()),
                            pad_id=pad_id,
                            request_len=max_req_tokens,
                            response_len=max_res_tokens,
                            target=target,
                            completion=response,
                            generator_logprobs=generator_logprobs,
                            loss_mask=loss_mask,
                        )
                        episodes.append(episode)

                    # Parallel reward evaluation using asyncio.gather
                    async def evaluate_single(
                        idx, episode, response, *, _prompt=prompt, _target=target
                    ):
                        try:
                            reward = await reward_actor.evaluate_response.route(
                                prompt=_prompt, response=response.text, target=_target
                            )
                            return idx, reward, None
                        except Exception as eval_exc:
                            return idx, 0.0, eval_exc

                    eval_tasks = [
                        evaluate_single(i, ep, resp)
                        for i, (ep, resp) in enumerate(
                            zip(episodes, responses, strict=True)
                        )
                    ]
                    eval_results = await asyncio.gather(*eval_tasks)

                    # Process results
                    for idx, reward, error in eval_results:
                        episodes[idx].reward = reward
                        if error is not None:
                            logger.warning(f"[ROLLOUT] Reward evaluation failed: {error}")
                            eval_errors_this_batch += 1
                            record_metric("main/continuous_rollouts/eval_error", 1, Reduce.SUM)

                    # Build input_ids after rewards are assigned
                    for i, episode in enumerate(episodes):
                        input_ids[i, :max_req_tokens] = episode.request_tensor
                        input_ids[i, max_req_tokens:] = episode.response_tensor

                    t.step("reward_evaluation")

                    # Episode dropout logic (aligned with GRPO reference implementation)
                    # Drop entire batch if:
                    # 1. Reward variance is too low (including all 0s and all 1s)
                    # 2. Any response was truncated (didn't end with EOS)
                    rewards = [e.reward for e in episodes]
                    rewards_std = torch.std(torch.tensor(rewards))
                    is_low_variance = rewards_std < variance_threshold

                    # DAPO/GRPO aggressive truncation dropout: Drop entire batch if ANY
                    # response was truncated (stop_reason == "length"). This is intentional
                    # per DAPO paper recommendations - truncated responses provide incomplete
                    # signal and can hurt training. The dropout is batch-level rather than
                    # per-episode to maintain advantage computation correctness within groups.
                    num_truncated = sum(
                        1 for e in episodes if e.stop_reason == "length"
                    )
                    is_truncated = num_truncated > 0

                    # Record dropout metrics
                    n = len(episodes)
                    if enable_variance_dropout:
                        record_metric(
                            "main/continuous_rollouts/episodes_dropped/low_variance",
                            n if is_low_variance else 0,
                            Reduce.SUM,
                        )

                    if enable_truncation_dropout:
                        record_metric(
                            "main/continuous_rollouts/episodes_dropped/truncated",
                            num_truncated,
                            Reduce.SUM,
                        )

                    # Determine if we should drop this batch
                    should_drop = (
                        (enable_variance_dropout and is_low_variance) or
                        (enable_truncation_dropout and is_truncated)
                    )

                    record_metric(
                        "main/continuous_rollouts/episodes_dropped/total",
                        n if should_drop else 0,
                        Reduce.SUM,
                    )

                    if should_drop:
                        if is_low_variance:
                            logger.debug(
                                f"[DROPOUT] Dropping batch: low reward variance "
                                f"(std={rewards_std:.4f} < {variance_threshold})"
                            )
                        if is_truncated:
                            logger.debug(
                                f"[DROPOUT] Dropping batch: {num_truncated}/{n} episodes truncated"
                            )
                        del input_ids, episodes
                        continue

                    # Circuit breaker: if ALL evaluations failed, something is wrong
                    if eval_errors_this_batch == len(responses):
                        consecutive_errors += 1
                        logger.warning(
                            f"[CIRCUIT BREAKER] All {len(responses)} evaluations failed. "
                            f"Consecutive error batches: {consecutive_errors}/{max_consecutive_errors}"
                        )
                        if consecutive_errors >= max_consecutive_errors:
                            raise RuntimeError(
                                f"[ROLLOUT FAILURE] {consecutive_errors} consecutive batches with all evaluations failing. "
                                f"Environment actor appears to be unresponsive. Check container health."
                            )
                    else:
                        # Reset error counter on partial success
                        consecutive_errors = 0

                    # Compute ref_logprobs only if ref_model is configured
                    if ref_model is not None:
                        try:
                            ref_logprobs = await asyncio.wait_for(
                                ref_model.forward.route(input_ids, return_logprobs=True),
                                timeout=60.0,
                            )
                        except asyncio.TimeoutError:
                            logger.error("[ROLLOUT] Timeout waiting for ref_model.forward()")
                            record_metric("main/continuous_rollouts/ref_model_timeout", 1, Reduce.SUM)
                            continue

                        t.step("reference_model_calculate_logprobs")

                        if not isinstance(ref_logprobs, torch.Tensor):
                            raise TypeError(
                                f"ref_model.forward.route() returned {type(ref_logprobs)} instead of torch.Tensor"
                            )

                        for i, episode in enumerate(episodes):
                            episode.ref_logprobs = ref_logprobs[i]
                        del ref_logprobs

                    del input_ids

                    advantages = await compute_advantages.compute.call_one(episodes)
                    for episode, advantage in zip(episodes, advantages, strict=True):
                        episode.advantage = advantage
                        await replay_buffer.add.call_one(episode)

                        # Track token-based metrics (aligned with GRPO)
                        prompt_tokens = episode.completion.prompt_ids.shape[0]
                        response_tokens = episode.completion.token_ids.shape[0]

                        record_metric("episode/avg_prompt_tokens", prompt_tokens, Reduce.MEAN)
                        record_metric("episode/max_prompt_tokens", prompt_tokens, Reduce.MAX)
                        record_metric("episode/min_prompt_tokens", prompt_tokens, Reduce.MIN)
                        record_metric("episode/avg_response_tokens", response_tokens, Reduce.MEAN)
                        record_metric("episode/max_response_tokens", response_tokens, Reduce.MAX)
                        record_metric("episode/min_response_tokens", response_tokens, Reduce.MIN)
                        record_metric("episode/avg_reward", episode.reward, Reduce.MEAN)

                    rollout_count += 1
                    record_metric(
                        "main/continuous_rollouts/count_rollout_iterations", 1, Reduce.SUM
                    )
                    t.stop()

                except Exception as rollout_err:
                    # Catch any unexpected errors in rollout loop to prevent thread crash
                    logger.error(f"[ROLLOUT] Unexpected error in rollout loop: {rollout_err}")
                    record_metric("main/continuous_rollouts/unexpected_error", 1, Reduce.SUM)
                    consecutive_errors += 1
                    if consecutive_errors >= max_consecutive_errors:
                        raise RuntimeError(
                            f"[ROLLOUT FAILURE] {consecutive_errors} consecutive errors in rollout loop. "
                            f"Last error: {rollout_err}"
                        ) from rollout_err
                    # Brief pause before retry
                    await asyncio.sleep(1.0)
        except Exception as e:
            import traceback
            logger.error(f"[ROLLOUT FATAL] continuous_rollouts() crashed with error: {e}")
            logger.error(f"[ROLLOUT FATAL] Traceback:\n{traceback.format_exc()}")
            raise

    async def continuous_training():
        training_step = 0
        restart_tracer = True
        consecutive_empty_samples = 0
        # Configurable via environment variable for advanced tuning
        max_empty_samples_before_error = int(
            os.environ.get("FORGE_MAX_EMPTY_BUFFER_WAIT_S", "120")
        ) * 10  # Convert seconds to 0.1s intervals

        while max_steps == -1 or training_step < max_steps:
            if restart_tracer:
                t = Tracer("main_perf/continuous_training")
                t.start()
                restart_tracer = False

            batch = await replay_buffer.sample.call_one(
                curr_policy_version=training_step
            )
            if batch is None:
                consecutive_empty_samples += 1

                # Log warning at increasing intervals
                if consecutive_empty_samples == 10:  # 1 second
                    logger.warning(
                        f"[BUFFER STARVATION] Buffer empty for 1s at step {training_step}. "
                        f"Rollouts may be blocked during weight update."
                    )
                elif consecutive_empty_samples == 100:  # 10 seconds
                    logger.warning(
                        f"[BUFFER STARVATION] Buffer empty for 10s at step {training_step}. "
                        f"Consider increasing max_policy_age or rollout_threads."
                    )
                elif consecutive_empty_samples == 300:  # 30 seconds
                    logger.error(
                        f"[BUFFER STARVATION] Buffer empty for 30s at step {training_step}. "
                        f"This indicates a likely deadlock. Check generator weight updates."
                    )

                # Fail after max wait to prevent infinite hangs
                if consecutive_empty_samples >= max_empty_samples_before_error:
                    raise RuntimeError(
                        f"[BUFFER STARVATION DEADLOCK] Replay buffer has been empty for "
                        f"{consecutive_empty_samples * 0.1:.1f} seconds at training step {training_step}. "
                        f"This typically indicates that:\n"
                        f"  1. All policy replicas are blocked during weight updates\n"
                        f"  2. max_policy_age ({cfg.get('off_by_n', 1)}) is too aggressive\n"
                        f"  3. rollout_threads ({num_rollout_threads}) is insufficient\n"
                        f"Solutions:\n"
                        f"  - Increase 'off_by_n' in config (recommended: 2-3)\n"
                        f"  - Increase 'rollout_threads' in config\n"
                        f"  - Increase policy service 'num_replicas'\n"
                        f"  - Set FORGE_MAX_EMPTY_BUFFER_WAIT_S env var to increase timeout"
                    )

                logger.debug("Running out of batch, now waiting")
                await asyncio.sleep(0.1)
            else:
                # Reset starvation counter on successful sample
                consecutive_empty_samples = 0
                t.step("waiting_for_buffer")

                await trainer.train_step.call(batch)
                training_step += 1
                t.step("train_step")

                # Push and update weights every step
                await trainer.push_weights.call(training_step)
                t.step("push_weights")

                # Backpressure: Check buffer health for NEXT policy version.
                # Weight updates block all rollouts, so we need enough buffer headroom
                # to survive the blocking period without starving.
                # CRITICAL: Check training_step + 1 because after weight update,
                # episodes from current version will be evicted!
                buffer_health = await replay_buffer.health_check.call_one(
                    curr_policy_version=training_step + 1  # Check NEXT version survivability
                )
                required_surviving = buffer_health["required"] * 2  # Need 2x batch for safety margin
                surviving = buffer_health["surviving_after_eviction"]

                if surviving < required_surviving:
                    backpressure_start = time.time()
                    max_backpressure_wait = float(os.environ.get("FORGE_BACKPRESSURE_TIMEOUT_S", "30"))
                    logger.warning(
                        f"[BACKPRESSURE] Buffer low before weight update at step {training_step}. "
                        f"surviving={surviving}, required={required_surviving}. "
                        f"Waiting up to {max_backpressure_wait}s for more episodes."
                    )
                    record_metric("backpressure/triggered", 1, Reduce.SUM)

                    # Wait with exponential backoff
                    wait_interval = 0.5
                    while (time.time() - backpressure_start) < max_backpressure_wait:
                        await asyncio.sleep(wait_interval)
                        wait_interval = min(wait_interval * 1.5, 5.0)  # Cap at 5s intervals

                        buffer_health = await replay_buffer.health_check.call_one(
                            curr_policy_version=training_step + 1
                        )
                        if buffer_health["surviving_after_eviction"] >= required_surviving:
                            wait_duration = time.time() - backpressure_start
                            logger.info(f"[BACKPRESSURE] Buffer recovered after {wait_duration:.1f}s")
                            record_metric("backpressure/wait_duration_s", wait_duration, Reduce.MEAN)
                            break
                    else:
                        wait_duration = time.time() - backpressure_start
                        logger.warning(
                            f"[BACKPRESSURE] Buffer still low after {wait_duration:.1f}s. "
                            f"Proceeding with weight update to prevent complete stall."
                        )
                        record_metric("backpressure/timeout", 1, Reduce.SUM)
                t.step("backpressure_check")

                # Track weight update duration for monitoring
                weight_update_start = time.time()
                await policy.update_weights.fanout(training_step)
                weight_update_duration = time.time() - weight_update_start
                record_metric("training/weight_update_duration_s", weight_update_duration, Reduce.MEAN)
                if weight_update_duration > 20.0:
                    logger.warning(
                        f"[SLOW WEIGHT UPDATE] Step {training_step} took {weight_update_duration:.1f}s. "
                        f"Consider increasing off_by_n or policy replicas."
                    )
                    record_metric("training/slow_weight_update_count", 1, Reduce.SUM)
                t.step("update_weights")

                if training_step >= 2:
                    await drop_weights(training_step - 1)
                    t.step("drop_weights")

                t.stop()
                restart_tracer = True

                await mlogger.flush.call_one(training_step)

                # Periodic health monitoring every 10 steps
                if training_step % 10 == 0:
                    health_buffer = await replay_buffer.health_check.call_one(
                        curr_policy_version=training_step
                    )
                    record_metric("health/buffer_size", health_buffer["size"], Reduce.MAX)
                    record_metric("health/buffer_surviving", health_buffer["surviving_after_eviction"], Reduce.MAX)
                    record_metric("health/buffer_freshness_ratio", health_buffer["freshness_ratio"], Reduce.MEAN)
                    record_metric("health/buffer_required", health_buffer["required"], Reduce.MAX)

                    # Log reward actor health
                    try:
                        reward_health = await reward_actor.get_health_status.route()
                        record_metric("health/env_actors_healthy", reward_health["healthy_count"], Reduce.MAX)
                        record_metric("health/env_actors_total", reward_health["total_count"], Reduce.MAX)
                    except Exception as health_err:
                        logger.debug(f"Could not get reward actor health: {health_err}")

                    # Log training progress
                    record_metric("training/step", training_step, Reduce.MAX)
                    progress_pct = 100.0 * training_step / max_steps if max_steps > 0 else 0
                    record_metric("training/progress_pct", progress_pct, Reduce.MAX)

                    logger.info(
                        f"[HEALTH] Step {training_step}/{max_steps} ({progress_pct:.1f}%) | "
                        f"Buffer: {health_buffer['size']} ({health_buffer['surviving_after_eviction']} surviving) | "
                        f"Freshness: {health_buffer['freshness_ratio']:.2f}"
                    )

        print(
            f"Reached training limit ({max_steps} steps). Exiting continuous_training loop."
        )

    num_rollout_threads = cfg.get("rollout_threads", 1)
    print(f"Starting OpenEnv GRPO with {num_rollout_threads} rollout threads")

    # Callback to immediately report rollout task failures
    def rollout_task_done_callback(task):
        try:
            exc = task.exception()
            if exc is not None:
                import traceback
                logger.error(f"[ROLLOUT TASK FAILED] Rollout task crashed: {exc}")
                tb_str = "".join(
                    traceback.format_exception(type(exc), exc, exc.__traceback__)
                )
                logger.error(f"[ROLLOUT TASK FAILED] Traceback:\n{tb_str}")
        except asyncio.CancelledError:
            pass  # Task was cancelled, not an error

    # Start rollout tasks first
    rollout_tasks = []
    for _ in range(num_rollout_threads):
        task = asyncio.create_task(continuous_rollouts())
        task.add_done_callback(rollout_task_done_callback)
        rollout_tasks.append(task)

    # Start training immediately (no warmup)
    training_task = asyncio.create_task(continuous_training())

    try:
        await training_task
    except KeyboardInterrupt:
        print("Training interrupted by user")
    except Exception as e:
        import traceback

        print(f"Training failed with error: {e}")
        traceback.print_exc()
        raise
    finally:
        print("Shutting down...")
        shutdown_event.set()

        try:
            await asyncio.wait_for(
                asyncio.gather(*rollout_tasks, return_exceptions=True),
                timeout=5,
            )
        except asyncio.TimeoutError:
            for t in rollout_tasks:
                t.cancel()
            await asyncio.gather(*rollout_tasks, return_exceptions=True)

        training_task.cancel()

        # Cancel any pending circuit breaker restart tasks
        try:
            cancelled = await reward_actor.cancel_restart_tasks.route()
            if cancelled > 0:
                print(f"Cancelled {cancelled} pending circuit breaker restart tasks")
        except Exception as cancel_err:
            print(f"Warning: Error cancelling restart tasks: {cancel_err}")

        print(f"Cleaning up {len(env_actors)} environment Docker containers...")
        for i, env_actor in enumerate(env_actors):
            try:
                await env_actor.teardown.call_one()
                print(f"Environment Docker container {i + 1}/{len(env_actors)} stopped successfully")
            except Exception as teardown_error:
                print(f"Warning: Error during environment teardown {i + 1}: {teardown_error}")

        await shutdown()


if __name__ == "__main__":

    @parse
    def _main(cfg):
        os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"
        os.environ["NCCL_TIMEOUT_MS"] = "60000"
        os.environ["MONARCH_HOSTMESH_V1"] = "1"
        os.environ["TORCHSTORE_RDMA_ENABLED"] = "1"
        asyncio.run(main(cfg))

    _main()
