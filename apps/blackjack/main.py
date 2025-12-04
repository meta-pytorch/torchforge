# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Usage: python -m apps.blackjack.main_v2 --config apps/blackjack/qwen3_1_7b.yaml

import asyncio
import multiprocessing
import os
import signal
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache, partial
from typing import Any, Optional

import requests

import torch
import torch.nn.functional as F
import torchstore as ts

from apps.blackjack.blackjack_env import BlackjackEnv, EnvStepResult
from apps.blackjack.token_accumulator import (
    EpisodeData,
    TokenAccumulator,
    TruncationReason,
    ValidationMode,
)
from envs.openspiel_env import OpenSpielAction, OpenSpielEnv
from forge.actors._torchstore_utils import (
    get_dcp_whole_state_dict_key,
    get_param_prefix,
)
from forge.actors.generator import Generator
from forge.actors.reference_model import ReferenceModel
from forge.actors.replay_buffer import ReplayBuffer
from forge.actors.trainer import TitanTrainer
from forge.controller.actor import ForgeActor
from forge.controller.provisioner import init_provisioner, shutdown
from forge.data.common import CROSS_ENTROPY_IGNORE_IDX
from forge.observability.metric_actors import get_or_create_metric_logger
from forge.observability.metrics import record_metric, Reduce
from forge.observability.perf_tracker import Tracer
from forge.types import LauncherConfig, ProvisionerConfig
from forge.util.config import parse
from forge.util.ops import compute_logprobs, create_shifted_targets
from monarch.actor import endpoint
from omegaconf import DictConfig
from vllm import SamplingParams
from vllm.transformers_utils.tokenizer import get_tokenizer

# ============================================================================
# Server Management Functions for OpenSpiel / OpenEnv
# TODO: Written by claude, probably very messy
# ============================================================================


def start_openspiel_server(game_name: str, port: int):
    """Start OpenSpiel server in background process."""
    os.environ["OPENSPIEL_GAME"] = game_name

    import uvicorn
    from envs.openspiel_env.server.app import app

    print(f"[SERVER] Starting uvicorn for game '{game_name}' on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info", access_log=False)


def kill_process_on_port(port: int):
    """Kill any process using the specified port."""
    result = subprocess.run(
        ["lsof", "-ti", f":{port}"],
        capture_output=True,
        text=True,
        timeout=5,
    )
    if result.stdout.strip():
        pids = result.stdout.strip().split("\n")
        for pid in pids:
            try:
                os.kill(int(pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
        time.sleep(0.5)


def _wait_for_server_health(port: int, timeout: int = 30) -> bool:
    """Wait for server health check to pass."""
    for attempt in range(timeout):
        try:
            resp = requests.get(
                f"http://localhost:{port}/health",
                timeout=1,
                proxies={"http": None, "https": None},
            )
            if resp.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(1)
    return False


def start_servers(
    num_servers: int, base_port: int, game_name: str
) -> tuple[list, list]:
    """Start OpenSpiel servers and wait for them to be ready.

    Args:
        num_servers: Number of servers to start
        base_port: Base port (will use base_port, base_port+1, ...)
        game_name: Name of the game (e.g., "blackjack")

    Returns:
        (server_processes, server_ports)

    Raises:
        RuntimeError: If any server fails to start
    """
    server_processes = []
    server_ports = []

    # Start all servers
    for i in range(num_servers):
        port = base_port + i
        server_ports.append(port)

        kill_process_on_port(port)  # Clean up existing

        proc = multiprocessing.Process(
            target=start_openspiel_server, args=(game_name, port)
        )
        proc.start()
        server_processes.append(proc)

    # Wait for health checks
    time.sleep(1)  # Give servers time to start
    for i, port in enumerate(server_ports):
        if not _wait_for_server_health(port, timeout=30):
            # Cleanup and fail
            for proc in server_processes:
                proc.terminate()
            raise RuntimeError(f"Server on port {port} failed to start")

    print(f"✓ Started {num_servers} OpenSpiel server(s)")
    return server_processes, server_ports


def shutdown_servers(server_processes: list):
    """Shutdown all OpenSpiel servers gracefully."""
    for proc in server_processes:
        proc.terminate()
        proc.join(timeout=2)
        if proc.is_alive():
            proc.kill()
            proc.join(timeout=1)


# ============================================================================
# debugging
# ============================================================================


def print_episode_debug(episode, tokenizer, rollout_count: int):
    """Print detailed episode debug info using TokenAccumulator's visualization.

    Creates a temporary TokenAccumulator and populates it with episode data
    to reuse the colorized token stream display.
    """
    print(f"\n[ROLLOUT {rollout_count}] Episode Debug")
    print(
        f"Reward: {episode.reward:.2f}, Tokens: {len(episode.all_token_ids)}, "
        f"Trainable: {episode.response_mask.sum().item()}, Truncated: {episode.is_truncated}"
    )

    # Create a minimal TokenAccumulator just for visualization
    # We need to provide the required init params, but we'll override internals
    dummy_messages = [{"role": "system", "content": ""}]
    acc = TokenAccumulator(
        tokenizer=tokenizer,
        messages=dummy_messages,
        max_len=len(episode.all_token_ids),
        eos_id=tokenizer.eos_token_id,
        thinking=False,
        validation=ValidationMode.OFF,
    )

    # Replace internal state with episode data
    acc._tokens = episode.all_token_ids.tolist()
    acc._mask = episode.response_mask.tolist()
    acc._logprobs = [0.0] * len(episode.all_token_ids)  # Dummy logprobs
    acc.messages = episode.message_log if episode.message_log else []

    # Use TokenAccumulator's existing show_messages method
    acc.show_messages(max_chars=2000)


# ============================================================================
# Episode
# ============================================================================


@dataclass
class Episode:
    """Episode data for GRPO training (new structure)."""

    episode_id: str
    all_token_ids: torch.Tensor  # [seq_len]
    response_mask: torch.Tensor  # [seq_len]
    loss_mask: torch.Tensor  # [seq_len]
    reward: float

    task_name: str = "blackjack"
    policy_version: int = 0
    is_truncated: bool = False
    advantage: float | None = None
    logprobs: torch.Tensor | None = None  # [seq_len]
    ref_logprobs: torch.Tensor | None = None  # [seq_len]
    metadata: dict[str, Any] = field(default_factory=dict)
    message_log: list[dict[str, str]] | None = None


# ============================================================================
# Rollout Functions (from v5)
# ============================================================================


async def do_single_rollout(
    env: BlackjackEnv,
    policy,
    tokenizer,
    max_seq_len: int,
    max_turns: int,
    messages: list[dict],
    game_id: str | None = None,
) -> Episode:
    """
    Play one game and return one Episode.

    Uses TokenAccumulator for efficient multi-turn token management with BASE anchor pattern.

    Args:
        env: BlackjackEnv instance
        policy: Policy for generation
        tokenizer: Tokenizer with apply_chat_template
        max_seq_len: Maximum tokens for full conversation
        max_turns: Maximum game turns
        messages: Initial messages (e.g., [{"role": "system", "content": "..."}])
        game_id: Optional game ID

    Returns:
        Episode with accumulated tokens, masks, and logprobs
    """

    if game_id is None:
        game_id = str(uuid.uuid4())

    # Initialize TokenAccumulator with BASE anchor pattern
    accumulator = TokenAccumulator(
        tokenizer=tokenizer,
        messages=messages,
        max_len=max_seq_len,
        eos_id=tokenizer.eos_token_id,
        validation=ValidationMode.OFF,
        thinking=False,
    )

    try:
        # ============ Reset environment ============
        initial_obs = env.reset()
        accumulator.add_user(initial_obs)

        # ============ Multi-turn loop ============
        final_reward = 0.0
        turn_num = 0
        game_done = False
        policy_version = 0

        while not game_done and turn_num < max_turns:
            remaining_budget = accumulator.budget

            if remaining_budget <= 0:
                break

            # ============ Generate ============
            prompt = accumulator.format_prompt()
            sampling_params = SamplingParams(max_tokens=remaining_budget)
            responses = await policy.generate.route(
                prompt, sampling_params=sampling_params
            )
            response = responses[0]

            policy_version = response.generator_version

            # ============ Add assistant response ============
            response_logprobs = response.logprobs
            response_text = response.text
            response_token_ids_list = list(response.token_ids)

            # success means not truncated. We drop the entire response if truncated.
            success = accumulator.add_assistant(
                text=response_text,
                token_ids=response_token_ids_list,
                logprobs=response_logprobs,
            )

            # If generation truncated, break
            if not success:
                break

            # ============ Step environment ============
            result = env.step(action_text=response.text)
            final_reward = result.reward
            game_done = result.done
            turn_num += 1

            # ============ Add environment observation ============
            if not result.done:
                obs_text = result.observation["content"]
                success = accumulator.add_user(obs_text)

                # If env obs would exceed budget, break
                if not success:
                    break

        # ============ Get episode data ============
        episode_data = accumulator.get_data()

        # Record metrics
        if episode_data.truncation_reason:
            record_metric(
                f"episode/truncated_{episode_data.truncation_reason}",
                1,
                Reduce.SUM,
            )
        record_metric("episode/total_tokens", len(episode_data.token_ids), Reduce.MEAN)
        record_metric("episode/turns", turn_num, Reduce.MEAN)

        # ============ Create episode ============
        # Create loss_mask by shifting response_mask
        loss_mask_tensor = torch.roll(
            episode_data.response_mask, shifts=-1, dims=0
        ).float()
        loss_mask_tensor[-1] = 0.0  # Last position should not train

        return Episode(
            episode_id=game_id,
            task_name="blackjack",
            policy_version=policy_version,
            is_truncated=episode_data.is_truncated,
            all_token_ids=episode_data.token_ids,
            response_mask=episode_data.response_mask,
            loss_mask=loss_mask_tensor,
            reward=final_reward,
            logprobs=episode_data.logprobs,
            message_log=accumulator.messages.copy(),
            metadata={
                "truncation_reason": episode_data.truncation_reason,
                "num_turns": turn_num,
                "num_trainable_tokens": episode_data.response_mask.sum().item(),
                **(result.metadata if "result" in locals() else {}),
            },
        )

    finally:
        env.close()


async def do_group_rollout(
    envs: list[BlackjackEnv],
    policy,
    tokenizer,
    max_seq_len: int,
    max_turns: int,
    messages: list[dict],
) -> list[Episode]:
    """
    Rollout multiple games in parallel.

    Args:
        envs: List of N BlackjackEnv instances
        policy: Policy for generation
        tokenizer: Tokenizer for chat template
        max_seq_len: Episode-level token budget
        max_turns: Max turns per game
        messages: Initial messages for all games (e.g., [{"role": "system", ...}])

    Returns:
        List of N Episodes
    """
    tasks = [
        do_single_rollout(
            env=envs[i],
            policy=policy,
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            max_turns=max_turns,
            messages=messages,
            game_id=f"game_{i}_{uuid.uuid4().hex[:8]}",
        )
        for i in range(len(envs))
    ]

    episodes = await asyncio.gather(*tasks)
    return list(episodes)


# ============================================================================
# Helper Actors (from main.py)
# ============================================================================


@dataclass
class ComputeAdvantages(ForgeActor):
    """Compute advantages for a group of episodes."""

    @endpoint
    async def compute(self, group: list[Episode]) -> list[float]:
        """Compute advantages using reward standardization."""
        rewards = torch.tensor([[e.reward for e in group]])
        mean = rewards.mean(1, keepdim=True)
        std = rewards.std(1, keepdim=True)
        advantages = (rewards - mean) / (std + 1e-4)
        return advantages.squeeze(0).tolist()


# ============================================================================
# Training Functions (from main.py)
# ============================================================================


def collate(
    batches: list[list[Episode]],
    pad_id: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Collates a list of batches (groups) into inputs and targets.

    Args:
        batches: List of groups, where each group is a list of Episodes
        pad_id: Padding token ID from tokenizer

    Returns:
        (inputs, targets) for training
    """
    inputs = []
    targets = []

    for batch in batches:
        # Stack all tensors (pad to max length in batch)
        all_tokens = [e.all_token_ids for e in batch]
        all_tokens = torch.nn.utils.rnn.pad_sequence(
            all_tokens, batch_first=True, padding_value=pad_id
        )

        loss_masks = [e.loss_mask for e in batch]
        loss_masks = torch.nn.utils.rnn.pad_sequence(
            loss_masks, batch_first=True, padding_value=0.0
        )

        ref_logprobs = [e.ref_logprobs for e in batch]
        ref_logprobs = torch.nn.utils.rnn.pad_sequence(
            ref_logprobs, batch_first=True, padding_value=0.0
        )

        advantages = torch.tensor([e.advantage for e in batch]).unsqueeze(-1)  # [b, 1]

        # Create input and target dicts
        input = {"tokens": all_tokens}
        target = {
            "input_ids": all_tokens,  # For torch.roll in loss
            "loss_mask": loss_masks,  # Trainable positions
            "ref_logprobs": ref_logprobs,
            "advantages": advantages,
        }

        inputs.append(input)
        targets.append(target)

    return inputs, targets


# TODO: delete extensive debugging
# TODO: make KL clipping optional
def simple_grpo_loss(
    logits: torch.Tensor,  # [b, seq_len, vocab]
    input_ids: torch.Tensor,  # [b, seq_len]
    loss_mask: torch.Tensor,  # [b, seq_len] float
    ref_logprobs: torch.Tensor,  # [b, seq_len]
    advantages: torch.Tensor,  # [b, 1]
    beta: float = 0.1,
) -> torch.Tensor:
    """
    GRPO loss with KL clipping

    Args:
        logits: Model logits [b, seq_len, vocab_size]
        input_ids: Input token IDs [b, seq_len]
        loss_mask: Loss mask [b, seq_len] - 1.0 for trainable positions
        ref_logprobs: Reference logprobs [b, seq_len]
        advantages: Advantages [b, 1]
        beta: KL penalty coefficient

    Returns:
        Loss scalar
    """
    # Create targets using utility function
    targets = create_shifted_targets(input_ids, loss_mask)  # [b, seq_len]

    # Compute policy logprobs (ignore_index automatically zeros masked positions)
    logprobs = compute_logprobs(
        logits, targets, ignore_index=CROSS_ENTROPY_IGNORE_IDX
    )  # [b, seq_len] - masked positions already 0.0!

    # ========================================================================
    # LOGGING: Input validation
    # ========================================================================
    record_metric("loss_debug/batch_size", float(input_ids.shape[0]), Reduce.MEAN)
    record_metric("loss_debug/seq_len", float(input_ids.shape[1]), Reduce.MEAN)
    record_metric(
        "loss_debug/num_trainable_tokens", loss_mask.sum().item(), Reduce.MEAN
    )
    record_metric("loss_debug/targets_min", targets.float().min().item(), Reduce.MEAN)
    record_metric("loss_debug/targets_max", targets.float().max().item(), Reduce.MEAN)

    # ========================================================================
    # LOGGING: Logprobs statistics
    # ========================================================================
    # Mask logprobs for stats (only look at trainable positions)
    masked_logprobs = logprobs * loss_mask
    masked_ref_logprobs = ref_logprobs * loss_mask
    num_trainable = loss_mask.sum().clamp(min=1.0)

    record_metric(
        "loss_debug/logprobs_mean",
        (masked_logprobs.sum() / num_trainable).item(),
        Reduce.MEAN,
    )
    record_metric(
        "loss_debug/logprobs_min",
        logprobs[loss_mask.bool()].min().item() if num_trainable > 0 else 0.0,
        Reduce.MEAN,
    )
    record_metric(
        "loss_debug/logprobs_max",
        logprobs[loss_mask.bool()].max().item() if num_trainable > 0 else 0.0,
        Reduce.MEAN,
    )
    record_metric(
        "loss_debug/logprobs_std",
        logprobs[loss_mask.bool()].std().item() if num_trainable > 0 else 0.0,
        Reduce.MEAN,
    )

    record_metric(
        "loss_debug/ref_logprobs_mean",
        (masked_ref_logprobs.sum() / num_trainable).item(),
        Reduce.MEAN,
    )
    record_metric(
        "loss_debug/ref_logprobs_min",
        ref_logprobs[loss_mask.bool()].min().item() if num_trainable > 0 else 0.0,
        Reduce.MEAN,
    )
    record_metric(
        "loss_debug/ref_logprobs_max",
        ref_logprobs[loss_mask.bool()].max().item() if num_trainable > 0 else 0.0,
        Reduce.MEAN,
    )
    record_metric(
        "loss_debug/ref_logprobs_std",
        ref_logprobs[loss_mask.bool()].std().item() if num_trainable > 0 else 0.0,
        Reduce.MEAN,
    )

    # Logprob difference
    logprob_diff = ref_logprobs - logprobs
    masked_logprob_diff = logprob_diff * loss_mask
    record_metric(
        "loss_debug/logprob_diff_mean",
        (masked_logprob_diff.sum() / num_trainable).item(),
        Reduce.MEAN,
    )
    record_metric(
        "loss_debug/logprob_diff_min",
        logprob_diff[loss_mask.bool()].min().item() if num_trainable > 0 else 0.0,
        Reduce.MEAN,
    )
    record_metric(
        "loss_debug/logprob_diff_max",
        logprob_diff[loss_mask.bool()].max().item() if num_trainable > 0 else 0.0,
        Reduce.MEAN,
    )

    # KL divergence (masked positions are 0.0, so they don't contribute)
    # Following VERL's approach: clip log difference before exp for numerical stability
    # See: verl/trainer/ppo/core_algos.py kl_penalty_forward()
    logprob_diff_clipped = torch.clamp(logprob_diff, min=-20.0, max=20.0)
    kl = torch.exp(logprob_diff_clipped) - logprob_diff_clipped - 1
    # Clip final KL to prevent extreme values
    kl = torch.clamp(kl, min=-10.0, max=10.0)

    # ========================================================================
    # LOGGING: KL divergence statistics
    # ========================================================================
    masked_kl = kl * loss_mask
    record_metric(
        "loss_debug/kl_mean", (masked_kl.sum() / num_trainable).item(), Reduce.MEAN
    )
    record_metric(
        "loss_debug/kl_min",
        kl[loss_mask.bool()].min().item() if num_trainable > 0 else 0.0,
        Reduce.MEAN,
    )
    record_metric(
        "loss_debug/kl_max",
        kl[loss_mask.bool()].max().item() if num_trainable > 0 else 0.0,
        Reduce.MEAN,
    )
    record_metric(
        "loss_debug/kl_std",
        kl[loss_mask.bool()].std().item() if num_trainable > 0 else 0.0,
        Reduce.MEAN,
    )
    record_metric(
        "loss_debug/beta_times_kl_mean",
        (beta * masked_kl.sum() / num_trainable).item(),
        Reduce.MEAN,
    )

    # ========================================================================
    # LOGGING: Advantages statistics
    # ========================================================================
    record_metric("loss_debug/advantages_mean", advantages.mean().item(), Reduce.MEAN)
    record_metric("loss_debug/advantages_min", advantages.min().item(), Reduce.MEAN)
    record_metric("loss_debug/advantages_max", advantages.max().item(), Reduce.MEAN)
    record_metric("loss_debug/advantages_std", advantages.std().item(), Reduce.MEAN)

    # Policy loss
    per_token_policy_loss = torch.exp(logprobs - logprobs.detach()) * advantages
    per_token_loss = -(per_token_policy_loss - beta * kl)  # [b, seq_len]

    # ========================================================================
    # LOGGING: Per-token loss statistics
    # ========================================================================
    masked_policy_loss = per_token_policy_loss * loss_mask
    masked_per_token_loss = per_token_loss * loss_mask

    record_metric(
        "loss_debug/policy_loss_mean",
        (masked_policy_loss.sum() / num_trainable).item(),
        Reduce.MEAN,
    )
    record_metric(
        "loss_debug/policy_loss_min",
        (
            per_token_policy_loss[loss_mask.bool()].min().item()
            if num_trainable > 0
            else 0.0
        ),
        Reduce.MEAN,
    )
    record_metric(
        "loss_debug/policy_loss_max",
        (
            per_token_policy_loss[loss_mask.bool()].max().item()
            if num_trainable > 0
            else 0.0
        ),
        Reduce.MEAN,
    )

    record_metric(
        "loss_debug/per_token_loss_mean",
        (masked_per_token_loss.sum() / num_trainable).item(),
        Reduce.MEAN,
    )
    record_metric(
        "loss_debug/per_token_loss_min",
        per_token_loss[loss_mask.bool()].min().item() if num_trainable > 0 else 0.0,
        Reduce.MEAN,
    )
    record_metric(
        "loss_debug/per_token_loss_max",
        per_token_loss[loss_mask.bool()].max().item() if num_trainable > 0 else 0.0,
        Reduce.MEAN,
    )

    # Masked average (per sample, then batch average)
    loss = (
        (per_token_loss * loss_mask).sum(dim=1) / loss_mask.sum(dim=1).clamp(min=1.0)
    ).mean()

    # ========================================================================
    # LOGGING: Final loss
    # ========================================================================
    record_metric("loss_debug/final_loss", loss.item(), Reduce.MEAN)

    # ========================================================================
    # EMERGENCY DUMP: If any value is huge, save tensors to file
    # ========================================================================
    huge_threshold = 1000.0
    all_stats = [
        ("logprobs_mean", (masked_logprobs.sum() / num_trainable).item()),
        ("ref_logprobs_mean", (masked_ref_logprobs.sum() / num_trainable).item()),
        ("kl_mean", (masked_kl.sum() / num_trainable).item()),
        ("kl_max", kl[loss_mask.bool()].max().item() if num_trainable > 0 else 0.0),
        ("advantages_mean", advantages.mean().item()),
        ("advantages_max", advantages.max().item()),
        ("policy_loss_mean", (masked_policy_loss.sum() / num_trainable).item()),
        (
            "policy_loss_max",
            (
                per_token_policy_loss[loss_mask.bool()].max().item()
                if num_trainable > 0
                else 0.0
            ),
        ),
        ("per_token_loss_mean", (masked_per_token_loss.sum() / num_trainable).item()),
        (
            "per_token_loss_max",
            per_token_loss[loss_mask.bool()].max().item() if num_trainable > 0 else 0.0,
        ),
        ("final_loss", loss.item()),
    ]

    # for name, value in all_stats:
    #     if abs(value) > huge_threshold:
    #         # Save all tensors to file for debugging
    #         import datetime

    #         timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    #         dump_file = f"/tmp/grpo_loss_debug_{timestamp}.pt"
    #         torch.save(
    #             {
    #                 "logits": logits.cpu(),
    #                 "input_ids": input_ids.cpu(),
    #                 "targets": targets.cpu(),
    #                 "loss_mask": loss_mask.cpu(),
    #                 "logprobs": logprobs.cpu(),
    #                 "ref_logprobs": ref_logprobs.cpu(),
    #                 "advantages": advantages.cpu(),
    #                 "kl": kl.cpu(),
    #                 "per_token_policy_loss": per_token_policy_loss.cpu(),
    #                 "per_token_loss": per_token_loss.cpu(),
    #                 "loss": loss.cpu(),
    #                 "beta": beta,
    #                 "trigger_stat": name,
    #                 "trigger_value": value,
    #             },
    #             dump_file,
    #         )
    #         print(f"\n{'='*80}")
    #         print(f"⚠️  HUGE VALUE DETECTED: {name} = {value:.2f}")
    #         print(f"Dumped all tensors to: {dump_file}")
    #         print(f"{'='*80}\n")
    #         break  # Only dump once

    return loss


async def drop_weights(version: int):
    """Drop old weights from torchstore."""
    print(f"Dropping weights @ version {version}")
    start_time = time.perf_counter()
    prefix = get_param_prefix(version)
    matching_keys = await ts.keys(prefix)
    dcp_key = get_dcp_whole_state_dict_key(version)
    if dcp_key in matching_keys:
        dcp_handle = await ts.get(dcp_key)
        dcp_handle.drop()
    for key in matching_keys:
        await ts.delete(key)
    elapsed = time.perf_counter() - start_time
    print(f"Dropped weights @ version {version}, took {elapsed:.2f} seconds")


# ============================================================================
# Main Training Loop
# ============================================================================


async def main(cfg: DictConfig):
    """Main GRPO training loop with rollout and training processes."""

    # ---- Start OpenSpiel Servers ---- #
    server_processes, server_ports = start_servers(
        num_servers=cfg.get("rollout_threads", 1),
        base_port=cfg.blackjack_env.server_port,
        game_name=cfg.blackjack_env.game_name,
    )

    # ---- Global setups ---- #
    provisioner = None
    if cfg.get("provisioner", None) is not None:
        provisioner = await init_provisioner(
            ProvisionerConfig(launcher_config=LauncherConfig(**cfg.provisioner))
        )
    else:
        provisioner = await init_provisioner()

    metric_logging_cfg = cfg.metric_logging
    mlogger = await get_or_create_metric_logger(process_name="Controller")
    await mlogger.init_backends.call_one(metric_logging_cfg)

    # ---- Setup tokenizers ---- #
    # Create N tokenizers for N rollout threads (one per thread, no sharing)
    num_rollout_threads = cfg.rollout_threads
    tokenizers = [
        get_tokenizer(cfg.blackjack_env.model) for _ in range(num_rollout_threads)
    ]
    pad_id = (
        tokenizers[0].pad_token_id
        if tokenizers[0].pad_token_id is not None
        else tokenizers[0].eos_token_id
    )

    # Create collate function with pad_id
    collate_fn = partial(collate, pad_id=pad_id)

    # ---- Setup services ---- #
    (
        policy,
        trainer,
        replay_buffer,
        compute_advantages,
        ref_model,
    ) = await asyncio.gather(
        Generator.options(**cfg.services.policy).as_service(**cfg.policy),
        TitanTrainer.options(**cfg.actors.trainer).as_actor(
            **cfg.trainer, loss=simple_grpo_loss
        ),
        ReplayBuffer.options(**cfg.actors.replay_buffer).as_actor(
            **cfg.replay_buffer, collate=collate_fn
        ),
        ComputeAdvantages.options(**cfg.actors.compute_advantages).as_actor(),
        ReferenceModel.options(**cfg.services.ref_model).as_service(**cfg.ref_model),
    )

    max_steps = cfg.trainer.training.steps or -1

    print("All services initialized successfully!")
    shutdown_event = asyncio.Event()

    # Initialize torchstore
    trainer_num_procs = cfg.actors.trainer["procs"]
    trainer_host_mesh_name = cfg.actors.trainer["mesh_name"]
    trainer_hosts = provisioner.get_host_mesh(trainer_host_mesh_name)
    await ts.initialize(
        mesh=trainer_hosts.spawn_procs(per_host={"procs": trainer_num_procs}),
        strategy=ts.LocalRankStrategy(),
    )
    print("Torchstore successfully initialized with local rank strategy")

    # ---- Core RL loops ---- #
    async def continuous_rollouts(thread_id: int, tokenizer):
        """Main GRPO rollout loop using new architecture."""
        rollout_count = 0

        # Config - use dedicated server for this thread
        server_url = f"http://localhost:{server_ports[thread_id]}"
        max_seq_len = cfg.blackjack_env.max_seq_len
        max_turns = cfg.blackjack_env.max_turns
        group_size = cfg.group_size

        print(f"[Thread {thread_id}] Using server at {server_url}")

        # Initial messages
        initial_messages = [
            {
                "role": "system",
                "content": """You are an expert Blackjack player.

GOAL: Get a hand total closer to 21 than the dealer without going over 21 (busting).

RULES:
- Card values: Ace=1 or 11, Face cards (J,Q,K)=10, Number cards=face value
- If you go over 21, you bust and lose immediately
- The dealer plays after you and must hit until reaching 17+

ACTIONS:
- HIT: Take another card (increases your hand total)
- STAND: Keep your current hand and end your turn

WIN CONDITIONS:
- Your hand is closer to 21 than the dealer's final hand
- Dealer busts (goes over 21) and you don't
- You get exactly 21

IMPORTANT: You MUST output your action in the following format:
<answer>HIT</answer> or <answer>STAND</answer>""",
            }
        ]

        while not shutdown_event.is_set():
            t = Tracer("main_perf/continuous_rollouts")
            t.start()

            # ============ Step 1: Rollout group ============
            # TODO: currently done serially
            episodes = []
            for i in range(group_size):
                env = BlackjackEnv(server_url=server_url)
                game_id = f"game_{i}_{uuid.uuid4().hex[:8]}"

                episode = await do_single_rollout(
                    env=env,
                    policy=policy,
                    tokenizer=tokenizer,
                    max_seq_len=max_seq_len,
                    max_turns=max_turns,
                    messages=initial_messages,
                    game_id=game_id,
                )
                episodes.append(episode)

            t.step("play_games")

            # Print episode details every 10 rollouts
            if episodes and rollout_count % 10 == 0:
                print_episode_debug(episodes[0], tokenizer, rollout_count)

            # ============ Step 2: Filter groups (constant rewards) ============
            rewards = [e.reward for e in episodes]
            if len(set(rewards)) == 1:
                print(
                    f"[ROLLOUT {rollout_count}] ⚠️  DROPPED GROUP - All {len(episodes)} episodes have same reward: {rewards[0]}"
                )
                record_metric("groups/rate_dropped", 1, Reduce.MEAN)
                rollout_count += 1
                t.stop()
                continue
            record_metric("groups/rate_dropped", 0, Reduce.MEAN)

            # ============ Step 3: Compute ref_model ============
            max_len = max(len(e.all_token_ids) for e in episodes)

            # Pad input_ids and loss_masks
            padded_input_ids, padded_loss_masks = [], []
            for i, e in enumerate(episodes):
                pad_len = max_len - len(e.all_token_ids)

                padded_input_ids.append(
                    F.pad(e.all_token_ids, (0, pad_len), value=pad_id)
                )
                padded_loss_masks.append(F.pad(e.loss_mask, (0, pad_len), value=0.0))

            input_ids = torch.stack(padded_input_ids)  # [batch, max_len]
            loss_mask_batch = torch.stack(padded_loss_masks)  # [batch, max_len]

            # Call ref_model with loss_mask - returns [batch, max_len]
            ref_logprobs_padded = await ref_model.forward.route(
                input_ids, return_logprobs=True, loss_mask=loss_mask_batch
            )

            t.step("reference_model_calculate_logprobs")

            # Assign ref_logprobs to episodes (unpad to original length)
            for i, episode in enumerate(episodes):
                seq_len = len(episode.all_token_ids)
                episode.ref_logprobs = ref_logprobs_padded[i, :seq_len]  # [seq_len]

            del ref_logprobs_padded, input_ids, loss_mask_batch

            # ============ Step 4: Compute advantages ============
            advantages = await compute_advantages.compute.call_one(episodes)
            for episode, advantage in zip(episodes, advantages):
                episode.advantage = advantage

            # ============ Step 5: Episode-level acceptance ============
            accepted = []
            for episode in episodes:
                if episode.is_truncated and not cfg.accept_truncated:
                    record_metric("buffer/rate_rejected_truncated", 1, Reduce.MEAN)
                else:
                    record_metric("buffer/rate_rejected_truncated", 0, Reduce.MEAN)
                    accepted.append(episode)

            # ============ Step 6: Add to buffer ============
            for episode in accepted:
                await replay_buffer.add.call_one(episode)

            record_metric("buffer/episodes_accepted", len(accepted), Reduce.SUM)
            record_metric(
                "buffer/episode_acceptance_rate",
                len(accepted) / len(episodes) if episodes else 0,
                Reduce.MEAN,
            )

            rollout_count += 1
            record_metric(
                "main/continuous_rollouts/count_rollout_iterations", 1, Reduce.SUM
            )
            t.stop()

    async def continuous_training():
        """Training loop."""
        training_step = 0
        restart_tracer = True

        while max_steps == -1 or training_step < max_steps:
            if restart_tracer:
                t = Tracer("main_perf/continuous_training")
                t.start()
                restart_tracer = False

            batch = await replay_buffer.sample.call_one(
                curr_policy_version=training_step
            )
            if batch is None:
                await asyncio.sleep(0.1)
            else:
                t.step("waiting_for_buffer")
                print(f"[TRAINING] Step {training_step}: Starting training")

                inputs, targets = batch
                await trainer.train_step.call(inputs, targets)
                training_step += 1
                t.step("train_step")

                await trainer.push_weights.call(training_step)
                t.step("push_weights")

                await policy.update_weights.fanout(training_step)
                t.step("update_weights")

                if training_step >= 2:
                    await drop_weights(training_step - 1)
                    t.step("drop_weights")

                t.stop()
                restart_tracer = True

                # Flush metrics every training step
                await mlogger.flush.call_one(training_step)

        print(
            f"Reached training limit ({max_steps} steps). Exiting continuous_training loop."
        )

    print(f"Starting GRPO with {num_rollout_threads} rollout threads")
    rollout_tasks = [
        asyncio.create_task(continuous_rollouts(thread_id=i, tokenizer=tokenizers[i]))
        for i in range(num_rollout_threads)
    ]
    training_task = asyncio.create_task(continuous_training())

    try:
        await training_task
    except KeyboardInterrupt:
        print("Training interrupted by user")
    finally:
        print("Shutting down... (this may take a few seconds)")
        shutdown_event.set()

        # Cancel rollout tasks
        try:
            await asyncio.wait_for(
                asyncio.gather(*rollout_tasks, return_exceptions=True),
                timeout=5,
            )
        except asyncio.TimeoutError:
            print("Timeout waiting for rollouts; forcing cancellation...")
            for t in rollout_tasks:
                t.cancel()
            await asyncio.gather(*rollout_tasks, return_exceptions=True)

        # Cancel training task
        training_task.cancel()
        try:
            await asyncio.wait_for(training_task, timeout=2)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            pass

        # Shutdown forge actors/services
        print("Shutting down Forge actors...")
        try:
            await asyncio.wait_for(shutdown(), timeout=10)
            print("✓ Forge actors shut down")
        except asyncio.TimeoutError:
            print("⚠ Forge shutdown timed out after 10s, forcing exit...")

        # Shutdown OpenSpiel servers
        shutdown_servers(server_processes)


if __name__ == "__main__":

    @parse
    def _main(cfg):
        asyncio.run(main(cfg))

    _main()  # @parse grabs the cfg from CLI
