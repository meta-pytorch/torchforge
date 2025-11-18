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
import threading
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from typing import Any

import requests
import torch
import torch.nn.functional as F
import torchstore as ts
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
from forge.observability.metric_actors import get_or_create_metric_logger
from forge.observability.metrics import record_metric, Reduce
from forge.observability.perf_tracker import Tracer
from forge.types import LauncherConfig, ProvisionerConfig
from forge.util.config import parse
from forge.util.ops import compute_logprobs
from monarch.actor import endpoint
from omegaconf import DictConfig
from vllm import SamplingParams
from vllm.transformers_utils.tokenizer import get_tokenizer


# ============================================================================
# Server Management Functions (from main.py)
# ============================================================================


def start_openspiel_server(game_name: str, port: int):
    """Start OpenSpiel server in background process."""
    os.environ["OPENSPIEL_GAME"] = game_name

    import uvicorn
    from envs.openspiel_env.server.app import app

    print(f"[SERVER] Starting uvicorn for game '{game_name}' on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")


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
                print(f"[DEBUG] Killed existing process {pid} on port {port}")
            except ProcessLookupError:
                pass
        time.sleep(0.5)
        return True
    return False


# ============================================================================
# New Data Models (from v5)
# ============================================================================


@dataclass
class Episode:
    """Episode data for GRPO training (new structure)."""

    # Required fields (no defaults)
    episode_id: str
    all_token_ids: torch.Tensor  # All tokens in conversation
    logprobs: torch.Tensor  # Logprobs for all tokens
    response_mask: torch.Tensor  # Mask: 1 = assistant token, 0 = other
    reward: float

    # Optional fields (with defaults)
    task_name: str = "blackjack"
    generator_version: int = 0
    is_truncated: bool = False
    advantage: float | None = None
    ref_logprobs: torch.Tensor | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    message_log: list[dict[str, str]] | None = None


@dataclass
class EnvStepResult:
    """Result from environment step."""

    observation: dict[str, str]  # Next message: {"role": "user", "content": "..."}
    reward: float  # Reward for this step
    done: bool  # Episode ended?
    metadata: dict[str, Any] = field(default_factory=dict)


# ============================================================================
# TokenAccumulator (from v5)
# ============================================================================
from enum import Enum


class SanityCheckMode(Enum):
    """Validation mode for finalize()."""

    STRICT = "strict"
    DISABLE = "disable"


class TruncationReason(Enum):
    """Why an episode was truncated."""

    MAX_TURNS = "max_turns"
    AGENT_TOO_LONG = "agent_too_long"  # No EOS token or exceeded budget
    USER_TOO_LONG = "user_too_long"
    TOOL_TOO_LONG = "tool_too_long"


class TokenAccumulator:
    """
    Accumulates tokens during multi-turn RL rollouts with strict budget constraints.
    **IMPORTANT** Truncation behavior:
    - Agent response incomplete (no EOS): Tokens are dropped, nothing accumulated
    - User message too long: Truncated to fit, episode marked for dropping

    Why do we need this class?
    Problem: We need to track tokens as the conversation grows turn-by-turn.

    Naive approach 1 - Just tokenize each message independently:
        user_text = "Hello"
        user_tokens = tokenizer.encode(user_text)  # [9906]
        WRONG! -> Missing special tokens! Should be: [<|im_start|>, user, \n, 9906, <|im_end|>]

    Naive approach 2 - Tokenize a full conversation
        WRONG! ->  Qwen's template strips <think> tags from past messages, tokens don't match!
        Also, hard to create mask for the tokens that are traianble

    Solution - Delta tokenization:
        We tokenize [anchor + new_message] and slice off only the new tokens, where anchor is just a dummy message to allow the tokenizer to apply the correct message tokens, e.g. <|im_start|>:

        Turn 1, adding user message:
          tokenize([system, empty_user, new_user]) → [...system..., ...empty_user..., ...new_user...]
          slice from anchor_len → get only new_user tokens

        Turn 1, adding assistant:
          tokenize([system, empty_user, new_assistant]) → [...system..., ...empty_user..., ...new_assistant...]
          slice from anchor_len → get only new_assistant tokens

        The anchor ([system, empty_user]) stays constant, so the chat template applies
        consistent formatting to the new message, and we extract just those tokens.

    Usage:
        acc = TokenAccumulator(tokenizer, messages=[...], max_seq_len=2048, eos_token_id=...)

        acc.add_user_message("Hello")

        input_text = acc.format_prompt()

        response = model.generate(input_text, max_tokens=acc.get_remaining_budget())

        acc.add_assistant_response(response.text, response.token_ids)

        if acc.is_truncated:
            return None  # Drop episode

        return Episode(
            token_ids=acc.accumulated_tokens,
            response_mask=acc.response_mask,
            log_probs=acc.log_probs,
            messages=messages,
            ...)
    """

    # Class-level lock for thread-safe tokenizer access across all instances
    _tokenizer_lock = threading.Lock()

    def __init__(
        self,
        tokenizer,
        messages: list[dict],
        max_seq_len: int,
        eos_token_id: int,
        sanity_check_mode: SanityCheckMode = SanityCheckMode.STRICT,
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.eos_token_id = eos_token_id
        self.sanity_check_mode = sanity_check_mode

        # Core state
        self.messages = []
        self.accumulated_tokens = []
        self.response_mask = []
        self.logprobs = []

        # Truncation tracking
        self.is_truncated = False
        self.truncation_reason = None

        self._setup_anchor(messages)
        self._initialize_messages(messages)

    # ============ Public API ============

    def add_user_message(self, content: str) -> bool:
        """
        Add user message, truncating to fit budget if necessary.
        Returns False if truncated.
        """
        user_tokens = self._tokenize_delta({"role": "user", "content": content}, "user")
        budget = self.get_remaining_budget()
        original_len = len(user_tokens)
        user_tokens = self._truncate_to_fit(
            user_tokens, budget, TruncationReason.USER_TOO_LONG
        )

        if user_tokens:
            self.messages.append({"role": "user", "content": content})
            self._accumulate(user_tokens, is_response=False)

        return len(user_tokens) == original_len

    def add_assistant_response(
        self,
        response_text: str,
        response_token_ids: list[int],
        response_logprobs: list[float] | None = None,
    ) -> bool:
        print(f"[TokenAccumulator] ===== ENTERED add_assistant_response =====")
        """
        Add assistant response. Returns False if response was truncated (no EOS).
        Episode should be dropped if this returns False.
        """
        # Check for truncation (missing EOS)
        if response_token_ids and response_token_ids[-1] != self.eos_token_id:
            return self._mark_truncated(TruncationReason.AGENT_TOO_LONG)

        print(f"[TokenAccumulator] About to tokenize assistant response")
        print(f"[TokenAccumulator] Response text length: {len(response_text)} chars")
        print(f"[TokenAccumulator] Response token_ids length: {len(response_token_ids)} tokens")
        print(f"[TokenAccumulator] First 150 chars: {response_text[:150]}")

        # Safety check: If response is suspiciously long, warn and potentially truncate
        if len(response_text) > 10000:  # 10k chars is way too much for blackjack
            print(f"[TokenAccumulator] ⚠️  WARNING: Response text is {len(response_text)} chars - this may cause slow tokenization!")
            print(f"[TokenAccumulator] Last 150 chars: {response_text[-150:]}")

        message = {"role": "assistant", "content": response_text}
        assistant_tokens = self._tokenize_delta(message, "assistant")
        print(f"[TokenAccumulator] Tokenization complete, got {len(assistant_tokens)} tokens")

        # Check budget - reject if would exceed max_seq_len
        if len(assistant_tokens) > self.get_remaining_budget():
            return self._mark_truncated(TruncationReason.AGENT_TOO_LONG)
        else:
            self.messages.append({"role": "assistant", "content": response_text})

        # Map logprobs: vLLM returns content tokens only, align from end (EOS)
        if response_logprobs and len(response_logprobs) == len(response_token_ids):
            prefix_len = len(assistant_tokens) - len(response_token_ids)
            logprobs = [0.0] * prefix_len + response_logprobs
        else:
            logprobs = None

        self._accumulate(assistant_tokens, is_response=True, logprobs=logprobs)
        return True

    def format_prompt(self) -> str:
        """Format current conversation for generation."""
        with self._tokenizer_lock:
            return self.tokenizer.apply_chat_template(
                self.messages, add_generation_prompt=True, tokenize=False
            )

    def get_remaining_budget(self) -> int:
        """
        Get remaining tokens available for generation.

        We reserve generation_prompt_len tokens (e.g., "<|im_start|>assistant\n")
        because format_prompt() adds these when preparing input for the model.
        """
        used = len(self.accumulated_tokens) + self.generation_prompt_len
        return max(0, self.max_seq_len - used)

    def finalize(self) -> bool:
        """
        Validate final episode state.
        Returns True if valid, raises ValueError if critical issue detected.
        """
        self._check_structure()

        if self.sanity_check_mode != SanityCheckMode.DISABLE:
            self._check_ground_truth()

        return True

    # ============ Private Helpers ============

    def _setup_anchor(self, messages: list[dict]):
        """
        Setup anchor conversation for delta tokenization.

        Delta tokenization: Instead of re-tokenizing the full conversation after each message,
        we tokenize only the new message against a fixed anchor ([system, empty_user]). The dummy anchor is necessary to ensure that all special tokens are added.

        Computes key lengths for budget calculation:
        - anchor_len: tokens in [system, empty_user]
        - generation_prompt_len: tokens added by add_generation_prompt=True (e.g., "<|im_start|>assistant\n")
        - system_len: tokens in [system] alone
        """
        if not messages:
            raise ValueError("Must provide at least system message")

        system_msg = (
            messages[0]
            if messages[0]["role"] == "system"
            else {"role": "system", "content": ""}
        )

        # Anchor: [system, empty_user] - stays constant for consistent tokenization
        self.anchor = [system_msg, {"role": "user", "content": ""}]

        # Length of anchor without generation prompt
        anchor_tokens = self.tokenizer.apply_chat_template(
            self.anchor, add_generation_prompt=False, tokenize=True
        )
        self.anchor_len = len(anchor_tokens)

        # Length of anchor WITH generation prompt - difference is the prompt overhead
        anchor_with_gen = self.tokenizer.apply_chat_template(
            self.anchor, add_generation_prompt=True, tokenize=True
        )
        self.generation_prompt_len = len(anchor_with_gen) - self.anchor_len

        # System message length alone (for user message delta slicing), e.g. full[self.system_len:]
        system_tokens = self.tokenizer.apply_chat_template(
            [system_msg], add_generation_prompt=False, tokenize=True
        )
        self.system_len = len(system_tokens)

    def _initialize_messages(self, messages: list[dict]):
        """Initialize conversation with provided messages."""
        if not messages:
            return

        initial_tokens = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=False, tokenize=True
        )

        if len(initial_tokens) > self.max_seq_len:
            self._mark_truncated(TruncationReason.USER_TOO_LONG)
            initial_tokens = initial_tokens[: self.max_seq_len]

        self.messages = messages.copy()
        self._accumulate(initial_tokens, is_response=False)

    def _tokenize_delta(self, message: dict, role: str) -> list[int]:
        """Tokenize single message using anchor conversation."""
        if role == "assistant":
            temp = [self.anchor[0], {"role": "user", "content": ""}, message]
            offset = self.anchor_len
        else:  # user
            temp = [self.anchor[0], message]
            offset = self.system_len

        with self._tokenizer_lock:
            full = self.tokenizer.apply_chat_template(
                temp, add_generation_prompt=False, tokenize=True
            )
        return full[offset:]

    def _truncate_to_fit(
        self, tokens: list[int], available: int, reason: TruncationReason
    ) -> list[int]:
        """
        Truncate tokens to fit available space. Marks truncation if needed.
        Returns truncated tokens.
        """
        if len(tokens) > available:
            self._mark_truncated(reason)
            return tokens[: max(0, available)]
        return tokens

    def _accumulate(
        self, tokens: list[int], is_response: bool, logprobs: list[float] | None = None
    ):
        """Add tokens to accumulator."""
        self.accumulated_tokens.extend(tokens)
        self.response_mask.extend([int(is_response)] * len(tokens))
        self.logprobs.extend(logprobs or [0.0] * len(tokens))

    def _mark_truncated(self, reason: TruncationReason) -> bool:
        """Mark episode as truncated and return False."""
        self.is_truncated = True
        self.truncation_reason = reason
        return False

    def _check_structure(self):
        """Verify basic structural invariants."""
        assert (
            len(self.accumulated_tokens)
            == len(self.response_mask)
            == len(self.logprobs)
        )

        if len(self.accumulated_tokens) > self.max_seq_len:
            raise ValueError(
                f"Budget overflow: {len(self.accumulated_tokens)} > {self.max_seq_len}"
            )

    def _check_ground_truth(self):
        """
        Compare with ground truth tokenization.
        May fail with chat templates that modify history (e.g., Qwen deletes <think> tokens from older messages. This would cause a disparate between accumulated tokens and tokenized messages, since we accumulated the tokens with the <think> tokens).
        """
        ground_truth = self.tokenizer.apply_chat_template(
            self.messages, add_generation_prompt=False, tokenize=True
        )

        if len(self.accumulated_tokens) == len(ground_truth):
            return

        if self.sanity_check_mode == SanityCheckMode.STRICT:
            diff = len(ground_truth) - len(self.accumulated_tokens)
            raise ValueError(
                f"Token count mismatch: {len(self.accumulated_tokens)} accumulated vs "
                f"{len(ground_truth)} ground truth (diff: {diff}). "
                f"This happens when chat template modifies history."
            )


# ============================================================================
# BlackjackEnv (from v5)
# ============================================================================


class BlackjackEnv:
    """
    Minimal blackjack environment.

    Responsibilities:
    - Manage game state via OpenSpielEnv
    - Parse actions from text
    - Return next observation message
    - Compute rewards

    Does NOT:
    - Hold message history (rollout loop does this)
    - Tokenize (rollout loop does this)
    - Track cumulative tokens (rollout loop does this)
    """

    def __init__(self, server_url: str):
        self.server_url = server_url
        self.client = OpenSpielEnv(base_url=server_url)
        self.client._http.trust_env = False

        # Game state
        self.turn_count = 0
        self.has_invalid_action = False

    def reset(self) -> str:
        """
        Reset game and return initial user message.

        Returns:
            Initial observation text (NOT a dict, just the content string)
        """
        self.turn_count = 0
        self.has_invalid_action = False

        # Reset game
        result = self.client.reset()

        # Build initial observation
        return self._format_observation(result.observation)

    def step(self, action_text: str) -> EnvStepResult:
        """
        Execute action and return next observation.

        Args:
            action_text: The assistant's text response

        Returns:
            EnvStepResult with next observation message, reward, done
        """

        # Parse action
        action_name = self._parse_action(action_text)
        if action_name == "INVALID":
            self.has_invalid_action = True
            action_name = "STAND"  # Fallback
            record_metric("game/invalid_action_rate", 1, Reduce.MEAN)
        else:
            record_metric("game/invalid_action_rate", 0, Reduce.MEAN)

        # Execute in game
        action_id = 0 if action_name == "HIT" else 1
        result = self.client.step(
            OpenSpielAction(action_id=action_id, game_name="blackjack")
        )

        self.turn_count += 1

        # Compute reward
        if result.done:
            reward = self._compute_reward(result.reward)
            # Record game outcome metrics
            record_metric("game/games_played", 1, Reduce.SUM)
            record_metric("game/average_turns", self.turn_count, Reduce.MEAN)
            record_metric("game/win_rate", 1 if result.reward > 0 else 0, Reduce.MEAN)
            record_metric("game/env_reward", result.reward, Reduce.MEAN)
        else:
            reward = 0.0  # No intermediate rewards

        # Build next observation (if game continues)
        if result.done:
            observation = {"role": "user", "content": ""}  # Empty, game ended
        else:
            obs_text = self._format_observation(result.observation)
            observation = {"role": "user", "content": obs_text}

        return EnvStepResult(
            observation=observation,
            reward=reward,
            done=result.done,
            metadata={
                "turn_count": self.turn_count,
                "has_invalid_action": self.has_invalid_action,
                "env_reward": result.reward if result.done else 0.0,
            },
        )

    def _format_observation(self, observation) -> str:
        """Format game observation into text."""
        player_total = observation.metadata.get("player_total", "?")
        dealer_card = observation.metadata.get("dealer_card", "?")
        dealer_str = "Ace" if dealer_card == 1 else str(dealer_card)

        return f"Hand: {player_total}, Dealer: {dealer_str}"

    def _parse_action(self, text: str) -> str:
        """Parse action from assistant text."""
        text_lower = text.lower().strip()
        if text_lower.endswith("hit"):
            return "HIT"
        elif text_lower.endswith("stand"):
            return "STAND"
        else:
            return "INVALID"

    def _compute_reward(self, env_reward: float) -> float:
        """Compute final reward."""
        if env_reward > 0:  # Win
            return 3.0
        else:  # Loss or push
            return -1.0

    def close(self):
        """Clean up."""
        self.client.close()


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
        max_seq_len=max_seq_len,
        eos_token_id=tokenizer.eos_token_id,
        sanity_check_mode=SanityCheckMode.DISABLE,  # Disable in production for speed
    )

    try:
        # ============ Reset environment ============
        initial_obs = env.reset()
        accumulator.add_user_message(initial_obs)

        # ============ Multi-turn loop ============
        final_reward = 0.0
        turn_num = 0
        game_done = False
        generator_version = 0

        while not game_done and turn_num < max_turns:
            print(f"\n[do_single_rollout] Turn {turn_num}")

            # Check budget
            remaining = accumulator.get_remaining_budget()
            print(f"  Remaining budget: {remaining}")
            print(f"  Current tokens: {len(accumulator.accumulated_tokens)}")
            print(f"  Max seq len: {max_seq_len}")

            if remaining <= 0:
                print(f"  ❌ No budget left, breaking")
                break
            # Format prompt
            prompt = accumulator.format_prompt()

            # ============ Generate ============
            # Create sampling params with remaining budget to prevent exceeding max_seq_len
            print(f"  Calling vLLM with max_tokens={remaining}")
            sampling_params = SamplingParams(max_tokens=remaining)
            responses = await policy.generate.route(
                prompt, sampling_params=sampling_params
            )
            response = responses[0]
            print(f"  vLLM returned {len(response.token_ids)} tokens")
            print(f"  [DEBUG] About to get generator_version")

            generator_version = (
                response.generator_version
                if hasattr(response, "generator_version")
                else 0
            )
            print(f"  [DEBUG] Got generator_version: {generator_version}")

            # Extract logprobs from response
            print(f"  [DEBUG] About to extract logprobs")
            response_logprobs = (
                response.logprobs if hasattr(response, "logprobs") else None
            )
            print(f"  [DEBUG] Got logprobs: {response_logprobs is not None}")

            # ============ Add assistant response ============
            print(f"  [DEBUG] About to access response.text")
            response_text = response.text
            print(f"  [DEBUG] Got response.text, length: {len(response_text)}")
            print(f"  [DEBUG] About to access response.token_ids as list")
            response_token_ids_list = list(response.token_ids)  # Explicitly convert to list
            print(f"  [DEBUG] Got response.token_ids, length: {len(response_token_ids_list)}")

            print(f"  [DEBUG] About to call add_assistant_response")
            success = accumulator.add_assistant_response(
                response_text=response_text,
                response_token_ids=response_token_ids_list,
                response_logprobs=response_logprobs,
            )

            # If generation truncated, break
            if not success:
                print(f"  ❌ Generation failed, breaking")
                break

            # ============ Step environment ============
            result = env.step(action_text=response.text)
            final_reward = result.reward
            game_done = result.done
            turn_num += 1

            # ============ Add environment observation ============
            if not result.done:
                obs_text = result.observation["content"]
                success = accumulator.add_user_message(obs_text)

                # If env obs would exceed budget, break
                if not success:
                    break

        # Check if hit max_turns - just for metadata, accumulator tracks token truncation
        hit_max_turns = turn_num >= max_turns and not game_done

        # Optional: Validate token accumulation (useful in dev/staging)
        # accumulator.finalize()

        # Record metrics once at the end
        if accumulator.truncation_reason:
            record_metric(
                f"episode/truncated_{accumulator.truncation_reason.value}",
                1,
                Reduce.SUM,
            )
        record_metric(
            "episode/total_tokens", len(accumulator.accumulated_tokens), Reduce.MEAN
        )
        record_metric("episode/turns", turn_num, Reduce.MEAN)

        # ============ Create episode ============
        print(f"\n[do_single_rollout] Creating episode {game_id}")
        print(f"  Final tokens: {len(accumulator.accumulated_tokens)}")
        print(f"  Final mask: {len(accumulator.response_mask)}")
        print(f"  Final logprobs: {len(accumulator.logprobs)}")
        print(f"  Is truncated: {accumulator.is_truncated}")
        print(
            f"  Truncation reason: {accumulator.truncation_reason.value if accumulator.truncation_reason else None}"
        )
        print(f"  Hit max turns: {hit_max_turns}")
        print(f"  Max seq len: {max_seq_len}")

        if len(accumulator.accumulated_tokens) > max_seq_len:
            print(
                f"  ❌❌❌ EPISODE EXCEEDS max_seq_len by {len(accumulator.accumulated_tokens) - max_seq_len} tokens!"
            )

        return Episode(
            episode_id=game_id,
            task_name="blackjack",
            generator_version=generator_version,
            is_truncated=accumulator.is_truncated,
            all_token_ids=torch.tensor(
                accumulator.accumulated_tokens, dtype=torch.long
            ),
            logprobs=torch.tensor(accumulator.logprobs, dtype=torch.float),
            response_mask=torch.tensor(accumulator.response_mask, dtype=torch.float),
            reward=final_reward,
            message_log=accumulator.messages.copy(),
            metadata={
                "truncation_reason": (
                    accumulator.truncation_reason.value
                    if accumulator.truncation_reason
                    else None
                ),
                "hit_max_turns": hit_max_turns,
                "num_turns": turn_num,
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


@dataclass
class EnvironmentActor(ForgeActor):
    """Actor that manages tokenizer access."""

    model: str = "Qwen/Qwen3-1.7B"

    @endpoint
    def setup(self):
        self._tokenizer = get_tokenizer(self.model)
        print(f"EnvironmentActor initialized (model: {self.model})")

    @endpoint
    async def get_tokenizer(self):
        return self._tokenizer

    @endpoint
    async def pad_token(self):
        # Use pad_token_id if available, otherwise use eos_token_id
        if self._tokenizer.pad_token_id is not None:
            return self._tokenizer.pad_token_id
        else:
            return self._tokenizer.eos_token_id


# ============================================================================
# Training Functions (from main.py)
# ============================================================================


def collate(
    batches: list[list[Episode]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Collates a list of batches (groups) into inputs and targets.

    Args:
        batches: List of groups, where each group is a list of Episodes

    Returns:
        (inputs, targets) for training
    """
    inputs = []
    targets = []

    for batch in batches:
        # Find max sequence length in this batch
        max_len = max(len(e.all_token_ids) for e in batch)

        # Get pad_id from tokenizer (we'll use 0 as default)
        # In practice, this should come from the tokenizer
        pad_id = 0

        # Stack all tokens with padding
        all_tokens = []
        response_masks = []
        ref_logprobs_list = []
        advantages_list = []

        for e in batch:
            seq_len = len(e.all_token_ids)
            pad_len = max_len - seq_len

            # Pad tokens (right padding)
            padded_tokens = F.pad(e.all_token_ids, (0, pad_len), value=pad_id)
            all_tokens.append(padded_tokens)

            # Pad response mask (right padding with 0)
            padded_mask = F.pad(e.response_mask, (0, pad_len), value=0)
            response_masks.append(padded_mask)

            # Pad ref_logprobs (right padding with 0)
            padded_ref_logprobs = F.pad(e.ref_logprobs, (0, pad_len), value=0.0)
            ref_logprobs_list.append(padded_ref_logprobs)

            # Advantage is scalar
            advantages_list.append(e.advantage)

        # Stack everything
        all_tokens_tensor = torch.stack(all_tokens)  # [b, max_len]
        response_mask = torch.stack(response_masks)  # [b, max_len]
        ref_logprobs = torch.stack(ref_logprobs_list)  # [b, max_len]
        advantages = torch.tensor(advantages_list).unsqueeze(-1)  # [b, 1]

        # Input is all tokens
        input = {"tokens": all_tokens_tensor}

        # Target includes response tokens (all tokens), ref_logprobs, advantages, and mask
        target = {
            "response": all_tokens_tensor,  # Use all tokens as response
            "ref_logprobs": ref_logprobs,
            "advantages": advantages,
            "padding_mask": response_mask,
        }

        inputs.append(input)
        targets.append(target)

    return inputs, targets


def simple_grpo_loss(
    logits: torch.Tensor,
    response: torch.Tensor,
    ref_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    padding_mask: torch.Tensor,
    beta: float = 0.1,
) -> torch.Tensor:
    """
    Simple GRPO loss function.

    Args:
        logits: Model logits [b, s, v]
        response: Response tokens [b, s]
        ref_logprobs: Reference model logprobs [b, s]
        advantages: Advantages [b, 1]
        padding_mask: Mask for valid tokens [b, s]
        beta: KL penalty coefficient

    Returns:
        Loss scalar
    """
    logprobs: torch.Tensor = compute_logprobs(logits, response)
    kl = torch.exp(ref_logprobs - logprobs) - (ref_logprobs - logprobs) - 1
    per_token_policy_loss = torch.exp(logprobs - logprobs.detach()) * advantages
    per_token_loss = -(per_token_policy_loss - beta * kl)
    loss = (
        (per_token_loss * padding_mask).sum(dim=1)
        / (padding_mask.sum(dim=1).clamp(min=1.0))
    ).mean()
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

    # ---- Start OpenSpiel Server ---- #
    game_name = cfg.blackjack_env.game_name
    server_port = cfg.blackjack_env.server_port

    # Clean up any existing server on this port
    if kill_process_on_port(server_port):
        print(f"Cleaned up existing server on port {server_port}")

    print(f"Starting OpenSpiel server for game '{game_name}' on port {server_port}...")
    server_process = multiprocessing.Process(
        target=start_openspiel_server, args=(game_name, server_port)
    )
    server_process.start()

    # Wait for server to be ready
    print("Waiting for OpenSpiel server to be ready...")
    server_ready = False
    for i in range(30):  # Try for 30 seconds
        if not server_process.is_alive():
            print(f"[ERROR] Server process died unexpectedly!")
            print(f"[ERROR] Exit code: {server_process.exitcode}")
            raise RuntimeError(
                f"OpenSpiel server process crashed during startup (exit code: {server_process.exitcode})"
            )

        try:
            resp = requests.get(
                f"http://localhost:{server_port}/health",
                timeout=1,
                proxies={"http": None, "https": None},
            )
            print(f"[DEBUG] Health check attempt {i+1}: status={resp.status_code}")
            if resp.status_code == 200:
                server_ready = True
                print(f"✓ OpenSpiel server ready (took {i+1}s)")
                break
        except Exception as e:
            print(f"[DEBUG] Health check attempt {i+1} failed: {type(e).__name__}: {e}")
            time.sleep(1)

    if not server_ready:
        server_process.terminate()
        raise RuntimeError(f"OpenSpiel server never became ready on port {server_port}")

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

    # ---- Setup services ---- #
    env_actor_config = {
        "model": cfg.blackjack_env.model,
    }

    (
        env_actor,
        policy,
        trainer,
        replay_buffer,
        compute_advantages,
        ref_model,
    ) = await asyncio.gather(
        EnvironmentActor.options(**cfg.actors.blackjack_env).as_actor(
            **env_actor_config
        ),
        Generator.options(**cfg.services.policy).as_service(**cfg.policy),
        TitanTrainer.options(**cfg.actors.trainer).as_actor(
            **cfg.trainer, loss=simple_grpo_loss
        ),
        ReplayBuffer.options(**cfg.actors.replay_buffer).as_actor(
            **cfg.replay_buffer, collate=collate
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

    # ---- Warmup policy ---- #
    print("Warming up policy with test generation...")
    test_prompt = "Test prompt to warm up the model."
    try:
        test_response = await asyncio.wait_for(
            policy.generate.route(test_prompt), timeout=120.0
        )
        print(f"✓ Policy ready, test response: '{test_response[0].text[:50]}...'")
    except asyncio.TimeoutError:
        raise RuntimeError("Policy warmup timed out after 120s")
    except Exception as e:
        raise RuntimeError(f"Policy warmup failed: {e}")

    # ---- Test OpenSpiel server ---- #
    print("Testing OpenSpiel server connection...")
    test_env = OpenSpielEnv(base_url=cfg.blackjack_env.server_url)
    test_env._http.trust_env = False
    try:
        print(
            f"[DEBUG] Test env base_url={test_env._base}, timeout={test_env._timeout}"
        )
        print(f"[DEBUG] Test env trust_env={test_env._http.trust_env}")
        print(f"[DEBUG] Calling test_env.reset()...")
        test_result = test_env.reset()
        print(
            f"✓ OpenSpiel server test successful, legal_actions={test_result.observation.legal_actions}"
        )
        test_env.close()
    except Exception as e:
        print(f"[ERROR] OpenSpiel server test failed: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        raise RuntimeError(f"OpenSpiel server test failed: {e}")

    # ---- Core RL loops ---- #
    async def continuous_rollouts():
        """Main GRPO rollout loop using new architecture."""
        rollout_count = 0
        pad_id = await env_actor.pad_token.call_one()
        tokenizer = await env_actor.get_tokenizer.call_one()

        # Config
        server_url = cfg.blackjack_env.server_url
        max_seq_len = cfg.blackjack_env.max_seq_len
        max_turns = cfg.blackjack_env.max_turns
        group_size = cfg.group_size

        # Initial messages
        initial_messages = [
            {
                "role": "system",
                "content": "You are an expert BlackJack player. Output only 'HIT' or 'STAND'. You must think briefly. Do not think for long.",
            }
        ]

        while not shutdown_event.is_set():
            t = Tracer("main_perf/continuous_rollouts")
            t.start()

            # ============ Step 1: Create environments ============
            envs = [BlackjackEnv(server_url=server_url) for _ in range(group_size)]

            # ============ Step 2: Rollout group ============
            episodes = await do_group_rollout(
                envs=envs,
                policy=policy,
                tokenizer=tokenizer,
                max_seq_len=max_seq_len,
                max_turns=max_turns,
                messages=initial_messages,
            )

            t.step("play_games")

            # ============ Step 3: Filter groups (constant rewards) ============
            rewards = [e.reward for e in episodes]
            if len(set(rewards)) == 1:
                record_metric("groups/rate_dropped", 1, Reduce.MEAN)
                rollout_count += 1
                t.stop()
                continue
            record_metric("groups/rate_dropped", 0, Reduce.MEAN)

            # ============ Step 4: Compute ref_model ============
            print(f"\n[continuous_rollouts] Preparing ref_model input")
            max_len = max(len(e.all_token_ids) for e in episodes)
            print(f"  Max episode length: {max_len}")
            print(f"  Max seq len config: {max_seq_len}")

            for i, e in enumerate(episodes):
                print(
                    f"  Episode {i}: tokens={len(e.all_token_ids)}, truncated={e.is_truncated}"
                )
                if len(e.all_token_ids) > max_seq_len:
                    print(
                        f"    ❌ Episode {i} EXCEEDS max_seq_len by {len(e.all_token_ids) - max_seq_len}!"
                    )

            padded_tokens = [
                F.pad(
                    e.all_token_ids, (0, max_len - len(e.all_token_ids)), value=pad_id
                )
                for e in episodes
            ]
            input_ids = torch.stack(padded_tokens)

            print(f"  input_ids shape: {input_ids.shape}")
            print(f"  Calling ref_model with max_req_tokens=0")

            if input_ids.shape[1] > max_seq_len:
                print(
                    f"  ❌❌❌ input_ids seq_len={input_ids.shape[1]} EXCEEDS max_seq_len={max_seq_len}!"
                )
                print(f"  This will cause RoPE assertion error in the model!")

            ref_logprobs_padded = await ref_model.forward.route(
                input_ids, 0, return_logprobs=True
            )
            t.step("reference_model_calculate_logprobs")

            for i, episode in enumerate(episodes):
                seq_len = len(episode.all_token_ids)
                episode.ref_logprobs = ref_logprobs_padded[i, :seq_len]

            del ref_logprobs_padded, input_ids

            # ============ Step 5: Compute advantages ============
            advantages = await compute_advantages.compute.call_one(episodes)
            for episode, advantage in zip(episodes, advantages):
                episode.advantage = advantage

            # ============ Step 6: Episode-level acceptance ============
            accepted = []
            for episode in episodes:
                if episode.is_truncated and not cfg.accept_truncated:
                    record_metric("buffer/rate_rejected_truncated", 1, Reduce.MEAN)
                else:
                    record_metric("buffer/rate_rejected_truncated", 0, Reduce.MEAN)
                    accepted.append(episode)

            # ============ Step 7: Add to buffer ============
            for episode in accepted:
                await replay_buffer.add.call_one(episode)

            record_metric("buffer/episodes_accepted", len(accepted), Reduce.SUM)
            record_metric("buffer/episodes_generated", len(episodes), Reduce.SUM)
            record_metric(
                "buffer/acceptance_rate",
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

    num_rollout_threads = cfg.rollout_threads
    print(f"Starting GRPO with {num_rollout_threads} rollout threads")
    rollout_tasks = [
        asyncio.create_task(continuous_rollouts()) for _ in range(num_rollout_threads)
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

        # Shutdown OpenSpiel server
        print("Stopping OpenSpiel server...")
        server_process.terminate()
        server_process.join(timeout=2)
        if server_process.is_alive():
            print("⚠ Server didn't stop gracefully, killing...")
            server_process.kill()
            server_process.join(timeout=1)
        print("✓ OpenSpiel server stopped")


if __name__ == "__main__":

    @parse
    def _main(cfg):
        asyncio.run(main(cfg))

    _main()  # @parse grabs the cfg from CLI
