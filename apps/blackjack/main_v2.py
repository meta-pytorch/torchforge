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
import threading
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
# Server Management Functions (from main.py)
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
    all_token_ids: torch.Tensor  # [seq_len]
    response_mask: torch.Tensor  # [seq_len]
    loss_mask: torch.Tensor  # [seq_len]
    reward: float

    # Optional fields (with defaults)
    task_name: str = "blackjack"
    policy_version: int = 0
    is_truncated: bool = False
    advantage: float | None = None
    logprobs: torch.Tensor | None = None  # [seq_len]
    ref_logprobs: torch.Tensor | None = None  # [seq_len]
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
# TokenAccumulator
# ============================================================================


class ValidationMode(Enum):
    """Validation strictness."""

    STRICT = "strict"  # Raise on failures
    WARN = "warn"  # Print warnings
    OFF = "off"  # No validation


class TruncationReason(Enum):
    """Truncation reason."""

    USER_TOO_LONG = "user_too_long"
    ASSISTANT_TOO_LONG = "assistant_too_long"
    TOOL_TOO_LONG = "tool_too_long"
    MAX_NUM_TURNS = "max_num_turns"


@dataclass
class EpisodeData:
    """
    Episode data as tensors, ready for training.

    All tensors have shape (T,) where T is sequence length.
    """

    token_ids: torch.Tensor  # dtype=long
    response_mask: torch.Tensor  # dtype=bool
    logprobs: torch.Tensor  # dtype=float
    is_truncated: bool
    truncation_reason: Optional[str] = None


class TokenAccumulator:
    """
    Accumulate tokens for multi-turn RL episodes using vLLM tokens directly.

    ## Why Delta Tokenization?

    vLLM only returns assistant response tokens. We need the full conversation with
    chat template tokens for training. We can't re-tokenize because it's expensive
    and error-prone.

    **What we get from vLLM:**
    ```
    response_tokens = [791, 19, 374, 220, 2]  # ["The", "answer", "is", "4", "<eos>"]
    ```

    **What we need for training:**
    ```
    [1, 2, 3]                    # ["You", "are", "helpful"]         (not trainable)
    [10, 11, 12, 13]             # ["What", "is", "2+2", "?"]        (not trainable)
    [150, 123]                   # ["<|im_start|>", "assistant"]     (not trainable)
    [791, 19, 374, 220, 2]       # ["The", "answer", "is", "4", eos] (TRAINABLE!)
    [151]                        # ["<|im_end|>"]                    (not trainable, Qwen only)
    ```

    **Solution:** Use an anchor conversation [system, empty_user] that never changes.
    Tokenize new messages against it and extract deltas. For assistant responses,
    add generation prompt prefix and any model-specific suffix.

    ## Truncation Behavior

    - **add_user**: If truncated, adds partial message (truncated to fit budget)
    - **add_assistant**: If truncated, DROPS entire response (nothing added)
    - Once truncated, all subsequent adds will fail (return False)

    ## Usage

    ```python
    acc = TokenAccumulator(tok, [{"role": "system", "content": "Help"}], 2048, eos_id=2)

    # Add messages
    acc.add_user("What is 2+2?")
    prompt = acc.format_prompt()
    response = vllm_generate(prompt)
    acc.add_assistant(response.text, response.token_ids, response.logprobs)

    # Show what will be trained on
    acc.show_messages()

    # Get episode data as tensors
    episode = acc.get_data()
    # episode.token_ids: torch.Tensor (long)
    # episode.response_mask: torch.Tensor (bool, True = trainable)
    # episode.logprobs: torch.Tensor (float)
    ```

    Args:
        tokenizer: HuggingFace tokenizer with apply_chat_template
        messages: Initial messages (must include system message)
        max_len: Maximum sequence length
        eos_id: End-of-sequence token ID
        thinking: Enable <think> tags for Qwen models
        validation: Validation mode (STRICT, WARN, OFF)
    """

    def __init__(
        self,
        tokenizer,
        messages: list[dict],
        max_len: int,
        eos_id: int,
        thinking: bool = True,
        validation: ValidationMode = ValidationMode.STRICT,
    ) -> None:
        self._validate_init(tokenizer, messages, max_len, eos_id)

        self.tokenizer = tokenizer
        self.max_len = max_len
        self.eos_id = eos_id
        self.thinking = thinking
        self.validation = validation

        # State
        self.messages: list[dict] = []
        self._tokens: list[int] = []
        self._mask: list[bool] = []
        self._logprobs: list[float] = []
        self.truncated: bool = False
        self.truncation_reason: Optional[TruncationReason] = None

        # Track message boundaries for efficient validation
        # Each entry: (end_idx, role, should_end_with_eos)
        self._message_ends: list[tuple[int, str, bool]] = []

        # Thread safety
        self._lock = threading.Lock()

        # Setup
        self._setup_anchor(messages)
        self._init_messages(messages)

    def __repr__(self) -> str:
        status = f", truncated" if self.truncated else ""
        return f"TokenAccumulator({len(self._tokens)}/{self.max_len}{status})"

    @property
    def budget(self) -> int:
        """Remaining token budget."""
        return max(0, self.max_len - len(self._tokens) - self.gen_prompt_len)

    def add_user(self, content: str) -> bool:
        """
        Add user message. If truncated, adds partial message (truncated to fit).

        Returns:
            True if not truncated, False if truncated
        """
        if not isinstance(content, str):
            raise TypeError(f"content must be str, got {type(content)}")

        msg = {"role": "user", "content": content}

        # Tokenize [system, user] and extract delta
        with self._lock:
            full = self.tokenizer.apply_chat_template(
                [self.anchor[0], msg],
                add_generation_prompt=False,
                tokenize=True,
                enable_thinking=self.thinking,
            )
        # Extract user tokens by slicing off system prefix
        tokens = full[self.sys_len :]

        if not tokens:
            return True

        # Check budget
        budget = self.budget
        if budget <= 0:
            self._mark_truncated(TruncationReason.USER_TOO_LONG)
            return False

        # Truncate if needed (still adds partial)
        was_truncated = len(tokens) > budget
        if was_truncated:
            tokens = tokens[:budget]
            self._mark_truncated(TruncationReason.USER_TOO_LONG)

        self.messages.append(msg)
        self._add_tokens(tokens, trainable=False, role="user", ends_with_eos=False)

        return not was_truncated

    def add_assistant(
        self, text: str, token_ids: list[int], logprobs: Optional[list[float]] = None
    ) -> bool:
        """
        Add assistant response from vLLM. If truncated, DROPS entire response (nothing added).

        Args:
            text: Response text (for message log)
            token_ids: Token IDs from vLLM (must end with EOS)
            logprobs: Log probabilities (optional)

        Returns:
            False if truncated/invalid (response dropped), True if added successfully
        """
        # Type validation
        if not isinstance(text, str):
            raise TypeError(f"text must be str, got {type(text)}")
        if not isinstance(token_ids, list):
            raise TypeError(f"token_ids must be list, got {type(token_ids)}")

        # Must have tokens and end with EOS
        if not token_ids:
            return self._mark_truncated(TruncationReason.ASSISTANT_TOO_LONG)
        if token_ids[-1] != self.eos_id:
            return self._mark_truncated(TruncationReason.ASSISTANT_TOO_LONG)

        # Check budget: generation_prompt + response + suffix
        total_len = self.gen_prompt_len + len(token_ids) + len(self.suffix)
        if total_len > self.budget:
            return self._mark_truncated(TruncationReason.ASSISTANT_TOO_LONG)

        # Validate logprobs if provided
        if logprobs is not None:
            if not isinstance(logprobs, list):
                raise TypeError(f"logprobs must be list or None")
            if len(logprobs) != len(token_ids):
                raise ValueError(
                    f"logprobs length mismatch: {len(logprobs)} != {len(token_ids)}"
                )

        self.messages.append({"role": "assistant", "content": text})

        # Generation prompt (not trainable)
        self._add_tokens(
            self.gen_prompt_tokens,
            trainable=False,
            logprobs=[0.0] * len(self.gen_prompt_tokens),
            role="assistant_prompt",
            ends_with_eos=False,
        )

        # Response tokens (trainable)
        self._add_tokens(
            token_ids,
            trainable=True,
            logprobs=logprobs,
            role="assistant",
            ends_with_eos=True,
        )

        # Suffix if needed (not trainable)
        if self.suffix:
            self._add_tokens(
                self.suffix,
                trainable=False,
                logprobs=[0.0] * len(self.suffix),
                role="assistant_suffix",
                ends_with_eos=False,
            )

        return True

    def format_prompt(self) -> str:
        """Format conversation for vLLM generation."""
        with self._lock:
            return self.tokenizer.apply_chat_template(
                self.messages,
                add_generation_prompt=True,
                tokenize=False,
                enable_thinking=self.thinking,
            )

    def get_data(self) -> EpisodeData:
        """
        Convert to tensors, validate, and return episode data.

        Returns:
            EpisodeData with torch tensors

        Raises:
            AssertionError/ValueError: If validation fails in STRICT mode
        """
        # Convert to tensors
        token_ids = torch.tensor(self._tokens, dtype=torch.long)
        response_mask = torch.tensor(self._mask, dtype=torch.bool)
        logprobs = torch.tensor(self._logprobs, dtype=torch.float)

        # Validate on tensors
        if self.validation != ValidationMode.OFF:
            self._validate(token_ids, response_mask, logprobs)

        return EpisodeData(
            token_ids=token_ids,
            response_mask=response_mask,
            logprobs=logprobs,
            is_truncated=self.truncated,
            truncation_reason=(
                self.truncation_reason.value if self.truncation_reason else None
            ),
        )

    def show_messages(self, max_chars: int = 5000) -> None:
        """
        Show token stream with trainability highlighted.

        Uses colored text runs for readability (similar to tinker-cookbook's format_colorized).
        Groups consecutive tokens with same trainability and decodes together for proper
        multi-byte character handling.

        Args:
            max_chars: Maximum characters to show in decoded output (default: 5000)
        """
        print("=" * 80)
        print(f"TokenAccumulator: {len(self._tokens)}/{self.max_len} tokens")
        trainable_count = sum(self._mask)
        trainable_pct = 100 * trainable_count / len(self._tokens) if self._tokens else 0
        print(
            f"Trainable: {trainable_count}/{len(self._tokens)} ({trainable_pct:.1f}%)"
        )
        print("=" * 80)

        if not self._tokens:
            print("(no tokens)")
            print("=" * 80)
            return

        # Show messages list
        print("\nMessages:")
        for i, msg in enumerate(self.messages):
            role = msg["role"]
            content = msg["content"]
            preview = content[:100] + "..." if len(content) > 100 else content
            print(f"  [{i}] {role:10s} {preview!r}")

        # Show colorized token stream
        print("\nToken stream:")
        self._show_colorized_token_stream(max_chars)

        print("=" * 80)

    def _show_colorized_token_stream(self, max_chars: int) -> None:
        """
        Show full token stream with color coding by trainability.

        Groups consecutive tokens with same trainability into "runs" and decodes
        them together. This handles multi-byte characters correctly.
        """
        chunks = []
        current_ids = []
        current_trainable = None
        total_chars = 0

        def flush_run():
            nonlocal total_chars
            if not current_ids:
                return

            # Decode entire run at once
            with self._lock:
                decoded = self.tokenizer.decode(current_ids)

            # Check if we've exceeded max_chars
            if total_chars >= max_chars:
                return

            # Truncate if needed
            if total_chars + len(decoded) > max_chars:
                remaining = max_chars - total_chars
                decoded = decoded[:remaining] + "..."

            total_chars += len(decoded)

            # Color based on trainability
            if current_trainable:
                color_code = "\033[92m"  # Green for trainable
                symbol = "✓"
            else:
                color_code = "\033[90m"  # Gray for not trainable
                symbol = "·"

            # Escape special characters for display
            decoded_repr = repr(decoded)[1:-1]  # Remove outer quotes
            chunks.append(f"{color_code}{symbol} {decoded_repr}\033[0m")

        # Group tokens into runs
        for i in range(len(self._tokens)):
            trainable = self._mask[i]

            # Flush when trainability changes
            if trainable != current_trainable and current_ids:
                flush_run()
                current_ids = []

            current_ids.append(self._tokens[i])
            current_trainable = trainable

        # Flush final run
        flush_run()

        # Print runs
        if chunks:
            print("  " + " ".join(chunks))

        if total_chars >= max_chars:
            print(f"\n  (output truncated at {max_chars} chars)")

    def _show_colorized_tokens(self, start_idx: int, end_idx: int) -> None:
        """
        DEPRECATED: Old method, kept for compatibility.
        Use _show_colorized_token_stream instead.
        """
        pass

    # Internal helpers
    def _validate_init(
        self, tokenizer, messages: list[dict], max_len: int, eos_id: int
    ) -> None:
        """Validate initialization parameters."""
        if not hasattr(tokenizer, "apply_chat_template"):
            raise ValueError("Tokenizer must have apply_chat_template method")
        if not messages:
            raise ValueError("Must provide at least a system message")
        if not isinstance(messages, list):
            raise TypeError(f"messages must be list, got {type(messages)}")
        for i, msg in enumerate(messages):
            if not isinstance(msg, dict):
                raise TypeError(f"Message {i} must be dict")
            if "role" not in msg or "content" not in msg:
                raise ValueError(f"Message {i} missing 'role' or 'content'")
        if not isinstance(max_len, int) or max_len <= 0:
            raise ValueError(f"max_len must be positive int, got {max_len}")
        if not isinstance(eos_id, int):
            raise TypeError(f"eos_id must be int, got {type(eos_id)}")

    def _setup_anchor(self, msgs: list[dict]) -> None:
        """
        Setup anchor for delta tokenization and compute suffix.

        The suffix is anything after EOS in the chat template. We create a test
        conversation with EOS and extract any tokens that follow it.
        """
        sys = (
            msgs[0]
            if msgs[0]["role"] == "system"
            else {"role": "system", "content": ""}
        )
        self.anchor = [sys, {"role": "user", "content": ""}]

        with self._lock:
            # Compute generation prompt
            without = self.tokenizer.apply_chat_template(
                self.anchor,
                add_generation_prompt=False,
                tokenize=True,
                enable_thinking=self.thinking,
            )
            with_gen = self.tokenizer.apply_chat_template(
                self.anchor,
                add_generation_prompt=True,
                tokenize=True,
                enable_thinking=self.thinking,
            )
            self.gen_prompt_tokens = with_gen[len(without) :]
            self.gen_prompt_len = len(self.gen_prompt_tokens)

            # Compute system length
            sys_tokens = self.tokenizer.apply_chat_template(
                [sys],
                add_generation_prompt=False,
                tokenize=True,
                enable_thinking=self.thinking,
            )
            self.sys_len = len(sys_tokens)

            # Compute suffix by tokenizing a test conversation
            test_conv = [
                sys,
                {"role": "user", "content": "test"},
                {"role": "assistant", "content": "response"},
            ]
            test_tokens = self.tokenizer.apply_chat_template(
                test_conv,
                add_generation_prompt=False,
                tokenize=True,
                enable_thinking=self.thinking,
            )

            # Find last EOS
            eos_idx = -1
            for i in range(len(test_tokens) - 1, -1, -1):
                if test_tokens[i] == self.eos_id:
                    eos_idx = i
                    break

            # Extract suffix (everything after EOS, or empty if nothing)
            if eos_idx >= 0 and eos_idx < len(test_tokens) - 1:
                self.suffix = test_tokens[eos_idx + 1 :]
            else:
                self.suffix = []

    def _init_messages(self, msgs: list[dict]) -> None:
        """Initialize with starting messages."""
        if not msgs:
            return

        with self._lock:
            tokens = self.tokenizer.apply_chat_template(
                msgs,
                add_generation_prompt=False,
                tokenize=True,
                enable_thinking=self.thinking,
            )

        if len(tokens) > self.max_len:
            self._mark_truncated(TruncationReason.USER_TOO_LONG)
            tokens = tokens[: self.max_len]

        self.messages = msgs.copy()
        self._add_tokens(tokens, trainable=False, role="initial", ends_with_eos=False)

    def _add_tokens(
        self,
        tokens: list[int],
        trainable: bool,
        logprobs: Optional[list[float]] = None,
        role: str = "",
        ends_with_eos: bool = False,
    ) -> None:
        """Add tokens to parallel arrays and track message boundary."""
        if not tokens:
            return

        self._tokens.extend(tokens)
        self._mask.extend([trainable] * len(tokens))
        self._logprobs.extend(logprobs if logprobs else [0.0] * len(tokens))

        # Track message end for validation
        end_idx = len(self._tokens) - 1
        self._message_ends.append((end_idx, role, ends_with_eos))

    def _mark_truncated(self, reason: TruncationReason) -> bool:
        """Mark as truncated."""
        self.truncated = True
        self.truncation_reason = reason
        return False

    def _validate(
        self,
        token_ids: torch.Tensor,
        response_mask: torch.Tensor,
        logprobs: torch.Tensor,
    ) -> None:
        """
        Run validation checks on tensors.

        Args:
            token_ids: Token IDs tensor (shape: T)
            response_mask: Response mask tensor (shape: T)
            logprobs: Log probabilities tensor (shape: T)
        """
        # Check 1: Shapes match
        if not (token_ids.shape == response_mask.shape == logprobs.shape):
            raise AssertionError(
                f"Shape mismatch: token_ids={token_ids.shape}, "
                f"mask={response_mask.shape}, logprobs={logprobs.shape}"
            )

        # Check 2: Budget not exceeded
        if len(token_ids) > self.max_len:
            raise ValueError(f"Budget overflow: {len(token_ids)} > {self.max_len}")

        # Check 3: Message boundaries are correct
        for end_idx, role, should_end_with_eos in self._message_ends:
            if should_end_with_eos:
                # Token at end_idx should be eos_id
                if token_ids[end_idx].item() != self.eos_id:
                    msg = f"{role} at {end_idx} has token {token_ids[end_idx].item()}, expected EOS {self.eos_id}"
                    if self.validation == ValidationMode.STRICT:
                        raise ValueError(msg)
                    print(f"WARNING: {msg}")

                # For assistant: end_idx should be trainable
                if role == "assistant" and not response_mask[end_idx].item():
                    msg = f"Assistant EOS at {end_idx} is not trainable"
                    if self.validation == ValidationMode.STRICT:
                        raise ValueError(msg)
                    print(f"WARNING: {msg}")

                # Token after EOS should not be trainable
                if end_idx + 1 < len(token_ids) and response_mask[end_idx + 1].item():
                    msg = (
                        f"Token after EOS at {end_idx+1} is trainable (should be False)"
                    )
                    if self.validation == ValidationMode.STRICT:
                        raise ValueError(msg)
                    print(f"WARNING: {msg}")

        # Check 4: Prefix consistency (incremental == full tokenization)
        # DISABLED: Qwen always adds think tags to LAST assistant message only,
        # but in incremental accumulation every assistant response IS the last one
        # at the time we add it. This causes mismatches:
        # - thinking=True: missing 4 tokens (last gets think tags in full tokenization)
        # - thinking=False: extra 4 tokens (first doesn't get think tags in full tokenization)
        # This is expected behavior for Qwen and not a bug.
        #
        # with self._lock:
        #     full_tokens = self.tokenizer.apply_chat_template(
        #         self.messages, add_generation_prompt=False, tokenize=True, enable_thinking=self.thinking
        #     )
        #
        # accumulated_len = len(token_ids)
        # expected_len = len(full_tokens)
        #
        # if accumulated_len != expected_len:
        #     msg = (
        #         f"Prefix consistency failed: "
        #         f"accumulated={accumulated_len} tokens, "
        #         f"expected={expected_len}"
        #     )
        #     if self.validation == ValidationMode.STRICT:
        #         raise AssertionError(msg)
        #     print(f"WARNING: {msg}")


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
        action_name, error_type = self._parse_action(action_text)

        # Track invalid actions
        is_invalid = action_name == "INVALID"
        if is_invalid:
            self.has_invalid_action = True
            action_name = "STAND"  # Treat invalid as STAND
            record_metric("game/invalid_action_rate", 1, Reduce.MEAN)

            if error_type == "NO_TAGS":
                print(f"[ENV] ⚠️  INVALID action: Missing <answer> tags!")
                print(f"[ENV]     Text: '{action_text}...'")
                record_metric("game/missing_answer_tags", 1, Reduce.SUM)
            elif error_type == "INVALID_CONTENT":
                print(f"[ENV] ⚠️  INVALID action: Bad content in <answer> tags!")
                print(f"[ENV]     Text: '{action_text}...'")
                record_metric("game/invalid_answer_content", 1, Reduce.SUM)

            print(f"[ENV]     Treating as STAND")
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

            # Apply penalty for invalid action format
            if self.has_invalid_action:
                reward -= 10.0  # Penalty for not ending with HIT/STAND
                record_metric("game/invalid_action_penalty", 1, Reduce.SUM)

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

    def _parse_action(self, text: str) -> tuple[str, str]:
        """Parse action from assistant text using <answer> tags.

        Returns:
            (action, error_type): action is "HIT", "STAND", or "INVALID"
                                  error_type is "" for valid, "NO_TAGS" or "INVALID_CONTENT"
        """
        import re

        # Try to extract content from <answer> tags
        match = re.search(
            r"<answer>\s*(.*?)\s*</answer>", text, re.IGNORECASE | re.DOTALL
        )

        if match:
            answer = match.group(1).strip().upper()
            if answer == "HIT":
                return ("HIT", "")
            elif answer == "STAND":
                return ("STAND", "")
            else:
                # Has <answer> tags but invalid content
                return ("INVALID", "INVALID_CONTENT")
        else:
            # No <answer> tags found
            return ("INVALID", "NO_TAGS")

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
            # Check budget
            remaining = accumulator.budget

            if remaining <= 0:
                break

            # Format prompt
            prompt = accumulator.format_prompt()

            # ============ Generate ============
            # Create sampling params with remaining budget to prevent exceeding max_seq_len
            sampling_params = SamplingParams(max_tokens=remaining)
            responses = await policy.generate.route(
                prompt, sampling_params=sampling_params
            )
            response = responses[0]

            policy_version = response.generator_version

            # Extract logprobs from response
            response_logprobs = (
                response.logprobs if hasattr(response, "logprobs") else None
            )

            # ============ Add assistant response ============
            response_text = response.text

            response_token_ids_list = list(
                response.token_ids
            )  # Explicitly convert to list

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

        # Check if hit max_turns - just for metadata, accumulator tracks token truncation
        hit_max_turns = turn_num >= max_turns and not game_done

        # ============ Get validated episode data ============
        episode_data = accumulator.get_data()

        # Record metrics once at the end
        if episode_data.truncation_reason:
            record_metric(
                f"episode/truncated_{episode_data.truncation_reason}",
                1,
                Reduce.SUM,
            )
        record_metric("episode/total_tokens", len(episode_data.token_ids), Reduce.MEAN)
        record_metric("episode/turns", turn_num, Reduce.MEAN)

        # ============ Create episode ============
        # Create loss_mask by shifting response_mask using torch.roll
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
                "hit_max_turns": hit_max_turns,
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


@dataclass
class EnvironmentActor(ForgeActor):
    """Actor that manages tokenizer access."""

    model: str = "Qwen/Qwen3-1.7B"

    @endpoint
    def setup(self):
        self._tokenizer = get_tokenizer(self.model)

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


def simple_grpo_loss(
    logits: torch.Tensor,  # [b, seq_len, vocab]
    input_ids: torch.Tensor,  # [b, seq_len]
    loss_mask: torch.Tensor,  # [b, seq_len] float
    ref_logprobs: torch.Tensor,  # [b, seq_len]
    advantages: torch.Tensor,  # [b, 1]
    beta: float = 0.1,
) -> torch.Tensor:
    """
    GRPO loss with proper next-token prediction using torch.roll.

    Per-sequence normalization: Each sequence's loss is averaged by its own
    trainable token count, then averaged across the batch.

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

    for name, value in all_stats:
        if abs(value) > huge_threshold:
            # Save all tensors to file for debugging
            import datetime

            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            dump_file = f"/tmp/grpo_loss_debug_{timestamp}.pt"
            torch.save(
                {
                    "logits": logits.cpu(),
                    "input_ids": input_ids.cpu(),
                    "targets": targets.cpu(),
                    "loss_mask": loss_mask.cpu(),
                    "logprobs": logprobs.cpu(),
                    "ref_logprobs": ref_logprobs.cpu(),
                    "advantages": advantages.cpu(),
                    "kl": kl.cpu(),
                    "per_token_policy_loss": per_token_policy_loss.cpu(),
                    "per_token_loss": per_token_loss.cpu(),
                    "loss": loss.cpu(),
                    "beta": beta,
                    "trigger_stat": name,
                    "trigger_value": value,
                },
                dump_file,
            )
            print(f"\n{'='*80}")
            print(f"⚠️  HUGE VALUE DETECTED: {name} = {value:.2f}")
            print(f"Dumped all tensors to: {dump_file}")
            print(f"{'='*80}\n")
            break  # Only dump once

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

    # ---- Start Multiple OpenSpiel Servers (one per rollout thread) ---- #
    game_name = cfg.blackjack_env.game_name
    base_server_port = cfg.blackjack_env.server_port
    num_rollout_threads = cfg.get("rollout_threads", 1)

    # Start one server per rollout thread to avoid race conditions
    server_processes = []
    server_ports = []

    for i in range(num_rollout_threads):
        server_port = base_server_port + i
        server_ports.append(server_port)

        # Clean up any existing server on this port
        if kill_process_on_port(server_port):
            print(f"Cleaned up existing server on port {server_port}")

        print(
            f"Starting OpenSpiel server {i} for game '{game_name}' on port {server_port}..."
        )
        server_process = multiprocessing.Process(
            target=start_openspiel_server, args=(game_name, server_port)
        )
        server_process.start()
        server_processes.append(server_process)

    # Wait for all servers to be ready
    print(f"Waiting for {num_rollout_threads} OpenSpiel servers to be ready...")
    all_ready = True
    for i, server_port in enumerate(server_ports):
        server_ready = False
        for attempt in range(30):  # Try for 30 seconds per server
            if not server_processes[i].is_alive():
                print(f"[ERROR] Server {i} process died unexpectedly!")
                print(f"[ERROR] Exit code: {server_processes[i].exitcode}")
                all_ready = False
                break

            try:
                resp = requests.get(
                    f"http://localhost:{server_port}/health",
                    timeout=1,
                    proxies={"http": None, "https": None},
                )
                if resp.status_code == 200:
                    server_ready = True
                    print(
                        f"✓ OpenSpiel server {i} ready on port {server_port} (took {attempt+1}s)"
                    )
                    break
            except Exception as e:
                if attempt == 0:
                    print(
                        f"[DEBUG] Server {i} health check attempt {attempt+1} failed: {type(e).__name__}"
                    )
                time.sleep(1)

        if not server_ready:
            print(f"[ERROR] Server {i} never became ready on port {server_port}")
            all_ready = False
            break

    if not all_ready:
        # Clean up all servers and exit
        for process in server_processes:
            process.terminate()
        raise RuntimeError("Failed to start all OpenSpiel servers")

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

    # First, initialize env_actor to get pad_id
    env_actor = await EnvironmentActor.options(**cfg.actors.blackjack_env).as_actor(
        **env_actor_config
    )
    pad_id = await env_actor.pad_token.call_one()

    # Create collate function with pad_id
    collate_fn = partial(collate, pad_id=pad_id)

    # Now initialize remaining services
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

    # ---- Test OpenSpiel servers ---- #
    print("Testing OpenSpiel server connections...")
    for i, server_port in enumerate(server_ports):
        test_url = f"http://localhost:{server_port}"
        test_env = OpenSpielEnv(base_url=test_url)
        test_env._http.trust_env = False
        try:
            test_result = test_env.reset()
            print(
                f"✓ Server {i} test successful (port {server_port}), legal_actions={test_result.observation.legal_actions}"
            )
            test_env.close()
        except Exception as e:
            print(f"[ERROR] Server {i} test failed: {type(e).__name__}: {e}")
            import traceback

            traceback.print_exc()
            # Clean up all servers
            for process in server_processes:
                process.terminate()
            raise RuntimeError(f"OpenSpiel server {i} test failed: {e}")

    # ---- Core RL loops ---- #
    async def continuous_rollouts(thread_id: int):
        """Main GRPO rollout loop using new architecture."""
        rollout_count = 0
        pad_id = await env_actor.pad_token.call_one()
        tokenizer = await env_actor.get_tokenizer.call_one()

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

            # ============ Step 1: Create environments ============
            # Run games SEQUENTIALLY to avoid race conditions on shared server
            # (each thread has its own server, but games within a thread share it)

            # ============ Step 2: Rollout group (SEQUENTIALLY) ============
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

            # ============ Debug: Print first episode ============
            if episodes:
                ep = episodes[0]
                print(f"\n{'='*80}")
                print(f"[ROLLOUT {rollout_count}] Episode 0 Debug Info")
                print(f"{'='*80}")
                print(
                    f"Reward: {ep.reward}, Truncated: {ep.is_truncated}, Turns: {ep.metadata.get('num_turns', '?')}"
                )
                print(
                    f"Total tokens: {len(ep.all_token_ids)}, Trainable tokens: {ep.response_mask.sum().item()}"
                )
                print(f"\n--- Messages ---")
                for i, msg in enumerate(ep.message_log):
                    content_preview = (
                        msg["content"][:100] + "..."
                        if len(msg["content"]) > 100
                        else msg["content"]
                    )
                    print(f"  [{i}] {msg['role']:10s}: {content_preview}")
                print(f"\n--- Decoded all_token_ids ---")
                decoded_text = tokenizer.decode(ep.all_token_ids.tolist())
                print(decoded_text)

                print(f"{'='*80}\n")
                print(f"\n--- decoded_response_text ---")
                decoded_response_text = tokenizer.decode(
                    ep.all_token_ids[ep.response_mask].tolist()
                )
                print(decoded_response_text)
                print(f"{'='*80}\n")

            # ============ Step 3: Filter groups (constant rewards) ============
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

            # ============ Step 4: Compute ref_model ============
            max_len = max(len(e.all_token_ids) for e in episodes)

            # Pad input_ids and loss_masks
            padded_input_ids = []
            padded_loss_masks = []

            for i, e in enumerate(episodes):
                seq_len = len(e.all_token_ids)
                pad_len = max_len - seq_len

                # Pad tokens
                padded_tokens = F.pad(e.all_token_ids, (0, pad_len), value=pad_id)
                padded_input_ids.append(padded_tokens)

                # Pad loss_mask
                padded_mask = F.pad(e.loss_mask, (0, pad_len), value=0.0)
                padded_loss_masks.append(padded_mask)

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
                # Verify shape matches other tensors
                assert (
                    episode.ref_logprobs.shape
                    == episode.loss_mask.shape
                    == episode.all_token_ids.shape
                ), f"Shape mismatch in episode {i}"

            del ref_logprobs_padded, input_ids, loss_mask_batch

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

            # Log buffer additions
            if accepted:
                print(
                    f"[BUFFER ADD] Added {len(accepted)}/{len(episodes)} episodes with policy_v={accepted[0].policy_version}"
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
                # Log only when stuck after initial training
                if training_step > 2 and training_step % 5 == 0:
                    print(
                        f"[TRAINING] Step {training_step}: Waiting for buffer to have enough data..."
                    )
                await asyncio.sleep(1.0)
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

    num_rollout_threads = cfg.rollout_threads
    print(f"Starting GRPO with {num_rollout_threads} rollout threads")
    rollout_tasks = [
        asyncio.create_task(continuous_rollouts(thread_id=i))
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
        print(f"Stopping {len(server_processes)} OpenSpiel servers...")
        for i, server_process in enumerate(server_processes):
            server_process.terminate()
            server_process.join(timeout=2)
            if server_process.is_alive():
                print(f"⚠ Server {i} didn't stop gracefully, killing...")
                server_process.kill()
                server_process.join(timeout=1)
        print("✓ All OpenSpiel servers stopped")


if __name__ == "__main__":

    @parse
    def _main(cfg):
        asyncio.run(main(cfg))

    _main()  # @parse grabs the cfg from CLI
