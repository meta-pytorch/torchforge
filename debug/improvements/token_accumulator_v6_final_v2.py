# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Token accumulation for multi-turn RL episodes using vLLM tokens directly.

See TokenAccumulator class for details.
"""

import threading
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import torch


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
        Show conversation with trainability highlighted.

        Uses colored text runs for readability (similar to tinker-cookbook's format_colorized).
        Groups consecutive tokens with same trainability and decodes together for proper
        multi-byte character handling.

        Args:
            max_chars: Maximum characters to show per message (default: 5000)
        """
        print("=" * 80)
        print(f"TokenAccumulator: {len(self._tokens)}/{self.max_len} tokens")
        print("=" * 80)

        if not self.messages:
            print("(no messages)")
            print("=" * 80)
            return

        # Show each message with trainability info
        current_idx = 0
        for msg_num, msg in enumerate(self.messages):
            role = msg["role"]
            content = msg["content"]

            # Find tokens for this message
            msg_end = None
            for end_idx, end_role, _ in self._message_ends:
                if end_idx > current_idx:
                    if role in end_role or end_role == "assistant_suffix":
                        msg_end = end_idx
                        break

            if msg_end is None:
                msg_end = len(self._tokens)

            # Count trainable tokens
            trainable_count = sum(self._mask[current_idx:msg_end])
            total_count = msg_end - current_idx

            # Visual indicator
            if trainable_count == total_count:
                indicator = "✓ TRAINABLE"
                color = "\033[92m"  # Green
            elif trainable_count > 0:
                indicator = f"⚠ PARTIAL ({trainable_count}/{total_count})"
                color = "\033[93m"  # Yellow
            else:
                indicator = "· not trainable"
                color = "\033[90m"  # Gray

            # Header
            print(
                f"\n{color}[{msg_num}] {role:10s} [{current_idx:4d}:{msg_end:4d}] {indicator}\033[0m"
            )

            # Content with optional truncation
            if len(content) > max_chars:
                preview = (
                    content[:max_chars]
                    + f"\n... ({len(content) - max_chars} more chars)"
                )
            else:
                preview = content

            print(f"    {preview}")

            # Show colorized tokens for this message
            self._show_colorized_tokens(current_idx, msg_end)

            current_idx = msg_end

        # Summary
        print(f"\n{'='*80}")
        trainable_total = sum(self._mask)
        pct = 100 * trainable_total / len(self._tokens) if self._tokens else 0
        print(
            f"Total: {trainable_total}/{len(self._tokens)} trainable tokens ({pct:.1f}%)"
        )
        print("=" * 80)

    def _show_colorized_tokens(self, start_idx: int, end_idx: int) -> None:
        """
        Show colorized token-level view for a message range.

        Groups consecutive tokens with same trainability into "runs" and decodes
        them together. This handles multi-byte characters correctly.
        """
        if start_idx >= end_idx:
            return

        chunks = []
        current_ids = []
        current_trainable = None

        def flush_run():
            if not current_ids:
                return
            # Decode entire run at once
            with self._lock:
                decoded = self.tokenizer.decode(current_ids)
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
        for i in range(start_idx, end_idx):
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
            print("    Tokens: " + " ".join(chunks))

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
        with self._lock:
            full_tokens = self.tokenizer.apply_chat_template(
                self.messages,
                add_generation_prompt=False,
                tokenize=True,
                enable_thinking=self.thinking,
            )

        # Account for suffix: accumulated = full + suffix_insertions
        num_assistant_msgs = sum(
            1 for msg in self.messages if msg["role"] == "assistant"
        )
        expected_suffix_tokens = num_assistant_msgs * len(self.suffix)

        accumulated_len = len(token_ids)
        expected_len = len(full_tokens) + expected_suffix_tokens

        if accumulated_len != expected_len:
            msg = (
                f"Prefix consistency failed: "
                f"accumulated={accumulated_len} tokens, "
                f"expected={expected_len} (full={len(full_tokens)} + suffix={expected_suffix_tokens})"
            )
            if self.validation == ValidationMode.STRICT:
                raise AssertionError(msg)
            print(f"WARNING: {msg}")
