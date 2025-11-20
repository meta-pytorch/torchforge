# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import threading
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
        enable_thinking: bool = True,
        sanity_check_mode: SanityCheckMode = SanityCheckMode.STRICT,
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.eos_token_id = eos_token_id
        self.enable_thinking = enable_thinking
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
            mask = [False] * len(user_tokens)
            self._accumulate(user_tokens, mask=mask)

        return len(user_tokens) == original_len

    def add_assistant_response(
        self,
        response_text: str,
        response_token_ids: list[int],
        response_logprobs: list[float] | None = None,
    ) -> bool:
        """
        Add assistant response. Returns False if response was truncated (no EOS).
        Episode should be dropped if this returns False.
        """
        # Check for truncation (missing EOS)
        if response_token_ids and response_token_ids[-1] != self.eos_token_id:
            return self._mark_truncated(TruncationReason.AGENT_TOO_LONG)

        message = {"role": "assistant", "content": response_text}
        assistant_tokens = self._tokenize_delta(message, "assistant")

        # Check budget - reject if would exceed max_seq_len
        if len(assistant_tokens) > self.get_remaining_budget():
            return self._mark_truncated(TruncationReason.AGENT_TOO_LONG)
        else:
            self.messages.append({"role": "assistant", "content": response_text})

        # Use pre-calculated generation_prompt_len for prefix
        # assistant_tokens includes prefix + content, so we mask prefix as False
        prefix_len = self.generation_prompt_len
        mask = [False] * prefix_len + [True] * (len(assistant_tokens) - prefix_len)

        # Map logprobs: vLLM returns content tokens only, pad at start for prefix
        if (
            response_logprobs
            and len(response_logprobs) <= len(assistant_tokens) - prefix_len
        ):
            logprobs = [0.0] * prefix_len + response_logprobs
            # Pad any remaining tokens after vLLM tokens (e.g., trailing newline)
            remaining = len(assistant_tokens) - prefix_len - len(response_logprobs)
            if remaining > 0:
                logprobs.extend([0.0] * remaining)
        else:
            logprobs = None

        self._accumulate(assistant_tokens, mask=mask, logprobs=logprobs)
        return True

    def format_prompt(self) -> str:
        """Format current conversation for generation."""
        with self._tokenizer_lock:
            return self.tokenizer.apply_chat_template(
                self.messages,
                add_generation_prompt=True,
                tokenize=False,
                enable_thinking=self.enable_thinking,
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
            self.anchor,
            add_generation_prompt=False,
            tokenize=True,
            enable_thinking=self.enable_thinking,
        )
        self.anchor_len = len(anchor_tokens)

        # Length of anchor WITH generation prompt (VERL approach)
        anchor_with_gen = self.tokenizer.apply_chat_template(
            self.anchor,
            add_generation_prompt=True,
            tokenize=True,
            enable_thinking=self.enable_thinking,
        )
        self.anchor_with_gen_len = len(anchor_with_gen)
        self.generation_prompt_len = self.anchor_with_gen_len - self.anchor_len

        # System message length alone (for user message delta slicing), e.g. full[self.system_len:]
        system_tokens = self.tokenizer.apply_chat_template(
            [system_msg],
            add_generation_prompt=False,
            tokenize=True,
            enable_thinking=self.enable_thinking,
        )
        self.system_len = len(system_tokens)

    def _initialize_messages(self, messages: list[dict]):
        """Initialize conversation with provided messages."""
        if not messages:
            return

        initial_tokens = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=False,
            tokenize=True,
            enable_thinking=self.enable_thinking,
        )

        if len(initial_tokens) > self.max_seq_len:
            self._mark_truncated(TruncationReason.USER_TOO_LONG)
            initial_tokens = initial_tokens[: self.max_seq_len]

        self.messages = messages.copy()
        mask = [False] * len(initial_tokens)
        self._accumulate(initial_tokens, mask=mask)

    def _tokenize_delta(self, message: dict, role: str) -> list[int]:
        """Tokenize single message using anchor conversation."""
        if role == "assistant":
            temp = [self.anchor[0], {"role": "user", "content": ""}, message]
            # Slice from anchor_len to include prefix tokens in accumulated_tokens
            offset = self.anchor_len
        else:  # user
            temp = [self.anchor[0], message]
            offset = self.system_len

        with self._tokenizer_lock:
            full = self.tokenizer.apply_chat_template(
                temp,
                add_generation_prompt=False,
                tokenize=True,
                enable_thinking=self.enable_thinking,
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
        self, tokens: list[int], mask: list[bool], logprobs: list[float] | None = None
    ):
        """Add tokens to accumulator."""
        self.accumulated_tokens.extend(tokens)
        self.response_mask.extend(mask)
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
            self.messages,
            add_generation_prompt=False,
            tokenize=True,
            enable_thinking=self.enable_thinking,
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
