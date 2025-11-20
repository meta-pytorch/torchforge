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
    AGENT_TOO_LONG = "agent_too_long"
    USER_TOO_LONG = "user_too_long"
    TOOL_TOO_LONG = "tool_too_long"


class TokenAccumulator:
    """
    Accumulates tokens during multi-turn RL rollouts using vLLM tokens directly (VERL approach).

    Key design:
    - Uses generation tokens from vLLM WITHOUT re-tokenizing (avoids chat template suffix bugs)
    - Generation prompt (<|im_start|>assistant\n) computed from anchor, added separately
    - Prefix has response_mask=False, vLLM content has response_mask=True

    Usage:
        acc = TokenAccumulator(tokenizer, messages=[...], max_seq_len=2048, eos_token_id=...)
        acc.add_user_message("Hello")
        prompt = acc.format_prompt()
        response = model.generate(prompt, max_tokens=acc.get_remaining_budget())
        acc.add_assistant_response(response.token_ids, response.logprobs)

        return Episode(
            token_ids=acc.accumulated_tokens,
            response_mask=acc.response_mask,
            ...)
    """

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

        self.messages = []
        self.accumulated_tokens = []
        self.response_mask = []
        self.logprobs = []
        self.is_truncated = False
        self.truncation_reason = None

        self._setup_anchor(messages)
        self._initialize_messages(messages)

    def add_user_message(self, content: str) -> bool:
        """Add user message, truncating to fit budget if necessary. Returns False if truncated."""

        message = {"role": "user", "content": content}

        with self._tokenizer_lock:
            # Tokenize [system, user] to get delta tokens
            full = self.tokenizer.apply_chat_template(
                [self.anchor[0], message],
                add_generation_prompt=False,
                tokenize=True,
                enable_thinking=self.enable_thinking,
            )

        # Extract only user tokens (remove system prefix)
        user_tokens = full[self.system_len :]

        # truncate
        budget = self.get_remaining_budget()
        original_len = len(user_tokens)
        user_tokens = self._truncate_to_fit(
            user_tokens, budget, TruncationReason.USER_TOO_LONG
        )

        if user_tokens:
            self.messages.append(message)
            self._accumulate(user_tokens, mask=[False] * len(user_tokens))

        # False if truncated
        return len(user_tokens) == original_len

    def add_assistant_response(
        self,
        response_text: str,
        response_token_ids: list[int],
        response_logprobs: list[float] | None = None,
    ) -> bool:
        """
        Add assistant response using vLLM tokens directly.
        Returns False if truncated (no EOS or budget exceeded).
        """
        # Check for truncation
        if not response_token_ids or response_token_ids[-1] != self.eos_token_id:
            return self._mark_truncated(TruncationReason.AGENT_TOO_LONG)

        # Check budget: generation_prompt + vLLM tokens
        total_len = self.generation_prompt_len + len(response_token_ids)
        if total_len > self.get_remaining_budget():
            return self._mark_truncated(TruncationReason.AGENT_TOO_LONG)

        # Decode for message log
        self.messages.append({"role": "assistant", "content": response_text})

        # Add generation prompt (not trainable)
        self._accumulate(
            self.generation_prompt_tokens,
            mask=[False] * len(self.generation_prompt_tokens),
            logprobs=[0.0] * len(self.generation_prompt_tokens),
        )

        # Add vLLM tokens (trainable)
        if response_logprobs and len(response_logprobs) == len(response_token_ids):
            logprobs = response_logprobs
        else:
            logprobs = [0.0] * len(response_token_ids)

        self._accumulate(
            response_token_ids, mask=[True] * len(response_token_ids), logprobs=logprobs
        )

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
        """Get remaining tokens. It also reserves space for generation prompt,
        e.g. "<|im_start|>assistant\n" """
        used = len(self.accumulated_tokens) + self.generation_prompt_len
        return max(0, self.max_seq_len - used)

    def finalize(self) -> bool:
        """Validate episode. Returns True if valid."""
        self._check_structure()
        # if self.sanity_check_mode != SanityCheckMode.DISABLE:
        #     self._check_eos_alignment()
        return True

    def _setup_anchor(self, messages: list[dict]):
        """
        Setup anchor conversation for delta tokenization.

        Delta tokenization: Instead of re-tokenizing the full conversation after each message,
        we tokenize only the new message against a fixed anchor ([system, empty_user]).

        Computes:
        - generation_prompt_tokens: tokens for "<|im_start|>assistant\n" (added separately from vLLM tokens)
        - generation_prompt_len: length of generation prompt (for budget calculation)
        - system_len: tokens in [system] alone (for user message delta slicing)
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

        # Compute generation prompt tokens from anchor
        anchor_without = self.tokenizer.apply_chat_template(
            self.anchor,
            add_generation_prompt=False,
            tokenize=True,
            enable_thinking=self.enable_thinking,
        )
        anchor_with = self.tokenizer.apply_chat_template(
            self.anchor,
            add_generation_prompt=True,
            tokenize=True,
            enable_thinking=self.enable_thinking,
        )

        # e.g., "<|im_start|>assistant\n"
        self.generation_prompt_tokens = anchor_with[len(anchor_without) :]
        self.generation_prompt_len = len(self.generation_prompt_tokens)

        # System message length alone (for user message delta slicing)
        self.system_len = len(
            self.tokenizer.apply_chat_template(
                [system_msg],
                add_generation_prompt=False,
                tokenize=True,
                enable_thinking=self.enable_thinking,
            )
        )

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
        self._accumulate(initial_tokens, mask=[False] * len(initial_tokens))

    def _truncate_to_fit(
        self, tokens: list[int], available: int, reason: TruncationReason
    ) -> list[int]:
        """Truncate tokens to fit available space."""
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

    # def _check_eos_alignment(self):
    #     """
    #     Verify no tokens after EOS have response_mask=True (the bug we fixed).

    #     For each assistant response, the last response_mask=True token must be EOS.
    #     This ensures we're not training on chat template suffix tokens (like \n after EOS).
    #     """
    #     in_response = False
    #     last_response_idx = -1

    #     for i, (token, is_response) in enumerate(
    #         zip(self.accumulated_tokens, self.response_mask)
    #     ):
    #         if is_response and not in_response:
    #             in_response = True
    #         elif is_response:
    #             last_response_idx = i
    #         elif not is_response and in_response:
    #             # End of response - check last token was EOS
    #             if (
    #                 last_response_idx >= 0
    #                 and self.accumulated_tokens[last_response_idx] != self.eos_token_id
    #             ):
    #                 raise ValueError(
    #                     f"Response ended at position {last_response_idx} with token "
    #                     f"{self.accumulated_tokens[last_response_idx]}, expected EOS {self.eos_token_id}"
    #                 )
    #             in_response = False
    #             last_response_idx = -1

    #     # Check final response if episode ends mid-response
    #     if in_response and last_response_idx >= 0:
    #         if self.accumulated_tokens[last_response_idx] != self.eos_token_id:
    #             raise ValueError(
    #                 f"Final response ended at position {last_response_idx} with token "
    #                 f"{self.accumulated_tokens[last_response_idx]}, expected EOS {self.eos_token_id}"
    #             )
