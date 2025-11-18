# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from enum import Enum
from functools import lru_cache


class SanityCheckMode(Enum):
    """Sanity check modes for finalize validation."""

    STRICT = "strict"
    IGNORE_STRIPPABLE = "ignore_strippable"
    DISABLE = "disable"


@lru_cache(maxsize=1)
def get_assistant_overhead(tokenizer) -> tuple[int, list[int], list[int]]:
    """
    Get role header and footer tokens for assistant responses.

    This computes the tokens that wrap assistant content:
    - Header: <|im_start|>assistant\n
    - Footer: <|im_end|>\n

    Returns:
        (overhead_count, header_tokens, footer_tokens)
    """
    base = [
        {"role": "system", "content": ""},
    ]
    base_tokens = tokenizer.apply_chat_template(
        base, add_generation_prompt=False, tokenize=True
    )

    # Use empty content to get pure role headers/footers
    with_assistant = base + [{"role": "assistant", "content": ""}]
    full_tokens = tokenizer.apply_chat_template(
        with_assistant, add_generation_prompt=False, tokenize=True
    )

    # Extract assistant portion (all tokens after base)
    assistant_full = full_tokens[len(base_tokens) :]

    # With empty content, all tokens are header + footer
    # Typically: header = <|im_start|>assistant\n, footer = <|im_end|>\n
    # We need to split them. The footer is usually just the EOS token at the end.

    # Assume last token is EOS (footer), everything else is header
    if len(assistant_full) > 0:
        header = assistant_full[:-1]
        footer = assistant_full[-1:]
    else:
        # Edge case: no tokens (shouldn't happen)
        header = []
        footer = []

    overhead = len(header) + len(footer)
    return overhead, header, footer


class TokenAccumulator:
    """
    Accumulates tokens during multi-turn rollout.

    Key improvements over prefix matching:
    1. Uses vLLM's token_ids directly (no re-tokenization of assistant content)
    2. Pre-computed role headers avoid chat template re-application
    3. No duplicate <think> tags from Qwen's auto-wrapper behavior
    4. Drops truncated episodes (following industry best practice)

    Instead of re-tokenizing full conversation history each turn, we:
    - Use BASE anchor for user messages (O(1) tokenization)
    - Use direct tokens + static headers for assistant messages (O(0) tokenization!)
    """

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

        self.messages = messages.copy()
        self.all_tokens: list[int] = []
        self.response_mask: list[int] = []
        self.logprobs: list[float] = []

        # Pre-compute assistant role headers/footers
        overhead, self.role_header, self.role_footer = get_assistant_overhead(tokenizer)
        self.assistant_overhead = overhead

        self.is_truncated = False
        self.truncation_reason: str | None = None

        # Setup BASE anchor
        if len(messages) == 0:
            raise ValueError("Must provide at least system message")

        system_msg = (
            messages[0]
            if messages[0]["role"] == "system"
            else {"role": "system", "content": ""}
        )

        self.BASE_CHAT_HISTORY = [
            system_msg,
            {"role": "user", "content": ""},
        ]

        # Pre-compute slice positions
        self.base_tokens_wo_gen = self.tokenizer.apply_chat_template(
            self.BASE_CHAT_HISTORY,
            add_generation_prompt=False,
            tokenize=True,
        )
        self.base_len_wo_gen = len(self.base_tokens_wo_gen)

        system_tokens = self.tokenizer.apply_chat_template(
            [system_msg],
            add_generation_prompt=False,
            tokenize=True,
        )
        self.system_len = len(system_tokens)

        # Initialize with initial messages
        if len(messages) > 0:
            initial_tokens = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=False,
                tokenize=True,
            )
            self.all_tokens.extend(initial_tokens)
            self.response_mask.extend([0] * len(initial_tokens))
            self.logprobs.extend([0.0] * len(initial_tokens))

    def get_remaining_budget(self) -> int:
        current_with_overhead = len(self.all_tokens) + self.assistant_overhead
        return self.max_seq_len - current_with_overhead

    def format_prompt(self) -> str:
        """Format prompt for generation."""
        return self.tokenizer.apply_chat_template(
            self.messages,
            add_generation_prompt=True,
            tokenize=False,
        )

    def add_assistant_response(
        self,
        response_text: str,
        response_token_ids: list[int],
        response_logprobs: list[float] | None = None,
    ) -> bool:
        """
        Add assistant response using DIRECT token extraction.

        This avoids re-applying chat_template on vLLM's response, which prevents
        Qwen's auto-wrapper from adding duplicate <think></think> tags when the
        response is truncated mid-tag.

        Args:
            response_text: Response text from vLLM (for message log)
            response_token_ids: Content token IDs from vLLM (includes <think> tags)
            response_logprobs: Logprobs from vLLM (content tokens only)

        Returns:
            True if not truncated (episode can continue)
            False if truncated (episode should be discarded)
        """
        # Check if truncated - if so, REJECT entire episode
        is_truncated = (
            len(response_token_ids) > 0 and response_token_ids[-1] != self.eos_token_id
        )

        if is_truncated:
            # Mark as truncated but don't accumulate
            self.is_truncated = True
            self.truncation_reason = "generation_hit_max_tokens"
            return False

        # Only handle COMPLETE responses
        # Remove EOS from content if present (footer already has it)
        content_tokens = response_token_ids
        if content_tokens and content_tokens[-1] == self.eos_token_id:
            content_tokens = content_tokens[:-1]

        # Combine: header + content (from vLLM) + footer
        assistant_tokens = self.role_header + content_tokens + self.role_footer

        # Create logprobs: zeros for headers/footers, actual for content
        assistant_logprobs = [0.0] * len(self.role_header)
        if response_logprobs is not None:
            assistant_logprobs.extend(response_logprobs[: len(content_tokens)])
        else:
            assistant_logprobs.extend([0.0] * len(content_tokens))
        assistant_logprobs.extend([0.0] * len(self.role_footer))

        # Accumulate (all complete responses are trainable, mask=1)
        self.all_tokens.extend(assistant_tokens)
        self.response_mask.extend([1] * len(assistant_tokens))
        self.logprobs.extend(assistant_logprobs)

        # Add to messages for next prompt
        self.messages.append({"role": "assistant", "content": response_text})

        return True

    def add_user_message(self, content: str) -> bool:
        """
        Add user message using BASE anchor.

        Args:
            content: User message content

        Returns:
            True if successful, False if would exceed budget
        """
        self.messages.append({"role": "user", "content": content})

        # Tokenize system + user to get delta
        temp_messages = [
            self.BASE_CHAT_HISTORY[0],
            {"role": "user", "content": content},
        ]
        full_with_user = self.tokenizer.apply_chat_template(
            temp_messages,
            add_generation_prompt=False,
            tokenize=True,
        )
        user_message_tokens = full_with_user[self.system_len :]

        # Check budget
        success = True
        new_amount_to_add = len(user_message_tokens) + self.assistant_overhead
        budget = self.max_seq_len - len(self.all_tokens)
        if new_amount_to_add > budget:
            self.is_truncated = True
            self.truncation_reason = "user_message_length"
            success = False

        # Accumulate
        maybe_truncated_tokens = user_message_tokens[:budget]
        self.all_tokens.extend(maybe_truncated_tokens)
        self.response_mask.extend([0] * len(maybe_truncated_tokens))
        self.logprobs.extend([0.0] * len(maybe_truncated_tokens))

        return success

    def finalize(self, strict: bool = None) -> bool:
        """
        Validate token accumulation against ground truth.

        With the v9 fix (direct token extraction), this should ALWAYS match
        for complete responses. Any mismatch indicates a bug.

        Args:
            strict: Override sanity_check_mode if provided

        Returns:
            True if validation passed or skipped, False if mismatch detected

        Raises:
            ValueError: If mismatch detected and mode is STRICT
        """
        assert len(self.logprobs) == len(self.all_tokens)
        assert len(self.logprobs) == len(self.response_mask)

        mode = self.sanity_check_mode
        if strict is not None:
            mode = SanityCheckMode.STRICT if strict else SanityCheckMode.DISABLE

        if mode == SanityCheckMode.DISABLE:
            return True

        ground_truth = self.tokenizer.apply_chat_template(
            self.messages,
            add_generation_prompt=False,
            tokenize=True,
        )

        if len(self.all_tokens) != len(ground_truth):
            diff = len(ground_truth) - len(self.all_tokens)

            # Check if only whitespace differs
            if mode == SanityCheckMode.IGNORE_STRIPPABLE:
                accumulated_text = self.tokenizer.decode(self.all_tokens)
                ground_truth_text = self.tokenizer.decode(ground_truth)
                if accumulated_text.strip() == ground_truth_text.strip():
                    return True

            error_msg = (
                f"Token accumulation mismatch!\n"
                f"  Accumulated: {len(self.all_tokens)} tokens\n"
                f"  Ground truth: {len(ground_truth)} tokens\n"
                f"  Difference: {diff}\n"
                f"  Last 20 accumulated: {self.all_tokens[-20:]}\n"
                f"  Last 20 ground truth: {ground_truth[-20:]}\n"
                f"  Sanity check mode: {mode.value}"
            )

            if mode == SanityCheckMode.STRICT:
                raise ValueError(error_msg)
            else:
                print(f"⚠️  {error_msg}")
                return False

        return True
