# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from enum import Enum


class SanityCheckMode(Enum):
    """Sanity check modes for finalize validation."""

    STRICT = "strict"
    IGNORE_STRIPPABLE = "ignore_strippable"
    DISABLE = "disable"


class TokenAccumulator:
    """
    Accumulates tokens during multi-turn rollout.

    Simplified V2 approach:
    - Use full re-tokenization with prefix matching (always correct)
    - Use vLLM's token_ids to find content location
    - Map logprobs to matching positions (1:1 with vLLM's token_ids)
    - Use 0.0 for role markers/headers/footers
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

        self.is_truncated = False
        self.truncation_reason: str | None = None

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
        """
        Get remaining token budget.

        Use conservative estimate: reserve ~10 tokens for assistant overhead.
        """
        estimated_overhead = 10
        return max(0, self.max_seq_len - len(self.all_tokens) - estimated_overhead)

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
        Add assistant response using prefix matching.

        Simple approach:
        1. Check truncation using vLLM's token_ids
        2. Use prefix matching to get new tokens (always correct)
        3. Find where vLLM's tokens appear in new tokens
        4. Map logprobs: vLLM's logprobs at matching positions, 0.0 elsewhere

        Args:
            response_text: Response text from vLLM
            response_token_ids: Token IDs from vLLM (includes EOS if complete)
            response_logprobs: Logprobs from vLLM (1:1 with token_ids)

        Returns:
            True if not truncated, False if truncated
        """
        # Check truncation
        is_truncated = (
            len(response_token_ids) > 0 and response_token_ids[-1] != self.eos_token_id
        )

        if is_truncated:
            self.is_truncated = True
            self.truncation_reason = "generation_hit_max_tokens"
            return False

        # Add message
        self.messages.append({"role": "assistant", "content": response_text})

        # Get ground truth tokens via prefix matching
        full_tokens = self.tokenizer.apply_chat_template(
            self.messages,
            add_generation_prompt=False,
            tokenize=True,
        )
        new_tokens = full_tokens[len(self.all_tokens) :]

        # Accumulate tokens
        self.all_tokens.extend(new_tokens)
        self.response_mask.extend([1] * len(new_tokens))

        # For logprobs: find where vLLM's tokens are in new_tokens
        content_start = None
        if response_logprobs is not None and len(response_logprobs) == len(
            response_token_ids
        ):
            # Search for vLLM's tokens as a substring
            for i in range(len(new_tokens) - len(response_token_ids) + 1):
                if new_tokens[i : i + len(response_token_ids)] == response_token_ids:
                    content_start = i
                    break

        # Build logprobs array
        if content_start is not None:
            # Found them! Map logprobs correctly
            logprobs = (
                [0.0] * content_start  # Role markers before
                + response_logprobs  # Actual logprobs from vLLM
                + [0.0]
                * (len(new_tokens) - content_start - len(response_token_ids))  # After
            )
        else:
            # Fallback: all zeros
            logprobs = [0.0] * len(new_tokens)

        self.logprobs.extend(logprobs)

        return True

    def add_user_message(self, content: str, check_budget: bool = True) -> bool:
        """
        Add user message using prefix matching.

        Args:
            content: User message content
            check_budget: Whether to check budget and truncate if necessary

        Returns:
            True if successful, False if truncated
        """
        # Add message
        self.messages.append({"role": "user", "content": content})

        # Re-tokenize full conversation
        full_tokens = self.tokenizer.apply_chat_template(
            self.messages,
            add_generation_prompt=False,
            tokenize=True,
        )

        # Extract new tokens
        new_tokens = full_tokens[len(self.all_tokens) :]

        # Check budget
        success = True
        if check_budget:
            estimated_assistant_overhead = 10
            budget = self.max_seq_len - len(self.all_tokens)

            if len(new_tokens) + estimated_assistant_overhead > budget:
                self.is_truncated = True
                self.truncation_reason = "user_message_length"
                success = False
                # Truncate tokens to fit
                new_tokens = new_tokens[: max(0, budget - estimated_assistant_overhead)]

        # Accumulate
        self.all_tokens.extend(new_tokens)
        self.response_mask.extend([0] * len(new_tokens))
        self.logprobs.extend([0.0] * len(new_tokens))

        return success

    def finalize(self, strict: bool = None) -> bool:
        """
        Validate token accumulation against ground truth.

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
