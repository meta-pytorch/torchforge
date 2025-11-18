from enum import Enum


class SanityCheckMode(Enum):
    """Sanity check modes for finalize validation."""

    STRICT = "strict"
    IGNORE_STRIPPABLE = "ignore_strippable"
    DISABLE = "disable"


class TruncationReason(Enum):
    """Reason for episode truncation."""

    max_num_turns = "max_num_turns"
    agent_max_length = "agent_max_length"  # Agent generation hit max_tokens (no EOS)
    tool_max_length = "tool_max_length"  # Tool response too long
    user_max_length = "user_max_length"  # User message too long


class TokenAccumulator:
    """
    Accumulates tokens during multi-turn rollout using BASE anchor pattern.

    Key insight: Qwen's chat template removes <think> tags from previous assistant
    messages when adding new messages. This breaks prefix matching.

    Solution: Never re-tokenize the full conversation. Instead:
    1. Use a fixed BASE conversation [system, empty_user] as anchor
    2. Tokenize only deltas (one new message at a time)
    3. Slice from pre-computed offsets to extract just the new tokens

    This approach:
    - Works with Qwen's thinking tag removal
    - Minimizes tokenization calls (1 per message instead of full conversation)
    - Provides accurate budget tracking

    Truncation behavior (CRITICAL):
        ⚠️ ASSISTANT TRUNCATION → EPISODE DROPPED
           If vLLM truncates assistant response (no EOS token), the entire
           episode is rejected. add_assistant_response() returns False and
           nothing is accumulated.

        ✓ USER TRUNCATION → EPISODE CONTINUES WITH TRUNCATION FLAG
           If user message would exceed budget, it's truncated to fit.
           add_user_message() returns False, sets is_truncated=True, but
           the truncated message is accumulated and episode can continue.

    Example - Multi-turn with budget constraints:
        ```python
        # Initialize with tight budget
        messages = [{"role": "system", "content": "You are helpful."}]
        acc = TokenAccumulator(
            tokenizer=tokenizer,
            messages=messages,
            max_seq_len=100,  # Tight budget
            eos_token_id=128001,
        )
        # State: all_tokens=[...], len=25 (system prompt)

        # Turn 1: User asks, assistant responds
        acc.add_user_message("Say hi")
        # State: all_tokens=[..., user_tokens], len=35
        # Remaining budget: 100 - 35 - 6 (overhead) = 59 tokens

        response = llm.generate(
            acc.format_prompt(),
            max_tokens=acc.get_remaining_budget()  # max_tokens=59
        )
        # response.text = "hi"
        # response.token_ids = [6151, 128001]  # "hi" + EOS

        success = acc.add_assistant_response("hi", response.token_ids)
        # success=True (has EOS token, complete response)
        # State: all_tokens=[..., user, assistant], len=45
        # is_truncated=False

        # Turn 2: Try to add very long user message
        long_msg = "Please explain quantum mechanics in detail..." * 100
        success = acc.add_user_message(long_msg)
        # User message is 200 tokens, but only 100-45-6=49 tokens available
        # Message is TRUNCATED to fit
        # success=False (truncated)
        # State: all_tokens=[..., truncated_user_msg], len=94
        # is_truncated=True, truncation_reason=TruncationReason.user_max_length
        # ⚠️ Episode is marked truncated but tokens are valid

        # Episode outcome:
        # - all_tokens.shape = (94,)
        # - response_mask.shape = (94,)  # 1s for assistant tokens, 0s elsewhere
        # - logprobs.shape = (94,)
        # - is_truncated = True
        # - Should be DROPPED in training (truncated episodes are invalid)
        ```

    Quick reference for 4 test scenarios:
        1. Complete single turn: success=True, is_truncated=False → ✓ Train
        2. Assistant truncated: success=False → ✗ Drop entire episode
        3. Complete multi-turn: all success=True → ✓ Train
        4. User truncated: success=False, is_truncated=True → ✗ Drop
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
        self.truncation_reason: TruncationReason | None = None

        # Setup BASE anchor for delta tokenization
        if len(messages) == 0:
            raise ValueError("Must provide at least system message")

        system_msg = (
            messages[0]
            if messages[0]["role"] == "system"
            else {"role": "system", "content": ""}
        )

        # BASE: [system, empty_user] - never changes, so consistent tokenization
        self.BASE_CHAT_HISTORY = [
            system_msg,
            {"role": "user", "content": ""},
        ]

        # Pre-compute base lengths for slicing
        base_wo_gen = tokenizer.apply_chat_template(
            self.BASE_CHAT_HISTORY,
            add_generation_prompt=False,
            tokenize=True,
        )
        self.base_wo_gen_len = len(base_wo_gen)

        base_with_gen = tokenizer.apply_chat_template(
            self.BASE_CHAT_HISTORY,
            add_generation_prompt=True,
            tokenize=True,
        )
        self.base_with_gen_len = len(base_with_gen)

        # System message length for user message slicing
        system_tokens = tokenizer.apply_chat_template(
            [system_msg],
            add_generation_prompt=False,
            tokenize=True,
        )
        self.system_len = len(system_tokens)

        # Assistant overhead = generation prompt tokens
        self.assistant_overhead = self.base_with_gen_len - self.base_wo_gen_len

        # Initialize with initial messages
        if len(messages) > 0:
            initial_tokens = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=False,
                tokenize=True,
            )

            # Check if initial messages exceed budget
            if len(initial_tokens) > max_seq_len:
                self.is_truncated = True
                self.truncation_reason = TruncationReason.user_max_length
                # Truncate to fit
                initial_tokens = initial_tokens[:max_seq_len]

            self.all_tokens.extend(initial_tokens)
            self.response_mask.extend([0] * len(initial_tokens))
            self.logprobs.extend([0.0] * len(initial_tokens))

    def get_remaining_budget(self) -> int:
        """Get remaining token budget accounting for assistant overhead."""
        current_with_overhead = len(self.all_tokens) + self.assistant_overhead
        return max(0, self.max_seq_len - current_with_overhead)

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
        Add assistant response using BASE anchor delta tokenization.

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
            self.truncation_reason = TruncationReason.agent_max_length
            return False

        # Add message
        self.messages.append({"role": "assistant", "content": response_text})

        # Delta tokenization: [system, empty_user, assistant_new]
        temp_messages = [
            self.BASE_CHAT_HISTORY[0],  # System
            {"role": "user", "content": ""},  # Empty user from base
            {"role": "assistant", "content": response_text},
        ]
        full_with_assistant = self.tokenizer.apply_chat_template(
            temp_messages,
            add_generation_prompt=False,
            tokenize=True,
        )

        # Extract only assistant tokens (everything after base)
        assistant_tokens = full_with_assistant[self.base_wo_gen_len :]

        # Check budget before accumulating
        available_space = self.max_seq_len - len(self.all_tokens)
        if len(assistant_tokens) > available_space:
            # Budget overflow - this shouldn't happen if caller used get_remaining_budget()
            # but we need to handle it gracefully
            self.is_truncated = True
            self.truncation_reason = TruncationReason.agent_max_length
            # Remove the message we just added
            self.messages.pop()
            return False

        # Accumulate tokens
        self.all_tokens.extend(assistant_tokens)
        self.response_mask.extend([1] * len(assistant_tokens))

        # Map logprobs: find where vLLM's tokens appear in assistant_tokens
        content_start = None
        if response_logprobs is not None and len(response_logprobs) == len(
            response_token_ids
        ):
            # Search for vLLM's token_ids as substring
            for i in range(len(assistant_tokens) - len(response_token_ids) + 1):
                if (
                    assistant_tokens[i : i + len(response_token_ids)]
                    == response_token_ids
                ):
                    content_start = i
                    break

        # Build logprobs array
        if content_start is not None:
            # Found exact match - map logprobs correctly
            logprobs = (
                [0.0] * content_start  # Role markers before
                + response_logprobs  # Actual logprobs from vLLM
                + [0.0]
                * (len(assistant_tokens) - content_start - len(response_token_ids))
            )
        else:
            # Fallback: all zeros (shouldn't happen with correct implementation)
            logprobs = [0.0] * len(assistant_tokens)

        self.logprobs.extend(logprobs)

        return True

    def add_user_message(self, content: str, check_budget: bool = True) -> bool:
        """
        Add user message using BASE anchor delta tokenization.

        Args:
            content: User message content
            check_budget: Whether to check budget and truncate if necessary

        Returns:
            True if successful, False if truncated
        """
        # Add message
        self.messages.append({"role": "user", "content": content})

        # Delta tokenization: [system, user_new]
        temp_messages = [
            self.BASE_CHAT_HISTORY[0],  # System
            {"role": "user", "content": content},
        ]
        full_with_user = self.tokenizer.apply_chat_template(
            temp_messages,
            add_generation_prompt=False,
            tokenize=True,
        )

        # Extract only user message tokens (everything after system)
        user_message_tokens = full_with_user[self.system_len :]

        # Check budget
        success = True
        if check_budget:
            new_amount = len(user_message_tokens) + self.assistant_overhead
            budget = self.max_seq_len - len(self.all_tokens)

            if new_amount > budget:
                self.is_truncated = True
                self.truncation_reason = TruncationReason.user_max_length
                success = False
                # Truncate to fit (if budget allows any tokens)
                available = max(0, budget - self.assistant_overhead)
                user_message_tokens = user_message_tokens[:available]

        # Accumulate (only if there are tokens to add)
        if len(user_message_tokens) > 0:
            self.all_tokens.extend(user_message_tokens)
            self.response_mask.extend([0] * len(user_message_tokens))
            self.logprobs.extend([0.0] * len(user_message_tokens))

        return success

    def finalize(self, strict: bool = None) -> bool:
        """
        Validate token accumulation.

        Note: With Qwen, ground truth comparison will fail because Qwen removes
        <think> tags from previous assistant messages. Our accumulated tokens
        are correct (they match what was actually generated). We validate
        structure instead of exact token match.

        Args:
            strict: Override sanity_check_mode if provided

        Returns:
            True if validation passed

        Raises:
            ValueError: If critical issues detected
        """
        # Always check basic structure
        assert len(self.all_tokens) == len(self.response_mask)
        assert len(self.all_tokens) == len(self.logprobs)

        # Check we didn't exceed budget
        if len(self.all_tokens) > self.max_seq_len:
            raise ValueError(
                f"Token accumulation exceeded max_seq_len! "
                f"{len(self.all_tokens)} > {self.max_seq_len}"
            )

        mode = self.sanity_check_mode
        if strict is not None:
            mode = SanityCheckMode.STRICT if strict else SanityCheckMode.DISABLE

        if mode == SanityCheckMode.DISABLE:
            return True

        # Try ground truth comparison (will fail with Qwen multi-turn)
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

            # Log warning about mismatch
            warning_msg = (
                f"Token accumulation mismatch detected:\n"
                f"  Accumulated: {len(self.all_tokens)} tokens\n"
                f"  Ground truth: {len(ground_truth)} tokens\n"
                f"  Difference: {diff}\n"
                f"  Note: This can happen when the chat template modifies previous messages\n"
                f"        (e.g., Qwen strips <think> tags). Accumulated tokens are correct\n"
                f"        (they match what was actually generated)."
            )

            if mode == SanityCheckMode.STRICT:
                raise ValueError(warning_msg)
            else:
                # Just warn and continue (like VERL does)
                print(f"⚠️  {warning_msg}")
                return True  # Still pass validation

        return True
