# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Token Accumulator V3 Tests

Tests for TokenAccumulator v6 (migrated from v2 tests for v5).
All tests run with both thinking=True and thinking=False.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from debug.token_accumulator_fn_v6 import (
    EpisodeData,
    TokenAccumulator,
    TruncationReason,
    ValidationMode,
)
from vllm.transformers_utils.tokenizer import get_tokenizer


# ============================================================================
# Utilities
# ============================================================================

MODEL_NAME = "Qwen/Qwen3-1.7B"


def assert_no_training_after_eos(tokens, response_mask, eos_token_id):
    """
    Verify no tokens after EOS are trainable (the bug we fixed).

    For each EOS token, check that the NEXT position does not have response_mask=True.
    This prevents training on chat template suffix tokens like '\n' after EOS.
    """
    if len(tokens) == 0:
        return

    # Create mask of positions that come AFTER an EOS token
    eos_mask = [t == eos_token_id for t in tokens]

    # Shift right: position i is True if position i-1 was EOS
    shifted_mask = [False] + eos_mask[
        :-1
    ]  # Prepend False since position 0 has no "before"

    for i, (after_eos, is_trainable) in enumerate(zip(shifted_mask, response_mask)):
        if after_eos and is_trainable:
            raise AssertionError(
                f"❌ BUG: Token at position {i} is trainable but comes after EOS!\n"
                f"   Token ID: {tokens[i]}\n"
                f"   response_mask: {is_trainable}\n"
                f"   Previous token (EOS): {tokens[i-1]}"
            )


def create_accumulator(max_len=2048, thinking=True, system_content="You are helpful."):
    """Factory for creating test accumulators."""
    tokenizer = get_tokenizer(MODEL_NAME)
    return TokenAccumulator(
        tokenizer=tokenizer,
        messages=[{"role": "system", "content": system_content}],
        max_len=max_len,
        eos_id=tokenizer.eos_token_id,
        thinking=thinking,
    )


def mock_vllm_response(tokenizer, text, include_eos=True):
    """
    Simulate vLLM generation (tokens without re-tokenizing with chat template).
    This is what vLLM returns: raw content tokens + EOS.
    """
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if include_eos:
        tokens.append(tokenizer.eos_token_id)
    return tokens


# ============================================================================
# Test Cases
# ============================================================================


@pytest.mark.parametrize("thinking", [True, False])
class TestBasicFunctionality:
    """Core functionality tests."""

    def test_single_turn_complete(self, thinking):
        """Test: system -> user -> assistant (complete with EOS)."""
        acc = create_accumulator(thinking=thinking)
        tokenizer = acc.tokenizer

        # User message
        success = acc.add_user("Say hi")
        assert success

        # Generate assistant response
        response_tokens = mock_vllm_response(tokenizer, "Hello!", include_eos=True)
        success = acc.add_assistant("Hello!", response_tokens)

        assert success, "Should accept complete response"
        assert not acc.truncated

        episode = acc.get_data()
        assert isinstance(episode, EpisodeData)
        assert_no_training_after_eos(
            episode.token_ids.tolist(),
            episode.response_mask.tolist(),
            tokenizer.eos_token_id,
        )

    def test_truncated_response_no_eos(self, thinking):
        """Test: Response without EOS is rejected."""
        acc = create_accumulator(thinking=thinking)
        tokenizer = acc.tokenizer

        acc.add_user("Say hi")
        response_tokens = mock_vllm_response(tokenizer, "Hello!", include_eos=False)
        success = acc.add_assistant("Hello!", response_tokens)

        assert not success, "Should reject response without EOS"
        assert acc.truncated
        assert acc.truncation_reason == TruncationReason.ASSISTANT_TOO_LONG

    def test_multi_turn(self, thinking):
        """Test: system -> user -> assistant -> user -> assistant."""
        acc = create_accumulator(thinking=thinking)
        tokenizer = acc.tokenizer

        # Turn 1
        assert acc.add_user("Hi")
        resp1 = mock_vllm_response(tokenizer, "Hello!")
        assert acc.add_assistant("Hello!", resp1)

        # Turn 2
        assert acc.add_user("Bye")
        resp2 = mock_vllm_response(tokenizer, "Goodbye!")
        assert acc.add_assistant("Goodbye!", resp2)

        episode = acc.get_data()
        assert not acc.truncated
        assert_no_training_after_eos(
            episode.token_ids.tolist(),
            episode.response_mask.tolist(),
            tokenizer.eos_token_id,
        )


@pytest.mark.parametrize("thinking", [True, False])
class TestBudgetAndTruncation:
    """Budget limits and truncation behavior."""

    def test_user_message_truncated(self, thinking):
        """Test: User message exceeds budget."""
        acc = create_accumulator(thinking=thinking, max_len=50)

        long_message = "word " * 100  # Way over budget
        success = acc.add_user(long_message)

        assert not success, "Should truncate user message"
        assert acc.truncated
        assert acc.truncation_reason == TruncationReason.USER_TOO_LONG

    def test_assistant_response_exceeds_budget(self, thinking):
        """Test: Assistant response exceeds budget."""
        acc = create_accumulator(thinking=thinking, max_len=100)
        tokenizer = acc.tokenizer

        acc.add_user("Hi")

        # Create response that exceeds remaining budget
        long_response = mock_vllm_response(tokenizer, "word " * 200, include_eos=True)
        success = acc.add_assistant("long response", long_response)

        assert not success, "Should reject oversized response"
        assert acc.truncated
        assert acc.truncation_reason == TruncationReason.ASSISTANT_TOO_LONG

    def test_zero_budget_user(self, thinking):
        """Test: Cannot add user message when budget=0."""
        system_content = "helpful " * 100  # Fill the budget
        acc = create_accumulator(
            thinking=thinking,
            max_len=100,
            system_content=system_content,
        )

        assert acc.budget == 0
        success = acc.add_user("Hi")

        assert not success, "Should fail with zero budget"

    def test_zero_budget_assistant(self, thinking):
        """Test: Cannot add assistant response when budget=0."""
        system_content = "helpful " * 100
        acc = create_accumulator(
            thinking=thinking,
            max_len=100,
            system_content=system_content,
        )
        tokenizer = acc.tokenizer

        assert acc.budget == 0
        response = mock_vllm_response(tokenizer, "Hi", include_eos=True)
        success = acc.add_assistant("Hi", response)

        assert not success, "Should fail with zero budget"

    def test_initial_messages_too_long(self, thinking):
        """Test: Initial system message exceeds max_len."""
        long_system = "You are helpful." * 20
        acc = create_accumulator(
            thinking=thinking, max_len=50, system_content=long_system
        )

        assert acc.truncated
        assert acc.truncation_reason == TruncationReason.USER_TOO_LONG
        assert len(acc._tokens) <= 50
        assert acc.budget == 0


@pytest.mark.parametrize("thinking", [True, False])
class TestResponseMaskCorrectness:
    """Verify response_mask correctness (the core bug fix)."""

    def test_generation_prompt_not_trainable(self, thinking):
        """Test: Generation prompt tokens have response_mask=False."""
        acc = create_accumulator(thinking=thinking)
        tokenizer = acc.tokenizer

        initial_len = len(acc._tokens)
        acc.add_user("Hi")
        response = mock_vllm_response(tokenizer, "Hello!", include_eos=True)
        acc.add_assistant("Hello!", response)

        # Count non-trainable tokens after initial messages
        # Should be: user message tokens + generation prompt tokens
        non_trainable_after_initial = sum(not mask for mask in acc._mask[initial_len:])

        # Generation prompt should not be trainable
        assert non_trainable_after_initial >= acc.gen_prompt_len, (
            f"Generation prompt ({acc.gen_prompt_len} tokens) should not be trainable, "
            f"but only {non_trainable_after_initial} non-trainable tokens found"
        )

    def test_vllm_tokens_trainable(self, thinking):
        """Test: All vLLM tokens (including EOS) are trainable."""
        acc = create_accumulator(thinking=thinking)
        tokenizer = acc.tokenizer

        initial_tokens = len(acc._tokens)
        acc.add_user("Hi")
        after_user = len(acc._tokens)

        response = mock_vllm_response(tokenizer, "Hello!", include_eos=True)
        acc.add_assistant("Hello!", response)

        # Count trainable tokens added by assistant response
        # Skip: initial + user message + generation prompt
        assistant_start = after_user + acc.gen_prompt_len
        trainable_assistant = sum(acc._mask[assistant_start:])

        assert trainable_assistant == len(response), (
            f"All {len(response)} vLLM tokens should be trainable, "
            f"got {trainable_assistant}"
        )

        # EOS should be trainable (it's before the suffix)
        # Find EOS position (should be len - suffix_len - 1)
        eos_pos = len(acc._tokens) - len(acc.suffix) - 1
        assert (
            acc._tokens[eos_pos] == tokenizer.eos_token_id
        ), f"Expected EOS at position {eos_pos}, got {acc._tokens[eos_pos]}"
        assert acc._mask[eos_pos] == True, "EOS token must be trainable"

    def test_no_training_after_eos_single_turn(self, thinking):
        """Test: No trainable tokens after EOS (single turn)."""
        acc = create_accumulator(thinking=thinking)
        tokenizer = acc.tokenizer

        acc.add_user("Hi")
        response = mock_vllm_response(tokenizer, "Hello!", include_eos=True)
        acc.add_assistant("Hello!", response)

        assert_no_training_after_eos(acc._tokens, acc._mask, tokenizer.eos_token_id)

    def test_no_training_after_eos_multi_turn(self, thinking):
        """Test: No trainable tokens after EOS (multi-turn)."""
        acc = create_accumulator(thinking=thinking)
        tokenizer = acc.tokenizer

        # Turn 1
        acc.add_user("Hi")
        acc.add_assistant("Hello!", mock_vllm_response(tokenizer, "Hello!"))

        # Turn 2
        acc.add_user("Bye")
        acc.add_assistant("Goodbye!", mock_vllm_response(tokenizer, "Goodbye!"))

        # Turn 3
        acc.add_user("See you")
        acc.add_assistant("Take care!", mock_vllm_response(tokenizer, "Take care!"))

        # Check no training after ANY EOS
        assert_no_training_after_eos(acc._tokens, acc._mask, tokenizer.eos_token_id)

    def test_eos_token_is_trainable(self, thinking):
        """Test: EOS token itself should be trainable."""
        acc = create_accumulator(thinking=thinking)
        tokenizer = acc.tokenizer

        acc.add_user("Hi")
        response = mock_vllm_response(tokenizer, "Hello!", include_eos=True)
        acc.add_assistant("Hello!", response)

        # Find all EOS positions
        eos_positions = [
            i for i, t in enumerate(acc._tokens) if t == tokenizer.eos_token_id
        ]

        # Last EOS (from assistant) should be trainable
        # Earlier EOS (from system/user) should NOT be trainable
        assistant_eos = eos_positions[-1]
        assert acc._mask[assistant_eos], "Assistant EOS must be trainable"


@pytest.mark.parametrize("thinking", [True, False])
class TestMultiTurnTruncation:
    """Multi-turn truncation scenarios."""

    def test_second_user_message_truncated(self, thinking):
        """Test: Second user message causes truncation."""
        acc = create_accumulator(thinking=thinking, max_len=100)
        tokenizer = acc.tokenizer

        # Turn 1 - should succeed
        acc.add_user("Say hi")
        resp1 = mock_vllm_response(tokenizer, "Hello! How can I help?")
        acc.add_assistant("Hello! How can I help?", resp1)

        # Turn 2 - long user message should truncate
        long_user = "This is a very long message. " * 20
        success = acc.add_user(long_user)

        assert not success, "Long user message should be truncated"
        assert acc.truncated
        assert acc.truncation_reason == TruncationReason.USER_TOO_LONG

    def test_second_assistant_response_truncated(self, thinking):
        """Test: Second assistant response exceeds budget."""
        acc = create_accumulator(thinking=thinking, max_len=100)
        tokenizer = acc.tokenizer

        # Turn 1
        acc.add_user("Hi")
        resp1 = mock_vllm_response(tokenizer, "Hello! How can I assist you today?")
        acc.add_assistant("Hello! How can I assist you today?", resp1)

        # Turn 2 - should fit
        acc.add_user("Bye")

        # Long response should be rejected
        long_response = mock_vllm_response(tokenizer, "word " * 100, include_eos=True)
        success = acc.add_assistant("long response", long_response)

        assert not success, "Long response should be rejected"
        assert acc.truncated
        assert acc.truncation_reason == TruncationReason.ASSISTANT_TOO_LONG


# ============================================================================
# Comparison Tests
# ============================================================================


def test_thinking_affects_generation_prompt_length():
    """Verify thinking changes generation prompt length."""
    acc_thinking = create_accumulator(thinking=True)
    acc_no_thinking = create_accumulator(thinking=False)

    # Qwen-specific behavior: thinking disabled adds placeholder tags
    if "Qwen" in MODEL_NAME:
        assert acc_thinking.gen_prompt_len < acc_no_thinking.gen_prompt_len
    else:
        # For models without thinking support, lengths should be equal
        assert acc_thinking.gen_prompt_len == acc_no_thinking.gen_prompt_len


def test_thinking_affects_budget():
    """Verify thinking changes budget calculations."""
    acc_thinking = create_accumulator(thinking=True, max_len=1000)
    acc_no_thinking = create_accumulator(thinking=False, max_len=1000)

    # Qwen-specific behavior: thinking enabled has larger budget
    if "Qwen" in MODEL_NAME:
        assert acc_thinking.budget > acc_no_thinking.budget
    else:
        # For models without thinking support, budgets should be equal
        assert acc_thinking.budget == acc_no_thinking.budget


def test_thinking_affects_total_tokens():
    """Verify thinking changes accumulated token count."""
    tokenizer = get_tokenizer(MODEL_NAME)

    acc_thinking = create_accumulator(thinking=True)
    acc_no_thinking = create_accumulator(thinking=False)

    # Add same conversation to both
    for acc in [acc_thinking, acc_no_thinking]:
        acc.add_user("Hi")
        response = mock_vllm_response(tokenizer, "Hello!")
        acc.add_assistant("Hello!", response)

    # Qwen-specific behavior: thinking disabled has more tokens
    if "Qwen" in MODEL_NAME:
        assert len(acc_thinking._tokens) < len(acc_no_thinking._tokens)
    else:
        # For models without thinking support, token counts should be equal
        assert len(acc_thinking._tokens) == len(acc_no_thinking._tokens)


# ============================================================================
# Golden Test - Exact Token/Mask Validation
# ============================================================================


def test_exact_token_and_mask_sequence_qwen():
    """
    Golden test: Verify EXACT token sequence and response_mask for a known conversation.

    This test uses hardcoded Qwen tokenizer to ensure we catch any regressions in:
    - Token ordering
    - Mask alignment
    - Generation prompt placement
    - vLLM response token handling
    - Suffix token handling

    Conversation:
    - System: "Help"
    - User: "Hi" → Assistant: "hello there"
    - User: "i am bob" → Assistant: "Hi Bob"
    """
    # Hardcode Qwen tokenizer for this golden test
    tokenizer = get_tokenizer("Qwen/Qwen3-1.7B")

    acc = TokenAccumulator(
        tokenizer=tokenizer,
        messages=[{"role": "system", "content": "Help"}],
        max_len=2048,
        eos_id=tokenizer.eos_token_id,
        thinking=False,
    )

    # Turn 1
    acc.add_user("Hi")
    resp1 = [14990, 1052, 151645]  # "hello there" + EOS
    acc.add_assistant("hello there", resp1)

    # Turn 2
    acc.add_user("i am bob")
    resp2 = [13048, 14261, 151645]  # "Hi Bob" + EOS
    acc.add_assistant("Hi Bob", resp2)

    # Expected tokens (golden values WITH suffix tokens after each EOS)
    expected_tokens = [
        151644,
        8948,
        198,
        12689,
        151645,
        198,  # System
        151644,
        872,
        198,
        13048,
        151645,
        198,  # User 1
        151644,
        77091,
        198,
        151667,
        271,
        151668,
        271,  # Gen prompt 1
        14990,
        1052,
        151645,  # Response 1
        198,  # Suffix 1
        151644,
        872,
        198,
        72,
        1079,
        35192,
        151645,
        198,  # User 2
        151644,
        77091,
        198,
        151667,
        271,
        151668,
        271,  # Gen prompt 2
        13048,
        14261,
        151645,  # Response 2
        198,  # Suffix 2
    ]

    # Expected mask (only vLLM response tokens are trainable, suffix is NOT trainable)
    expected_mask = [
        False,
        False,
        False,
        False,
        False,
        False,  # [0-5]: System
        False,
        False,
        False,
        False,
        False,
        False,  # [6-11]: User 1
        False,
        False,
        False,
        False,
        False,
        False,
        False,  # [12-18]: Gen prompt 1
        True,
        True,
        True,  # [19-21]: Response 1 (trainable!)
        False,  # [22]: Suffix 1
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,  # [23-30]: User 2
        False,
        False,
        False,
        False,
        False,
        False,
        False,  # [31-37]: Gen prompt 2
        True,
        True,
        True,  # [38-40]: Response 2 (trainable!)
        False,  # [41]: Suffix 2
    ]

    # Verify exact sequence
    assert acc._tokens == expected_tokens, (
        f"Token mismatch!\n"
        f"Expected: {expected_tokens}\n"
        f"Got:      {acc._tokens}\n"
        f"\nFirst diff at index {next((i for i, (a, b) in enumerate(zip(expected_tokens, acc._tokens)) if a != b), -1)}"
    )

    assert acc._mask == expected_mask, (
        f"Mask mismatch!\n"
        f"Expected: {expected_mask}\n"
        f"Got:      {acc._mask}\n"
        f"\nFirst diff at index {next((i for i, (a, b) in enumerate(zip(expected_mask, acc._mask)) if a != b), -1)}"
    )

    # Verify trainable count (only the 6 vLLM response tokens)
    assert (
        sum(expected_mask) == 6
    ), "Should have exactly 6 trainable tokens (2 responses × 3 tokens each)"

    # Verify EOS positions are trainable
    eos_positions = [i for i, t in enumerate(expected_tokens) if t == 151645]
    assistant_eos_positions = [
        21,
        40,
    ]  # Positions of assistant EOS tokens (before suffix)
    for pos in assistant_eos_positions:
        assert pos in eos_positions, f"Expected EOS at position {pos}"
        assert expected_mask[pos], f"Assistant EOS at position {pos} must be trainable"

    # Verify no training after EOS
    assert_no_training_after_eos(expected_tokens, expected_mask, tokenizer.eos_token_id)


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
