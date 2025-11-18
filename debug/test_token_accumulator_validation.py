#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Minimal validation test for TokenAccumulator v9 fix.

Tests 4 scenarios using actual vLLM:
1. prompt -> user -> assistant (COMPLETE)
2. prompt -> user -> assistant-truncated (DROPPED)
3. prompt -> user -> assistant -> user (COMPLETE MULTI-TURN)
4. prompt -> user -> assistant-truncated -> user-truncated (DROPPED)

Expected results:
- Test 1, 3: Should PASS (complete responses, no duplicates)
- Test 2, 4: Should be DROPPED (truncated episodes rejected)
"""

import asyncio
import sys

sys.path.insert(0, "/home/felipemello/forge/debug")

from forge.actors.generator import Generator
from token_accumulator_fn_v4 import SanityCheckMode, TokenAccumulator, TruncationReason
from transformers import AutoTokenizer
from vllm.engine.arg_utils import EngineArgs
from vllm.sampling_params import SamplingParams


async def test_scenario_1_complete(tokenizer, generator):
    """Test 1: prompt -> user -> assistant (COMPLETE)"""
    print("\n" + "=" * 5)
    print("TEST 1: prompt -> user -> assistant (COMPLETE)")
    print("=" * 5)

    # Initialize accumulator
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant.",
        }
    ]
    acc = TokenAccumulator(
        tokenizer=tokenizer,
        messages=messages,
        max_seq_len=2048,
        eos_token_id=tokenizer.eos_token_id,
        sanity_check_mode=SanityCheckMode.STRICT,
    )

    # Add user message with trivial task
    acc.add_user_message("Just reply to me with 'hi'. Do not think about it.")

    # Generate with vLLM (high max_tokens to ensure completion)
    prompt = acc.format_prompt()
    sampling_params = SamplingParams(temperature=0.0, top_p=0.9, max_tokens=1000)
    completions = await generator.generate.route(
        prompt, sampling_params=sampling_params
    )
    completion = completions[0]

    print(f"Response text: {repr(completion.text)}")
    print(f"Stop reason: {completion.stop_reason}")
    print(
        f"Last token == EOS: {completion.token_ids.tolist()[-1] == tokenizer.eos_token_id}"
    )

    # Add assistant response
    success = acc.add_assistant_response(
        response_text=completion.text,
        response_token_ids=completion.token_ids.tolist(),
    )

    print(
        f"\nEpisode accepted: {success}, Is truncated: {acc.is_truncated}, Truncation reason: {acc.truncation_reason}"
    )

    # Always show decoded conversation
    print("\n" + "-" * 5)
    print("DECODED CONVERSATION:")
    print("-" * 5)
    decoded = tokenizer.decode(acc.accumulated_tokens)
    print(decoded)
    print("-" * 5)

    errors = []

    if success:
        print(f"Total tokens: {len(acc.accumulated_tokens)}")

        # Validate
        try:
            acc.finalize()
            print("✅ FINALIZE PASSED")
        except ValueError as e:
            errors.append(f"FINALIZE FAILED: {e}")
    else:
        errors.append("Episode was DROPPED (expected to be accepted)")
        errors.append(
            f"Response was truncated at {len(completion.token_ids.tolist())} tokens"
        )
        errors.append("This test expects a COMPLETE response, not truncated")

    if errors:
        print("\n❌ ERRORS FOUND:")
        for error in errors:
            print(f"  - {error}")
        return False

    return True


async def test_scenario_2_truncated(tokenizer, generator):
    """Test 2: prompt -> user -> assistant-truncated (DROPPED)"""
    print("\n" + "=" * 5)
    print("TEST 2: prompt -> user -> assistant-truncated (DROPPED)")
    print("=" * 5)

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant.",
        }
    ]
    acc = TokenAccumulator(
        tokenizer=tokenizer,
        messages=messages,
        max_seq_len=2048,
        eos_token_id=tokenizer.eos_token_id,
        sanity_check_mode=SanityCheckMode.STRICT,
    )

    acc.add_user_message("Just reply to me with 'hi'. Do not think about it.")

    # Force truncation with very low max_tokens
    prompt = acc.format_prompt()
    sampling_params = SamplingParams(temperature=0.0, top_p=0.9, max_tokens=1)
    completions = await generator.generate.route(
        prompt, sampling_params=sampling_params
    )
    completion = completions[0]

    print(f"Response text: {repr(completion.text)}")
    print(f"Stop reason: {completion.stop_reason}")
    print(
        f"Last token == EOS: {completion.token_ids.tolist()[-1] == tokenizer.eos_token_id}"
    )

    # Try to add assistant response
    success = acc.add_assistant_response(
        response_text=completion.text,
        response_token_ids=completion.token_ids.tolist(),
    )

    print(
        f"\nEpisode accepted: {success}, Is truncated: {acc.is_truncated}, Truncation reason: {acc.truncation_reason}"
    )
    print(f"Remaining budget after truncation: {acc.get_remaining_budget()}")
    print(
        f"Current tokens: {len(acc.accumulated_tokens)}, max_seq_len: {acc.max_seq_len}"
    )

    # Always show decoded conversation
    print("DECODED CONVERSATION (what was accumulated BEFORE drop):")
    decoded = tokenizer.decode(acc.accumulated_tokens)
    print("-" * 5, decoded, "-" * 5)

    if success:
        print("\n❌ ERRORS FOUND:")
        print("  - Truncated episode was accepted (should be dropped)!")
        return False

    print(
        f"✅ PASS: Total tokens in accumulator: {len(acc.accumulated_tokens)} (only initial messages)"
    )
    return True


async def test_scenario_3_multiturn(tokenizer, generator):
    """Test 3: prompt -> user -> assistant -> user (COMPLETE MULTI-TURN)"""
    print("\n" + "=" * 5)
    print("TEST 3: prompt -> user -> assistant -> user (COMPLETE MULTI-TURN)")
    print("=" * 5)

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant.",
        }
    ]
    acc = TokenAccumulator(
        tokenizer=tokenizer,
        messages=messages,
        max_seq_len=2048,
        eos_token_id=tokenizer.eos_token_id,
        sanity_check_mode=SanityCheckMode.STRICT,
    )

    # Turn 1
    print("\nTurn 1:")
    acc.add_user_message("Just reply to me with 'hi'. Do not think about it.")
    prompt = acc.format_prompt()
    sampling_params = SamplingParams(temperature=0.0, top_p=0.9, max_tokens=1000)
    completions = await generator.generate.route(
        prompt, sampling_params=sampling_params
    )
    completion = completions[0]

    print(f"  Response: {repr(completion.text)}")
    print(f"  Tokens: {len(completion.token_ids.tolist())}")
    print(f"  Stop reason: {completion.stop_reason}")
    print(
        f"  Last token == EOS: {completion.token_ids.tolist()[-1] == tokenizer.eos_token_id}"
    )

    success = acc.add_assistant_response(
        response_text=completion.text,
        response_token_ids=completion.token_ids.tolist(),
    )

    # Always show state after turn 1
    print("\n" + "-" * 5)
    print("DECODED CONVERSATION (after turn 1 attempt):")
    print("-" * 5)
    decoded = tokenizer.decode(acc.accumulated_tokens)
    print(decoded)
    print("-" * 5)

    # Collect errors instead of failing early
    errors = []

    if not success:
        errors.append("Turn 1 truncated - test expected success")
        errors.append(
            f"Response was truncated at {len(completion.token_ids.tolist())} tokens"
        )

    # Turn 2 - just add user message
    print("\nTurn 2:")
    acc.add_user_message("Now say 'bye'.")

    # Validate
    try:
        acc.finalize()
        print("✅ FINALIZE PASSED")
    except ValueError as e:
        errors.append(f"FINALIZE FAILED: {e}")

    # Check for duplicates in the decoded output
    decoded_final = tokenizer.decode(acc.accumulated_tokens)
    print("\nFINAL DECODED CONVERSATION:")
    print("-" * 5)
    print(decoded_final)
    print("-" * 5)
    print(f"   Total tokens in accumulator: {len(acc.accumulated_tokens)}")

    # Check for duplicate thinking tags (the main bug we're trying to avoid)
    if decoded_final.count("<think>") > decoded_final.count("</think>") + 1:
        errors.append("Found unclosed <think> tags!")

    if "<think>" in decoded_final and "</think>" in decoded_final:
        # Count occurrences - should match
        think_open_count = decoded_final.count("<think>")
        think_close_count = decoded_final.count("</think>")
        if think_open_count != think_close_count:
            errors.append(
                f"Mismatched thinking tags! Open: {think_open_count}, Close: {think_close_count}"
            )
        else:
            print(f"✅ Thinking tags are balanced ({think_open_count} pairs)")

    # Report all errors at once
    if errors:
        print("\n❌ ERRORS FOUND:")
        for error in errors:
            print(f"  - {error}")
        return False

    return True


async def test_scenario_4_truncated_multiturn(tokenizer, generator):
    """Test 4: prompt -> user -> assistant -> user-truncated (DROPPED)"""
    print("\n" + "=" * 5)
    print("TEST 4: prompt -> user -> assistant -> user-truncated (DROPPED)")
    print("=" * 5)

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant.",
        }
    ]
    acc = TokenAccumulator(
        tokenizer=tokenizer,
        messages=messages,
        max_seq_len=180,
        eos_token_id=tokenizer.eos_token_id,
        sanity_check_mode=SanityCheckMode.STRICT,
    )

    # Turn 1 - complete generation
    print("\nTurn 1")
    acc.add_user_message("Just reply to me with 'hi'. Do not think about it.")
    prompt = acc.format_prompt()

    # ✅ Use get_remaining_budget() to prevent overflow
    remaining = acc.get_remaining_budget()
    print(f"  Remaining budget before generation: {remaining}")
    sampling_params = SamplingParams(temperature=0.0, top_p=0.9, max_tokens=remaining)
    completions = await generator.generate.route(
        prompt, sampling_params=sampling_params
    )
    completion = completions[0]

    print(f"  Response: {repr(completion.text)}")
    print(f"  Tokens: {len(completion.token_ids.tolist())}")
    print(f"  Stop reason: {completion.stop_reason}")
    print(
        f"  Last token == EOS: {completion.token_ids.tolist()[-1] == tokenizer.eos_token_id}"
    )

    success = acc.add_assistant_response(
        response_text=completion.text,
        response_token_ids=completion.token_ids.tolist(),
    )

    print("TOTAL TOKENS IN ACCUMULATOR: ", len(acc.accumulated_tokens))
    print("get_remaining_budget: ", acc.get_remaining_budget())
    print("max_seq_len: ", acc.max_seq_len)

    success = acc.add_user_message("This is a very long message" * 100)

    print(
        f"\nUser message accepted: {success}, Is truncated: {acc.is_truncated}, Truncation reason: {acc.truncation_reason}"
    )
    print(f"Remaining budget after user truncation: {acc.get_remaining_budget()}")
    print(
        f"Current tokens: {len(acc.accumulated_tokens)}, max_seq_len: {acc.max_seq_len}"
    )

    # Always show decoded conversation
    print("\nDECODED CONVERSATION (what was accumulated before/during truncation):")
    decoded = tokenizer.decode(acc.accumulated_tokens)
    print(decoded)
    print("-" * 5)
    print(f"   Total tokens in accumulator: {len(acc.accumulated_tokens)}")

    # Collect all errors instead of failing early
    errors = []

    # The test expects truncation
    if not acc.is_truncated:
        errors.append("Episode should have been truncated!")

    if acc.truncation_reason != TruncationReason.USER_TOO_LONG:
        errors.append(f"Wrong truncation reason: {acc.truncation_reason}")

    # ✅ Critical check: After user truncation, budget MUST be 0
    # If budget > 0, that's a bug in truncation logic that could allow agent responses
    # to be generated and added even though episode is already truncated
    remaining_budget = acc.get_remaining_budget()
    if remaining_budget > 0:
        errors.append(
            f"Budget calculation bug! After user truncation, budget should be 0, got {remaining_budget}"
        )
        errors.append(
            "This could allow agent responses to be added to truncated episodes!"
        )

    # ✅ Verify we never exceeded max_seq_len
    if len(acc.accumulated_tokens) > acc.max_seq_len:
        errors.append(
            f"Budget overflow! {len(acc.accumulated_tokens)} > {acc.max_seq_len}"
        )

    # Report all errors at once
    if errors:
        print("\n❌ ERRORS FOUND:")
        for error in errors:
            print(f"  - {error}")
        return False

    print("✅ PASS: Episode correctly marked as truncated")
    print(
        f"✅ PASS: Budget respected ({len(acc.accumulated_tokens)} <= {acc.max_seq_len})"
    )
    return True


def test_initial_messages_too_long(tokenizer):
    """Test 5: Initial messages exceed max_seq_len"""
    print("\n" + "=" * 5)
    print("TEST 5: Initial messages > max_seq_len")
    print("=" * 5)

    # Create very long system message
    long_system = "You are helpful. " * 100  # Very long
    messages = [{"role": "system", "content": long_system}]

    acc = TokenAccumulator(
        tokenizer=tokenizer,
        messages=messages,
        max_seq_len=50,  # Tiny budget
        eos_token_id=tokenizer.eos_token_id,
    )

    print(
        f"Initial tokens: {len(acc.accumulated_tokens)}, max_seq_len: {acc.max_seq_len}"
    )
    print(f"is_truncated: {acc.is_truncated}")
    print(f"truncation_reason: {acc.truncation_reason}")
    print(f"Remaining budget: {acc.get_remaining_budget()}")

    # Show decoded conversation
    print("\nDECODED CONVERSATION:")
    decoded = tokenizer.decode(acc.accumulated_tokens)
    print("-" * 5)
    print(decoded)
    print("-" * 5)

    # Collect errors
    errors = []

    # Check truncation
    if not acc.is_truncated:
        errors.append("Should be marked truncated!")

    if acc.truncation_reason != TruncationReason.USER_TOO_LONG:
        errors.append(f"Wrong truncation type: {acc.truncation_reason}")

    if len(acc.accumulated_tokens) != acc.max_seq_len:
        errors.append(
            f"Should be truncated to {acc.max_seq_len}, got {len(acc.accumulated_tokens)}"
        )

    if errors:
        print("\n❌ ERRORS FOUND:")
        for error in errors:
            print(f"  - {error}")
        return False

    # Budget might not be exactly 0 due to assistant_overhead subtraction
    print(f"✅ PASS: Initial messages correctly truncated")
    print(
        f"   Note: Remaining budget = {acc.get_remaining_budget()} (may be >0 due to overhead calculation)"
    )
    return True


def test_zero_budget_user_message(tokenizer):
    """Test 6: Try to add user message with zero budget"""
    print("\n" + "=" * 5)
    print("TEST 6: Add user message with budget=0")
    print("=" * 5)

    messages = [
        {"role": "system", "content": "You are helpful." * 50}
    ]  # Takes all budget
    acc = TokenAccumulator(
        tokenizer=tokenizer,
        messages=messages,
        max_seq_len=100,
        eos_token_id=tokenizer.eos_token_id,
    )

    initial_len = len(acc.accumulated_tokens)
    print(f"Initial: {initial_len} tokens, budget: {acc.get_remaining_budget()}")

    # Try to add user message (budget should be ~0 or negative)
    success = acc.add_user_message("Hello")

    print(f"After add_user: {len(acc.accumulated_tokens)} tokens")
    print(f"success: {success}, is_truncated: {acc.is_truncated}")
    print(f"Remaining budget after attempt: {acc.get_remaining_budget()}")

    # Show decoded conversation
    print("\nDECODED CONVERSATION:")
    decoded = tokenizer.decode(acc.accumulated_tokens)
    print("-" * 5)
    print(decoded)
    print("-" * 5)

    errors = []

    # Should fail and not add anything (or add 0 tokens if budget was exactly 0)
    if success:
        errors.append("Should have failed (no budget)")

    if (
        len(acc.accumulated_tokens) > initial_len + 1
    ):  # Allow at most 1 token if budget allowed
        errors.append(
            f"Added too many tokens! {len(acc.accumulated_tokens) - initial_len}"
        )

    if errors:
        print("\n❌ ERRORS FOUND:")
        for error in errors:
            print(f"  - {error}")
        return False

    print("✅ PASS: User message correctly rejected/truncated with zero budget")
    return True


def test_zero_budget_assistant_message(tokenizer):
    """Test 7: Try to add assistant message with zero budget"""
    print("\n" + "=" * 5)
    print("TEST 7: Add assistant message with budget=0")
    print("=" * 5)

    messages = [{"role": "system", "content": "You are helpful." * 50}]
    acc = TokenAccumulator(
        tokenizer=tokenizer,
        messages=messages,
        max_seq_len=100,
        eos_token_id=tokenizer.eos_token_id,
    )

    initial_len = len(acc.accumulated_tokens)
    budget = acc.get_remaining_budget()
    print(f"Initial: {initial_len} tokens, budget: {budget}")

    # Assistant response with EOS
    response_token_ids = [6151, tokenizer.eos_token_id]  # "hi" + EOS

    success = acc.add_assistant_response("hi", response_token_ids)

    print(f"After add_assistant: {len(acc.accumulated_tokens)} tokens")
    print(f"success: {success}")
    print(f"Remaining budget after attempt: {acc.get_remaining_budget()}")

    # Show decoded conversation
    print("\nDECODED CONVERSATION:")
    decoded = tokenizer.decode(acc.accumulated_tokens)
    print("-" * 5)
    print(decoded)
    print("-" * 5)

    # With zero/low budget, the assistant response should be rejected
    # The key test is that we don't overflow max_seq_len
    if len(acc.accumulated_tokens) > acc.max_seq_len:
        print(
            f"❌ ERROR: Exceeded max_seq_len! {len(acc.accumulated_tokens)} > {acc.max_seq_len}"
        )
        return False

    # With the budget check, this should now be rejected
    if success and budget == 0:
        print("❌ ERROR: Assistant response should have been rejected (zero budget)")
        return False

    print("✅ PASS: Assistant message handled correctly with zero budget")
    return True


async def main():
    # Setup
    model_path = "Qwen/Qwen3-1.7B"  # "meta-llama/Meta-Llama-3.1-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)

    print(f"Model: {model_path}")
    print(f"EOS token: {tokenizer.eos_token} (id={tokenizer.eos_token_id})")

    # Start generator
    engine_args = EngineArgs(
        model=model_path,
        tensor_parallel_size=1,
        max_model_len=2048,
        enable_prefix_caching=True,
    )

    generator = await Generator.options(
        procs=1,
        num_replicas=1,
        with_gpus=True,
    ).as_service(
        engine_args=engine_args,
        sampling_params=SamplingParams(),
    )

    print("✅ Generator ready\n")

    # Run tests
    results = []

    results.append(
        ("Test 1 (complete)", await test_scenario_1_complete(tokenizer, generator))
    )
    results.append(
        (
            "Test 2 (truncated-drop)",
            await test_scenario_2_truncated(tokenizer, generator),
        )
    )
    results.append(
        (
            "Test 3 (multi-turn)",
            await test_scenario_3_multiturn(tokenizer, generator),
        )
    )
    results.append(
        (
            "Test 4 (multi-turn-truncated-drop)",
            await test_scenario_4_truncated_multiturn(tokenizer, generator),
        )
    )
    results.append(
        ("Test 5 (initial-too-long)", test_initial_messages_too_long(tokenizer))
    )
    results.append(
        ("Test 6 (zero-budget-user)", test_zero_budget_user_message(tokenizer))
    )
    results.append(
        (
            "Test 7 (zero-budget-assistant)",
            test_zero_budget_assistant_message(tokenizer),
        )
    )
    # Summary
    print("\n" + "=" * 5)
    print("SUMMARY")
    print("=" * 5)

    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {name}")

    all_passed = all(p for _, p in results)
    print("\n" + "=" * 5)
    if all_passed:
        print("✅✅✅ ALL TESTS PASSED ✅✅✅")
        print("\nThe v9 fix works correctly:")
        print("  1. Complete responses match ground truth (no token mismatch)")
        print("  2. No duplicate <think> tags in decoded output")
        print("  3. Truncated episodes are correctly dropped")
        print("  4. Multi-turn conversations work correctly")
    else:
        print("❌❌❌ SOME TESTS FAILED ❌❌❌")
        print("\nPlease check the output above for details")
    print("=" * 5)


if __name__ == "__main__":
    asyncio.run(main())
