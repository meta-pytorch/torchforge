"""
Multi-turn token accumulation with BASE anchor pattern.

Features:
- BASE anchor: Tokenize BASE + 1 message (O(N) instead of O(N²))
- Automatic role headers: Delta extraction includes chat template formatting
- Immediate env obs accumulation
- Finalize validation: Detects tokenization mismatches
- Configurable sanity check modes

Test cases:
1. Normal rollout (no truncation)
2. vLLM truncation (generation hits max_tokens)
3. Env observation truncation (adding env obs exceeds max_seq_len)
4. Early exit (initial prompt already exceeds max_seq_len)
5. Long env observation (truncate mid-content)
"""

from enum import Enum
from functools import lru_cache

import torch
from vllm import LLM, SamplingParams
from vllm.transformers_utils.tokenizer import get_tokenizer


def test_normal_rollout(llm, tokenizer, max_seq_len: int, max_turns: int):
    """Test rollout with NO truncation (normal case)"""

    print("\n" + "=" * 80)
    print("TEST CASE 1: NORMAL ROLLOUT (NO TRUNCATION)")
    print("=" * 80)

    messages = [
        {
            "role": "system",
            "content": "You are an expert BlackJack player. Output only 'HIT' or 'STAND'.",
        }
    ]

    accumulator = TokenAccumulator(
        tokenizer=tokenizer,
        messages=messages,
        max_seq_len=max_seq_len,
        eos_token_id=tokenizer.eos_token_id,
        sanity_check_mode=SanityCheckMode.STRICT,
    )

    accumulator.add_user_message("Hand: 15, Dealer: 10", check_budget=False)

    sampling_params = SamplingParams(
        temperature=0.8,
        max_tokens=50,
        logprobs=1,
    )

    for turn in range(max_turns):
        print(f"\n{'='*60}")
        print(f"TURN {turn + 1}")
        print(f"{'='*60}")

        remaining = accumulator.get_remaining_budget()

        print(f"\n[Budget Check]")
        print(f"  Current tokens: {len(accumulator.all_tokens)}")
        print(f"  Assistant overhead: {accumulator.assistant_overhead}")
        print(f"  Max seq len: {max_seq_len}")
        print(f"  Remaining: {remaining}")

        if remaining <= 0:
            print(f"  ❌ Out of budget!")
            break

        prompt_text = accumulator.format_prompt()

        print(f"\n[Generation]")
        print(f"  Generating...")

        sampling_params.max_tokens = min(remaining, 50)
        outputs = llm.generate([prompt_text], sampling_params)
        output = outputs[0].outputs[0]

        response_text = output.text
        response_tokens = output.token_ids

        response_logprobs = None
        if output.logprobs is not None:
            response_logprobs = [
                lp[token_id] for lp, token_id in zip(output.logprobs, response_tokens)
            ]

        print(f"  Response: '{response_text}'")
        print(f"  Response token_ids: {len(response_tokens)} tokens (content only)")
        print(f"  Stop reason: {output.stop_reason}")

        success = accumulator.add_assistant_response(
            response_text=response_text,
            response_token_ids=response_tokens,
            response_logprobs=response_logprobs,
        )

        ground_truth_before = tokenizer.apply_chat_template(
            accumulator.messages[:-1], add_generation_prompt=False, tokenize=True
        )
        ground_truth_after = tokenizer.apply_chat_template(
            accumulator.messages, add_generation_prompt=False, tokenize=True
        )
        assistant_tokens_added = len(ground_truth_after) - len(ground_truth_before)

        print(f"  Assistant tokens added: {assistant_tokens_added}")
        print(f"  Total tokens now: {len(accumulator.all_tokens)}")

        if success:
            print(f"  ✅ Generation complete (ends with eos)")
        else:
            print(f"  ⚠️  Generation TRUNCATED")

        print(f"\n[Validation]")
        print(f"  all_tokens: {len(accumulator.all_tokens)}")
        ground_truth = tokenizer.apply_chat_template(
            accumulator.messages, add_generation_prompt=False, tokenize=True
        )
        print(f"  ground_truth: {len(ground_truth)}")
        if len(accumulator.all_tokens) == len(ground_truth):
            print(f"  ✅ PERFECT MATCH!")
        else:
            print(f"  ❌ MISMATCH")

        if not success:
            print(f"\n[Episode Truncated]")
            break

        game_done = turn >= 2
        if game_done:
            print(f"\n[Game Done]")
            break

        env_obs = f"Hand: {16 + turn}, Dealer: 10"
        print(f"\n[Env Observation]")
        print(f"  Observation: '{env_obs}'")

        success = accumulator.add_user_message(env_obs, check_budget=True)

        if success:
            print(f"  ✅ Env obs added successfully")
        else:
            print(f"  ⚠️  Env obs would exceed budget - breaking")
            break

    print(f"\n{'='*60}")
    print(f"FINAL VALIDATION")
    print(f"{'='*60}")

    final_ground_truth = tokenizer.apply_chat_template(
        accumulator.messages, add_generation_prompt=False, tokenize=True
    )

    print(f"all_tokens: {len(accumulator.all_tokens)}")
    print(f"ground_truth: {len(final_ground_truth)}")

    if len(accumulator.all_tokens) == len(final_ground_truth):
        print(f"✅ ✅ ✅ PERFECT MATCH! ✅ ✅ ✅")
    else:
        print(f"❌ MISMATCH")
        print(
            f"Difference: {len(final_ground_truth) - len(accumulator.all_tokens)} tokens"
        )

    print(f"\n{'='*60}")
    print(f"DECODED CONVERSATION")
    print(f"{'='*60}")
    decoded = tokenizer.decode(accumulator.all_tokens)
    print(decoded)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Total tokens: {len(accumulator.all_tokens)}")
    print(f"Trainable tokens (mask=1): {sum(accumulator.response_mask)}")
    print(
        f"Non-trainable tokens (mask=0): {len(accumulator.all_tokens) - sum(accumulator.response_mask)}"
    )
    print(
        f"Turns completed: {sum(1 for m in accumulator.messages if m['role'] == 'assistant')}"
    )
    print(f"Response mask: {accumulator.response_mask}")

    print(f"\n{'='*60}")
    print("FINALIZE VALIDATION (VERL pattern)")
    print(f"{'='*60}")
    if accumulator.finalize():
        print("✅ FINALIZE PASSED - BASE anchor accumulation matches ground truth!")
    else:
        print("⚠️  FINALIZE WARNING - see details above")

    return accumulator.all_tokens, accumulator.response_mask, accumulator.messages


def test_vllm_truncation(llm, tokenizer):
    """Test case: vLLM generation hits max_tokens (stop_reason='length')"""

    print("\n" + "=" * 80)
    print("TEST CASE 2: vLLM TRUNCATION (generation hits max_tokens)")
    print("=" * 80)
    print("Setting max_tokens=1 to force mid-word truncation\n")

    messages = [
        {
            "role": "system",
            "content": "You are an expert BlackJack player. Output only 'HIT' or 'STAND'.",
        }
    ]

    accumulator = TokenAccumulator(
        tokenizer=tokenizer,
        messages=messages,
        max_seq_len=2048,
        eos_token_id=tokenizer.eos_token_id,
        sanity_check_mode=SanityCheckMode.STRICT,
    )

    accumulator.add_user_message("Hand: 15, Dealer: 10", check_budget=False)

    sampling_params = SamplingParams(temperature=0.8, max_tokens=1, logprobs=1)

    max_turns = 3

    for turn in range(max_turns):
        print(f"\n{'='*60}")
        print(f"TURN {turn + 1}")
        print(f"{'='*60}")

        remaining = accumulator.get_remaining_budget()
        print(f"\n[Budget Check]")
        print(f"  Remaining: {remaining}")

        if remaining <= 0:
            break

        prompt_text = accumulator.format_prompt()

        print(f"\n[Generation]")
        print(
            f"  Generating with max_tokens={sampling_params.max_tokens} (VERY LOW - will truncate)..."
        )

        outputs = llm.generate([prompt_text], sampling_params)
        output = outputs[0].outputs[0]

        response_text = output.text
        response_tokens = output.token_ids

        response_logprobs = None
        if output.logprobs is not None:
            response_logprobs = [
                lp[token_id] for lp, token_id in zip(output.logprobs, response_tokens)
            ]

        print(f"  Response: '{response_text}'")
        print(f"  Response token_ids: {len(response_tokens)} tokens")
        print(f"  Stop reason: {output.stop_reason}")

        success = accumulator.add_assistant_response(
            response_text=response_text,
            response_token_ids=response_tokens,
            response_logprobs=response_logprobs,
        )

        print(f"  Total tokens now: {len(accumulator.all_tokens)}")

        if not success:
            print(f"\n  ⚠️  ⚠️  ⚠️  GENERATION TRUNCATED! ⚠️  ⚠️  ⚠️")
            print(
                f"  Last token {response_tokens[-1]} != eos_token_id {tokenizer.eos_token_id}"
            )
            print(f"  Setting response_mask=0 for truncated response")
            print(f"  Episode will be marked as truncated")

        print(f"\n[Validation]")
        ground_truth = tokenizer.apply_chat_template(
            accumulator.messages, add_generation_prompt=False, tokenize=True
        )
        print(f"  all_tokens: {len(accumulator.all_tokens)}")
        print(f"  ground_truth: {len(ground_truth)}")

        if len(accumulator.all_tokens) == len(ground_truth):
            print(f"  ✅ PERFECT MATCH!")
        else:
            print(f"  ❌ MISMATCH")

        if not success:
            print(f"\n[Episode Truncated]")
            print(f"  Breaking episode due to generation truncation")
            break

        if turn >= max_turns - 1:
            break

        env_obs = f"Hand: {16 + turn}, Dealer: 10"
        print(f"\n[Env Observation]")
        print(f"  Observation: '{env_obs}'")
        accumulator.add_user_message(env_obs, check_budget=False)

    print(f"\n{'='*60}")
    print(f"FINAL VALIDATION")
    print(f"{'='*60}")

    final_ground_truth = tokenizer.apply_chat_template(
        accumulator.messages, add_generation_prompt=False, tokenize=True
    )

    print(f"all_tokens: {len(accumulator.all_tokens)}")
    print(f"ground_truth: {len(final_ground_truth)}")

    if len(accumulator.all_tokens) == len(final_ground_truth):
        print(f"✅ ✅ ✅ PERFECT MATCH! ✅ ✅ ✅")
    else:
        print(f"❌ MISMATCH")

    print(f"\n{'='*60}")
    print(f"DECODED CONVERSATION")
    print(f"{'='*60}")
    decoded = tokenizer.decode(accumulator.all_tokens)
    print(decoded)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Total tokens: {len(accumulator.all_tokens)}")
    print(f"Trainable tokens (mask=1): {sum(accumulator.response_mask)}")
    print(
        f"Non-trainable tokens (mask=0): {len(accumulator.all_tokens) - sum(accumulator.response_mask)}"
    )
    print(
        f"Turns completed: {sum(1 for m in accumulator.messages if m['role'] == 'assistant')}"
    )
    print(f"Response mask: {accumulator.response_mask}")
    print(
        f"\n⚠️  Episode marked as TRUNCATED - would be filtered or accepted based on config"
    )

    print(f"\n{'='*60}")
    print("FINALIZE VALIDATION (VERL pattern)")
    print(f"{'='*60}")
    if accumulator.finalize():
        print("✅ FINALIZE PASSED - BASE anchor accumulation matches ground truth!")
    else:
        print("⚠️  FINALIZE WARNING - see details above")

    return accumulator.all_tokens, accumulator.response_mask, accumulator.messages


def test_env_obs_truncation(llm, tokenizer):
    """Test case: Env observation would exceed max_seq_len"""

    print("\n" + "=" * 80)
    print("TEST CASE 3: ENV OBSERVATION TRUNCATION (adding env obs exceeds budget)")
    print("=" * 80)
    print("Setting max_seq_len=75 to force env observation truncation\n")

    messages = [
        {
            "role": "system",
            "content": "You are an expert BlackJack player. Output only 'HIT' or 'STAND'.",
        }
    ]

    max_seq_len = 75
    accumulator = TokenAccumulator(
        tokenizer=tokenizer,
        messages=messages,
        max_seq_len=max_seq_len,
        eos_token_id=tokenizer.eos_token_id,
        sanity_check_mode=SanityCheckMode.STRICT,
    )

    accumulator.add_user_message("Hand: 15, Dealer: 10", check_budget=False)

    sampling_params = SamplingParams(temperature=0.8, max_tokens=50, logprobs=1)
    max_turns = 3

    for turn in range(max_turns):
        print(f"\n{'='*60}")
        print(f"TURN {turn + 1}")
        print(f"{'='*60}")

        remaining = accumulator.get_remaining_budget()

        print(f"\n[Budget Check]")
        print(f"  Current tokens: {len(accumulator.all_tokens)}")
        print(f"  Max seq len: {max_seq_len}")
        print(f"  Remaining: {remaining}")

        if remaining <= 0:
            print(f"  ❌ Out of budget!")
            break

        prompt_text = accumulator.format_prompt()

        print(f"\n[Generation]")
        print(f"  Generating...")

        sampling_params.max_tokens = min(remaining, 50)
        outputs = llm.generate([prompt_text], sampling_params)
        output = outputs[0].outputs[0]

        response_text = output.text
        response_tokens = output.token_ids

        response_logprobs = None
        if output.logprobs is not None:
            response_logprobs = [
                lp[token_id] for lp, token_id in zip(output.logprobs, response_tokens)
            ]

        print(f"  Response: '{response_text}'")
        print(f"  Response token_ids: {len(response_tokens)} tokens")

        success = accumulator.add_assistant_response(
            response_text=response_text,
            response_token_ids=response_tokens,
            response_logprobs=response_logprobs,
        )

        print(f"  Total tokens now: {len(accumulator.all_tokens)}")

        if success:
            print(f"  ✅ Generation complete (ends with eos)")
        else:
            print(f"  ⚠️  Generation TRUNCATED")

        print(f"\n[Validation]")
        ground_truth = tokenizer.apply_chat_template(
            accumulator.messages, add_generation_prompt=False, tokenize=True
        )
        print(f"  all_tokens: {len(accumulator.all_tokens)}")
        print(f"  ground_truth: {len(ground_truth)}")

        if len(accumulator.all_tokens) == len(ground_truth):
            print(f"  ✅ PERFECT MATCH!")
        else:
            print(f"  ❌ MISMATCH")

        if not success:
            print(f"\n[Episode Truncated - Generation]")
            break

        game_done = turn >= 2
        if game_done:
            print(f"\n[Game Done]")
            break

        env_obs = f"Hand: {16 + turn}, Dealer: 10"
        print(f"\n[Env Observation]")
        print(f"  Observation: '{env_obs}'")

        success = accumulator.add_user_message(env_obs, check_budget=True)

        if not success:
            print(f"\n  ⚠️  ⚠️  ⚠️  ENV OBSERVATION TRUNCATION! ⚠️  ⚠️  ⚠️")
            print(f"  Env obs would exceed max_seq_len")
            print(f"  Episode marked as truncated")
            break
        else:
            print(f"  ✅ Env obs added successfully")

    print(f"\n{'='*60}")
    print(f"FINAL VALIDATION")
    print(f"{'='*60}")

    final_ground_truth = tokenizer.apply_chat_template(
        accumulator.messages, add_generation_prompt=False, tokenize=True
    )

    print(f"all_tokens: {len(accumulator.all_tokens)}")
    print(f"ground_truth: {len(final_ground_truth)}")

    if len(accumulator.all_tokens) == len(final_ground_truth):
        print(f"✅ ✅ ✅ PERFECT MATCH! ✅ ✅ ✅")
    else:
        print(f"❌ MISMATCH")
        print(
            f"Difference: {len(final_ground_truth) - len(accumulator.all_tokens)} tokens"
        )

    print(f"\n{'='*60}")
    print(f"DECODED CONVERSATION")
    print(f"{'='*60}")
    decoded = tokenizer.decode(accumulator.all_tokens)
    print(decoded)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Total tokens: {len(accumulator.all_tokens)}")
    print(f"Trainable tokens (mask=1): {sum(accumulator.response_mask)}")
    print(
        f"Non-trainable tokens (mask=0): {len(accumulator.all_tokens) - sum(accumulator.response_mask)}"
    )
    print(
        f"Turns completed: {sum(1 for m in accumulator.messages if m['role'] == 'assistant')}"
    )
    print(f"Response mask: {accumulator.response_mask}")
    print(
        f"\n⚠️  Episode marked as TRUNCATED - would be filtered or accepted based on config"
    )

    print(f"\n{'='*60}")
    print("FINALIZE VALIDATION (VERL pattern)")
    print(f"{'='*60}")
    if accumulator.finalize():
        print("✅ FINALIZE PASSED - BASE anchor accumulation matches ground truth!")
    else:
        print("⚠️  FINALIZE WARNING - see details above")

    return accumulator.all_tokens, accumulator.response_mask, accumulator.messages


def test_early_exit_budget(llm, tokenizer):
    """Test case: Initial prompt already exceeds max_seq_len (early exit)"""

    print("\n" + "=" * 80)
    print("TEST CASE 4: EARLY EXIT (initial prompt exceeds budget)")
    print("=" * 80)
    print("Setting max_seq_len=30 (smaller than initial prompt ~40 tokens)\n")

    messages = [
        {
            "role": "system",
            "content": "You are an expert BlackJack player. Output only 'HIT' or 'STAND'.",
        }
    ]

    max_seq_len = 30
    accumulator = TokenAccumulator(
        tokenizer=tokenizer,
        messages=messages,
        max_seq_len=max_seq_len,
        eos_token_id=tokenizer.eos_token_id,
        sanity_check_mode=SanityCheckMode.STRICT,
    )

    accumulator.add_user_message("Hand: 15, Dealer: 10", check_budget=False)

    print(f"{'='*60}")
    print(f"CHECKING INITIAL BUDGET")
    print(f"{'='*60}")

    print(f"\n[Initial State]")
    print(f"  Initial tokens: {len(accumulator.all_tokens)}")

    remaining = accumulator.get_remaining_budget()

    print(f"\n[Budget Check]")
    print(f"  Current tokens: {len(accumulator.all_tokens)}")
    print(f"  Assistant overhead: {accumulator.assistant_overhead}")
    print(f"  Max seq len: {max_seq_len}")
    print(f"  Remaining: {remaining}")

    if remaining <= 0:
        print(f"\n  ⚠️  ⚠️  ⚠️  EARLY EXIT! ⚠️  ⚠️  ⚠️")
        print(f"  Initial prompt already exceeds max_seq_len")
        print(f"  Cannot generate - breaking immediately")
        print(f"  Episode marked as truncated")
        accumulator.is_truncated = True
        accumulator.truncation_reason = "max_seq_len"

    print(f"\n{'='*60}")
    print(f"FINAL VALIDATION")
    print(f"{'='*60}")

    final_ground_truth = tokenizer.apply_chat_template(
        accumulator.messages, add_generation_prompt=False, tokenize=True
    )

    print(f"all_tokens: {len(accumulator.all_tokens)}")
    print(f"ground_truth: {len(final_ground_truth)}")

    if len(accumulator.all_tokens) == len(final_ground_truth):
        print(f"✅ ✅ ✅ PERFECT MATCH! ✅ ✅ ✅")
    else:
        print(f"❌ MISMATCH")

    print(f"\n{'='*60}")
    print(f"DECODED CONVERSATION")
    print(f"{'='*60}")
    decoded = tokenizer.decode(accumulator.all_tokens)
    print(decoded)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Total tokens: {len(accumulator.all_tokens)}")
    print(f"Trainable tokens (mask=1): {sum(accumulator.response_mask)}")
    print(
        f"Non-trainable tokens (mask=0): {len(accumulator.all_tokens) - sum(accumulator.response_mask)}"
    )
    print(
        f"Turns completed: {sum(1 for m in accumulator.messages if m['role'] == 'assistant')}"
    )
    print(f"Response mask: {accumulator.response_mask}")
    print(f"\n⚠️  Episode marked as TRUNCATED - early exit, no generation possible")

    print(f"\n{'='*60}")
    print("FINALIZE VALIDATION (VERL pattern)")
    print(f"{'='*60}")
    if accumulator.finalize():
        print("✅ FINALIZE PASSED - BASE anchor accumulation matches ground truth!")
    else:
        print("⚠️  FINALIZE WARNING - see details above")

    return accumulator.all_tokens, accumulator.response_mask, accumulator.messages


def test_long_env_obs_truncation(llm, tokenizer):
    """Test case: Env observation is very long and gets truncated mid-content"""

    print("\n" + "=" * 80)
    print("TEST CASE 5: LONG ENV OBSERVATION (truncate mid-content)")
    print("=" * 80)
    print("Using short initial prompt, tight budget to truncate env obs in turn 2\n")

    messages = [
        {
            "role": "system",
            "content": "You are an expert BlackJack player. Output only 'HIT' or 'STAND'.",
        }
    ]

    max_seq_len = 55
    accumulator = TokenAccumulator(
        tokenizer=tokenizer,
        messages=messages,
        max_seq_len=max_seq_len,
        eos_token_id=tokenizer.eos_token_id,
        sanity_check_mode=SanityCheckMode.DISABLE,
    )

    accumulator.add_user_message("Hand: 15, Dealer: 10", check_budget=False)

    sampling_params = SamplingParams(temperature=0.8, max_tokens=10, logprobs=1)
    max_turns = 2

    for turn in range(max_turns):
        print(f"\n{'='*60}")
        print(f"TURN {turn + 1}")
        print(f"{'='*60}")

        remaining = accumulator.get_remaining_budget()

        print(f"\n[Budget Check]")
        print(f"  Current tokens: {len(accumulator.all_tokens)}")
        print(f"  Max seq len: {max_seq_len}")
        print(f"  Remaining: {remaining}")

        if remaining <= 0:
            print(f"  ❌ Out of budget!")
            break

        prompt_text = accumulator.format_prompt()

        print(f"\n[Generation]")
        print(f"  Generating...")

        sampling_params.max_tokens = min(remaining, 50)
        outputs = llm.generate([prompt_text], sampling_params)
        output = outputs[0].outputs[0]

        response_text = output.text
        response_tokens = output.token_ids

        response_logprobs = None
        if output.logprobs is not None:
            response_logprobs = [
                lp[token_id] for lp, token_id in zip(output.logprobs, response_tokens)
            ]

        print(f"  Response: '{response_text}'")
        print(f"  Response token_ids: {len(response_tokens)} tokens")

        success = accumulator.add_assistant_response(
            response_text=response_text,
            response_token_ids=response_tokens,
            response_logprobs=response_logprobs,
        )

        print(f"  Total tokens now: {len(accumulator.all_tokens)}")

        if success:
            print(f"  ✅ Generation complete (ends with eos)")
        else:
            print(f"  ⚠️  Generation TRUNCATED")

        if not success:
            print(f"\n[Episode Truncated - Generation]")
            break

        if turn >= max_turns - 1:
            print(f"\n[Max Turns Reached]")
            break

        long_obs = f"Turn {turn + 2}: Your hand now has total: {17 + turn}. Dealer still showing: 10 of clubs. Dealer likely has strong hand. Risk of bust is moderate. Make your decision carefully."
        print(f"\n[Env Observation]")
        print(f"  Observation: '{long_obs[:50]}...' ({len(long_obs)} chars)")

        success = accumulator.add_user_message(long_obs, check_budget=True)

        if not success:
            print(f"\n  ⚠️  ⚠️  ⚠️  ENV OBS EXCEEDS BUDGET! ⚠️  ⚠️  ⚠️")
            print(f"  Cannot fit full observation")

            remaining_budget = max_seq_len - len(accumulator.all_tokens)
            print(f"  Remaining budget: {remaining_budget} tokens")

            if remaining_budget > 0:
                accumulator.messages.append({"role": "user", "content": long_obs})

                full_with_obs = tokenizer.apply_chat_template(
                    accumulator.messages,
                    add_generation_prompt=False,
                    tokenize=True,
                )

                obs_tokens = full_with_obs[len(accumulator.all_tokens) :]
                print(f"  Full env obs would be: {len(obs_tokens)} tokens")

                truncated_obs_tokens = obs_tokens[:remaining_budget]
                print(
                    f"  TRUNCATING from {len(obs_tokens)} to {len(truncated_obs_tokens)} tokens"
                )

                accumulator.all_tokens.extend(truncated_obs_tokens)
                accumulator.response_mask.extend([0] * len(truncated_obs_tokens))
                accumulator.logprobs.extend([0.0] * len(truncated_obs_tokens))

                truncated_text = tokenizer.decode(truncated_obs_tokens)
                print(f"  Truncated text: '{truncated_text[:50]}...'")

                print(
                    f"  ⚠️  Lost {len(obs_tokens) - len(truncated_obs_tokens)} tokens!"
                )
            else:
                print(f"  No budget left - cannot add any tokens")

            accumulator.is_truncated = True
            accumulator.truncation_reason = "env_observation_length"

            print(f"\n  Cannot generate - no budget left")
            print(f"  Episode marked as truncated")
            break
        else:
            print(
                f"  ✅ Env obs added successfully (should not happen with tight budget!)"
            )
            break

    print(f"\n{'='*60}")
    print(f"FINAL STATE")
    print(f"{'='*60}")

    print(f"\nall_tokens: {len(accumulator.all_tokens)}")

    print(f"\n{'='*60}")
    print(f"DECODED CONVERSATION (showing truncation)")
    print(f"{'='*60}")
    decoded = tokenizer.decode(accumulator.all_tokens)
    print(decoded)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Total tokens: {len(accumulator.all_tokens)}")
    print(f"Trainable tokens (mask=1): {sum(accumulator.response_mask)}")
    print(
        f"Non-trainable tokens (mask=0): {len(accumulator.all_tokens) - sum(accumulator.response_mask)}"
    )
    print(
        f"Turns completed: {sum(1 for m in accumulator.messages if m['role'] == 'assistant')}"
    )
    print(f"First 20 of response_mask: {accumulator.response_mask[:20]}")
    print(f"Last 20 of response_mask: {accumulator.response_mask[-20:]}")
    print(f"\n⚠️  Episode shows what happens when content is truncated mid-observation")

    print(f"\n{'='*60}")
    print("FINALIZE VALIDATION (VERL pattern)")
    print(f"{'='*60}")
    print("⚠️  Validation disabled for this test (mid-content truncation)")
    if accumulator.finalize():
        print("✅ FINALIZE PASSED (skipped)")
    else:
        print("⚠️  FINALIZE WARNING - see details above")

    return accumulator.all_tokens, accumulator.response_mask, accumulator.messages


def test_chat_template_overhead(llm, tokenizer):
    """Test case: Check if chat template overhead causes budget overruns"""

    print("\n" + "=" * 80)
    print("TEST CASE 6: CHAT TEMPLATE OVERHEAD (verify budget accounting)")
    print("=" * 80)
    print("Test that remaining_budget accounts for role header tokens\n")

    messages = [
        {
            "role": "system",
            "content": "You are an expert BlackJack player. Output only 'HIT' or 'STAND'.",
        }
    ]

    max_seq_len = 200
    accumulator = TokenAccumulator(
        tokenizer=tokenizer,
        messages=messages,
        max_seq_len=max_seq_len,
        eos_token_id=tokenizer.eos_token_id,
        sanity_check_mode=SanityCheckMode.STRICT,
    )

    accumulator.add_user_message("Hand: 15, Dealer: 10", check_budget=False)

    sampling_params = SamplingParams(temperature=0.8, max_tokens=50, logprobs=1)
    max_turns = 5

    for turn in range(max_turns):
        print(f"\n{'='*60}")
        print(f"TURN {turn + 1}")
        print(f"{'='*60}")

        remaining = accumulator.get_remaining_budget()

        print(f"\n[Budget Check]")
        print(f"  Current tokens: {len(accumulator.all_tokens)}")
        print(f"  Assistant overhead: {accumulator.assistant_overhead}")
        print(f"  Max seq len: {max_seq_len}")
        print(f"  Remaining budget: {remaining}")
        print(f"  → Will pass max_tokens={remaining} to vLLM")

        if remaining <= 0:
            print(f"  ❌ Out of budget!")
            accumulator.is_truncated = True
            accumulator.truncation_reason = "max_seq_len"
            break

        prompt_text = accumulator.format_prompt()

        print(f"\n[Generation]")
        print(f"  Generating with max_tokens={remaining}...")

        sampling_params.max_tokens = remaining
        outputs = llm.generate([prompt_text], sampling_params)
        output = outputs[0].outputs[0]

        response_text = output.text
        response_tokens = output.token_ids

        response_logprobs = None
        if output.logprobs is not None:
            response_logprobs = [
                lp[token_id] for lp, token_id in zip(output.logprobs, response_tokens)
            ]

        print(f"  vLLM generated: {len(response_tokens)} content tokens")
        print(f"  Response text: '{response_text[:50]}...'")

        # Now check what happens when we add it
        tokens_before = len(accumulator.all_tokens)

        success = accumulator.add_assistant_response(
            response_text=response_text,
            response_token_ids=response_tokens,
            response_logprobs=response_logprobs,
        )

        tokens_after = len(accumulator.all_tokens)
        tokens_added = tokens_after - tokens_before

        print(f"\n[After Adding Response]")
        print(f"  vLLM content tokens: {len(response_tokens)}")
        print(f"  Total tokens added (with headers): {tokens_added}")
        print(f"  Role header overhead: {tokens_added - len(response_tokens)}")
        print(f"  Total tokens now: {tokens_after}")
        print(f"  Max allowed: {max_seq_len}")

        if tokens_after > max_seq_len:
            print(f"  ❌❌❌ BUDGET EXCEEDED! ❌❌❌")
            print(f"  Overrun by: {tokens_after - max_seq_len} tokens")
            print(f"\n  ROOT CAUSE: remaining_budget doesn't account for role headers!")
            print(f"  We passed max_tokens={remaining} to vLLM")
            print(f"  vLLM generated {len(response_tokens)} tokens")
            print(
                f"  But chat template added {tokens_added - len(response_tokens)} header tokens"
            )
            print(
                f"  Result: {tokens_before} + {tokens_added} = {tokens_after} > {max_seq_len}"
            )
            return False
        else:
            print(f"  ✅ Within budget ({tokens_after} <= {max_seq_len})")

        if not success:
            print(f"\n[Episode Truncated - Generation]")
            break

        game_done = turn >= max_turns - 1
        if game_done:
            print(f"\n[Max Turns Reached]")
            break

        env_obs = f"Hand: {16 + turn}, Dealer: 10"
        print(f"\n[Env Observation]")
        print(f"  Observation: '{env_obs}'")

        success = accumulator.add_user_message(env_obs, check_budget=True)

        if not success:
            print(f"  ⚠️  Env obs would exceed budget - breaking")
            break
        else:
            print(f"  ✅ Env obs added successfully")

    print(f"\n{'='*60}")
    print(f"FINAL CHECK")
    print(f"{'='*60}")

    print(f"Final token count: {len(accumulator.all_tokens)}")
    print(f"Max seq len: {max_seq_len}")

    if len(accumulator.all_tokens) <= max_seq_len:
        print(f"✅ ✅ ✅ BUDGET RESPECTED! ✅ ✅ ✅")
        print(f"The budget calculation correctly accounts for chat template overhead")
    else:
        print(f"❌ BUDGET VIOLATED!")
        print(f"Exceeded by: {len(accumulator.all_tokens) - max_seq_len} tokens")

    print(f"\n{'='*60}")
    print(f"DECODED CONVERSATION")
    print(f"{'='*60}")
    decoded = tokenizer.decode(accumulator.all_tokens)
    print(decoded)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Total tokens: {len(accumulator.all_tokens)}")
    print(f"Trainable tokens (mask=1): {sum(accumulator.response_mask)}")
    print(
        f"Non-trainable tokens (mask=0): {len(accumulator.all_tokens) - sum(accumulator.response_mask)}"
    )
    print(
        f"Turns completed: {sum(1 for m in accumulator.messages if m['role'] == 'assistant')}"
    )

    if len(accumulator.all_tokens) <= max_seq_len:
        return True
    else:
        return False


def test_prefix_vs_direct(llm, tokenizer):
    """Compare prefix matching (current) vs direct extraction (other libraries)."""

    print("\n" + "=" * 80)
    print("TEST CASE 7: PREFIX MATCHING vs DIRECT EXTRACTION")
    print("=" * 80)
    print("Comparing our approach vs industry standard (TRL, VERL, etc.)\n")

    messages = [
        {
            "role": "system",
            "content": "You are an expert BlackJack player. Output only 'HIT' or 'STAND'.",
        },
        {"role": "user", "content": "Hand: 15, Dealer: 10"},
    ]

    prompt = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )

    sampling_params = SamplingParams(temperature=0.0, max_tokens=5, logprobs=1)
    outputs = llm.generate([prompt], sampling_params)
    output = outputs[0].outputs[0]

    print("=" * 80)
    print("APPROACH 1: PREFIX MATCHING (OUR CURRENT IMPLEMENTATION)")
    print("=" * 80)

    # Simulate what TokenAccumulator.add_assistant_response() does
    BASE_CHAT_HISTORY = [
        {
            "role": "system",
            "content": "You are an expert BlackJack player. Output only 'HIT' or 'STAND'.",
        },
        {"role": "user", "content": ""},
    ]
    base_tokens_wo_gen = tokenizer.apply_chat_template(
        BASE_CHAT_HISTORY,
        add_generation_prompt=False,
        tokenize=True,
    )
    base_len_wo_gen = len(base_tokens_wo_gen)

    # Re-tokenize the full assistant message
    temp_messages = [
        *BASE_CHAT_HISTORY,
        {"role": "assistant", "content": output.text},
    ]
    full_with_assistant = tokenizer.apply_chat_template(
        temp_messages,
        add_generation_prompt=False,
        tokenize=True,
    )
    assistant_tokens_prefix = full_with_assistant[base_len_wo_gen:]

    print(f"  1. Get vLLM output.token_ids: {output.token_ids}")
    print(f"     Decoded: '{tokenizer.decode(output.token_ids)}'")
    print(f"  2. ❌ IGNORE those token_ids!")
    print(f"  3. Re-tokenize assistant message via chat template")
    print(f"  4. Extract via prefix matching: {assistant_tokens_prefix}")
    print(f"     Length: {len(assistant_tokens_prefix)} tokens")
    print(f"     Decoded: '{tokenizer.decode(assistant_tokens_prefix)}'")
    print(f"\n  ⚠️  PROBLEM: We called tokenizer.apply_chat_template() unnecessarily!")

    print("\n" + "=" * 80)
    print("APPROACH 2: DIRECT EXTRACTION (TRL, VERL, PRIME-RL, etc.)")
    print("=" * 80)

    # Get role header tokens (pre-compute once at init)
    base_empty = [
        {"role": "system", "content": ""},
        {"role": "user", "content": ""},
    ]
    base_empty_tokens = tokenizer.apply_chat_template(
        base_empty,
        add_generation_prompt=False,
        tokenize=True,
    )

    with_empty_assistant = base_empty + [{"role": "assistant", "content": ""}]
    with_assistant_tokens = tokenizer.apply_chat_template(
        with_empty_assistant,
        add_generation_prompt=False,
        tokenize=True,
    )

    role_header_tokens = with_assistant_tokens[len(base_empty_tokens) :]

    # Combine: role_header + content_tokens (from vLLM)
    assistant_tokens_direct = role_header_tokens + output.token_ids

    print(f"  1. Get vLLM output.token_ids: {output.token_ids}")
    print(f"     Decoded: '{tokenizer.decode(output.token_ids)}'")
    print(f"  2. ✅ USE those token_ids directly!")
    print(f"  3. Get pre-computed role header: {role_header_tokens}")
    print(f"     Decoded: '{tokenizer.decode(role_header_tokens)}'")
    print(f"  4. Combine: role_header + content_tokens")
    print(f"     Result: {assistant_tokens_direct}")
    print(f"     Length: {len(assistant_tokens_direct)} tokens")
    print(f"     Decoded: '{tokenizer.decode(assistant_tokens_direct)}'")
    print(f"\n  ✅ BENEFIT: Only 1 tokenization call (at init), not every turn!")

    print("\n" + "=" * 80)
    print("COMPARISON")
    print("=" * 80)

    if assistant_tokens_prefix == assistant_tokens_direct:
        print(f"  ✅ Both approaches give SAME result")
        print(f"  ✅ Length: {len(assistant_tokens_prefix)} tokens")
    else:
        print(f"  ❌ MISMATCH!")
        print(f"     Prefix: {assistant_tokens_prefix}")
        print(f"     Direct: {assistant_tokens_direct}")

    print(f"\n  Tokenization calls:")
    print(f"    Prefix matching: O(N) - one call per turn")
    print(f"    Direct extraction: O(1) - pre-computed at init")

    print("\n" + "=" * 80)
    print("BUDGET CALCULATION FIX")
    print("=" * 80)

    # Current (wrong)
    test_msgs = [{"role": "user", "content": "x"}]
    without_gen = tokenizer.apply_chat_template(
        test_msgs, add_generation_prompt=False, tokenize=True
    )
    with_gen = tokenizer.apply_chat_template(
        test_msgs, add_generation_prompt=True, tokenize=True
    )
    gen_prompt_len = len(with_gen) - len(without_gen)

    # Correct
    assistant_overhead = len(role_header_tokens)

    print(f"  ❌ Current: gen_prompt_len = {gen_prompt_len}")
    print(f"     (Only counts prompt-side '<|im_start|>assistant\\n')")
    print(f"\n  ✅ Correct: assistant_overhead = {assistant_overhead}")
    print(f"     (Counts full role header + EOS)")
    print(f"\n  Difference: {assistant_overhead - gen_prompt_len} tokens")
    print(f"  This is why we exceed max_seq_len!")

    print("\n" + "=" * 80)
    print("FULL CONVERSATION EXAMPLE")
    print("=" * 80)

    # Show a full multi-turn example
    example_messages = [
        {
            "role": "system",
            "content": "You are an expert BlackJack player. Output only 'HIT' or 'STAND'.",
        },
        {"role": "user", "content": "Hand: 15, Dealer: 10"},
        {"role": "assistant", "content": output.text},
        {"role": "user", "content": "Hand: 16, Dealer: 10"},
        {"role": "assistant", "content": output.text},
    ]

    full_conversation_tokens = tokenizer.apply_chat_template(
        example_messages,
        add_generation_prompt=False,
        tokenize=True,
    )

    full_decoded = tokenizer.decode(full_conversation_tokens)

    print(f"Message sequence: system -> user -> assistant -> user -> assistant")
    print(f"Total tokens: {len(full_conversation_tokens)}")
    print(f"\nDecoded:\n{full_decoded}")

    print("\n" + "=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)
    print("  1. Use direct extraction (like all 6 libraries we studied)")
    print(
        "  2. Fix budget calculation: use assistant_overhead instead of gen_prompt_len"
    )
    print("  3. Performance: 3x fewer tokenization calls")

    return True


def main():
    print("Loading model and tokenizer...")
    model_name = "Qwen/Qwen3-1.7B"

    llm = LLM(
        model=model_name,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.3,
        max_model_len=4096,
        enable_prefix_caching=True,
    )

    tokenizer = get_tokenizer(model_name)

    print("✅ Model loaded!\n")

    print("\n" + "#" * 80)
    print("# RUNNING ALL 7 TEST CASES (V2 - SIMPLIFIED)")
    print("#" * 80)

    test_normal_rollout(
        llm=llm,
        tokenizer=tokenizer,
        max_seq_len=2048,
        max_turns=3,
    )

    test_vllm_truncation(
        llm=llm,
        tokenizer=tokenizer,
    )

    test_env_obs_truncation(
        llm=llm,
        tokenizer=tokenizer,
    )

    test_early_exit_budget(
        llm=llm,
        tokenizer=tokenizer,
    )

    test_long_env_obs_truncation(
        llm=llm,
        tokenizer=tokenizer,
    )

    # NEW: Test chat template overhead
    budget_ok = test_chat_template_overhead(
        llm=llm,
        tokenizer=tokenizer,
    )

    # NEW: Compare prefix vs direct
    test_prefix_vs_direct(
        llm=llm,
        tokenizer=tokenizer,
    )

    print("\n" + "#" * 80)
    print("# ALL 7 TESTS COMPLETED")
    print("#" * 80)

    if not budget_ok:
        print("\n⚠️  CRITICAL: Chat template overhead causes budget violations!")
        print("This explains why episodes exceed max_seq_len in production")
    else:
        print("\n✅ All budget checks passed")


if __name__ == "__main__":
    main()
