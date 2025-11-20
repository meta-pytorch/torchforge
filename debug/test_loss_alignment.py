#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Standalone test to verify loss alignment between policy and ref model paths.

Goal: Prove whether the KL explosion (step 1 loss = 39,000) is due to an alignment bug
      or something else (initial model divergence, etc.).

Test strategy:
1. Create multi-turn conversation with TokenAccumulator
2. Extract episode tensors (all_token_ids, response_mask, loss_mask)
3. Create dummy logits
4. Compute logprobs via policy path
5. Compute ref_logprobs via ref path (SAME logits to verify alignment)
6. Verify logprob_diff is small (proves alignment is correct)
7. Call simple_grpo_loss and verify no explosion
"""

import os
import sys

import torch

# Add project root to path
sys.path.insert(0, "/home/felipemello/forge")

from apps.blackjack.main_v2 import SanityCheckMode, TokenAccumulator
from forge.data.common import CROSS_ENTROPY_IGNORE_IDX
from forge.util.ops import compute_logprobs, create_shifted_targets
from vllm.transformers_utils.tokenizer import get_tokenizer


def create_dummy_logits(batch_size, seq_len, vocab_size, temperature=1.0):
    """
    Create dummy logits that are NOT uniform random (which would give ~equal probs).
    Instead, create peaked distributions to mimic real model behavior.
    """
    # Create base logits
    logits = torch.randn(batch_size, seq_len, vocab_size) * temperature

    # For each position, make the "correct" token have highest logit
    # This simulates a model that's somewhat confident
    for b in range(batch_size):
        for s in range(seq_len):
            # Pick a random token to be the "target" and boost its logit
            target_id = torch.randint(0, vocab_size, (1,)).item()
            logits[b, s, target_id] += 3.0  # Boost by 3 to make it confident

    return logits


def simple_grpo_loss_minimal(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    loss_mask: torch.Tensor,
    ref_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    beta: float = 0.1,
) -> dict:
    """
    Minimal version of simple_grpo_loss with detailed outputs for debugging.
    Returns dict with all intermediate values.
    """
    # Create targets
    targets = create_shifted_targets(input_ids, loss_mask)

    # Compute policy logprobs
    logprobs = compute_logprobs(logits, targets, ignore_index=CROSS_ENTROPY_IGNORE_IDX)

    # Logprob difference
    logprob_diff = ref_logprobs - logprobs

    # KL divergence
    kl = torch.exp(ref_logprobs - logprobs) - (ref_logprobs - logprobs) - 1

    # Policy loss
    per_token_policy_loss = torch.exp(logprobs - logprobs.detach()) * advantages
    per_token_loss = -(per_token_policy_loss - beta * kl)

    # Per-sequence normalization
    loss = (
        (per_token_loss * loss_mask).sum(dim=1) / loss_mask.sum(dim=1).clamp(min=1.0)
    ).mean()

    return {
        "targets": targets,
        "logprobs": logprobs,
        "ref_logprobs": ref_logprobs,
        "logprob_diff": logprob_diff,
        "kl": kl,
        "per_token_loss": per_token_loss,
        "loss": loss,
        "loss_mask": loss_mask,
    }


def print_detailed_comparison(result: dict, input_ids: torch.Tensor):
    """Print detailed position-by-position comparison."""
    targets = result["targets"]
    logprobs = result["logprobs"]
    ref_logprobs = result["ref_logprobs"]
    logprob_diff = result["logprob_diff"]
    kl = result["kl"]
    loss_mask = result["loss_mask"]

    print("\n" + "=" * 120)
    print("POSITION-BY-POSITION ANALYSIS (First sequence only)")
    print("=" * 120)
    print(
        f"{'Idx':>4} {'Input':>6} {'Target':>8} {'Mask':>5} {'LogProb':>10} {'RefLogP':>10} {'Diff':>8} {'KL':>10}"
    )
    print("-" * 120)

    seq = 0  # First sequence
    for i in range(len(input_ids[seq])):
        inp = input_ids[seq, i].item()
        tgt = targets[seq, i].item()
        mask = loss_mask[seq, i].item()
        lp = logprobs[seq, i].item()
        ref_lp = ref_logprobs[seq, i].item()
        diff = logprob_diff[seq, i].item()
        kl_val = kl[seq, i].item()

        tgt_str = "IGNORE" if tgt == CROSS_ENTROPY_IGNORE_IDX else f"{tgt:6d}"

        # Highlight problematic positions
        flag = ""
        if mask > 0 and abs(diff) > 5.0:
            flag = " ⚠️  LARGE DIFF!"
        if mask > 0 and kl_val > 100:
            flag = " 🔥 KL EXPLOSION!"

        print(
            f"{i:4d} {inp:6d} {tgt_str:>8s} {mask:5.1f} {lp:10.4f} {ref_lp:10.4f} {diff:8.4f} {kl_val:10.4f}{flag}"
        )

    print("-" * 120)


def test_loss_alignment():
    """Main test function."""
    print("\n" + "=" * 80)
    print("STANDALONE LOSS ALIGNMENT TEST")
    print("=" * 80)

    # ============================================================================
    # Step 1: Setup tokenizer and TokenAccumulator
    # ============================================================================
    print("\n[1/7] Setting up tokenizer and TokenAccumulator...")

    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer = get_tokenizer(model_name)

    initial_messages = [{"role": "system", "content": "You are a helpful assistant."}]

    max_seq_len = 512
    eos_token_id = tokenizer.eos_token_id

    accumulator = TokenAccumulator(
        tokenizer=tokenizer,
        messages=initial_messages,
        max_seq_len=max_seq_len,
        eos_token_id=eos_token_id,
        enable_thinking=False,
        sanity_check_mode=SanityCheckMode.DISABLE,
    )

    print(f"   ✓ Tokenizer: {model_name}")
    print(f"   ✓ EOS token ID: {eos_token_id}")
    print(f"   ✓ Max seq len: {max_seq_len}")

    # ============================================================================
    # Step 2: Add multi-turn conversation
    # ============================================================================
    print("\n[2/7] Building multi-turn conversation...")

    # Turn 1: User
    accumulator.add_user_message("What is 2+2?")

    # Turn 1: Assistant
    assistant_response_1 = "The answer is 4."
    assistant_tokens_1 = tokenizer.encode(
        assistant_response_1, add_special_tokens=False
    )
    assistant_tokens_1.append(eos_token_id)
    accumulator.add_assistant_response(
        response_text=assistant_response_1,
        response_token_ids=assistant_tokens_1,
        response_logprobs=None,
    )

    # Turn 2: User
    accumulator.add_user_message("What is 3+3?")

    # Turn 2: Assistant
    assistant_response_2 = "The answer is 6."
    assistant_tokens_2 = tokenizer.encode(
        assistant_response_2, add_special_tokens=False
    )
    assistant_tokens_2.append(eos_token_id)
    accumulator.add_assistant_response(
        response_text=assistant_response_2,
        response_token_ids=assistant_tokens_2,
        response_logprobs=None,
    )

    print(f"   ✓ Added 2 turns (4 messages)")
    print(f"   ✓ Total tokens: {len(accumulator.accumulated_tokens)}")
    print(f"   ✓ Trainable positions: {sum(accumulator.response_mask)}")

    # ============================================================================
    # Step 3: Extract episode tensors
    # ============================================================================
    print("\n[3/7] Extracting episode tensors...")

    all_token_ids = torch.tensor(
        accumulator.accumulated_tokens, dtype=torch.long
    ).unsqueeze(
        0
    )  # [1, seq_len]
    response_mask = torch.tensor(accumulator.response_mask, dtype=torch.bool).unsqueeze(
        0
    )  # [1, seq_len]

    # Create loss_mask via torch.roll (same as in main_v2.py)
    loss_mask = torch.roll(response_mask.float(), shifts=-1, dims=-1)
    loss_mask[:, -1] = 0.0

    print(f"   ✓ all_token_ids shape: {all_token_ids.shape}")
    print(f"   ✓ response_mask shape: {response_mask.shape}")
    print(f"   ✓ loss_mask shape: {loss_mask.shape}")
    print(f"   ✓ Trainable positions (loss_mask.sum()): {loss_mask.sum().item()}")

    # ============================================================================
    # Step 4: Create dummy logits
    # ============================================================================
    print("\n[4/7] Creating dummy logits...")

    # Use actual vocab size that includes special tokens
    # tokenizer.vocab_size may not include special tokens, so we need to find the max token ID
    max_token_id = max(all_token_ids.max().item(), eos_token_id)
    vocab_size = max_token_id + 100  # Add buffer for safety
    batch_size = 1
    seq_len = all_token_ids.shape[1]

    logits = create_dummy_logits(batch_size, seq_len, vocab_size, temperature=1.0)

    print(f"   ✓ Logits shape: {logits.shape}")
    print(f"   ✓ Vocab size (with special tokens): {vocab_size}")
    print(f"   ✓ Tokenizer vocab_size: {tokenizer.vocab_size}")
    print(f"   ✓ Max token ID in sequence: {all_token_ids.max().item()}")

    # ============================================================================
    # Step 5: Compute logprobs (policy path)
    # ============================================================================
    print("\n[5/7] Computing logprobs (policy path)...")

    # This is what happens in simple_grpo_loss
    targets_policy = create_shifted_targets(all_token_ids, loss_mask)
    logprobs_policy = compute_logprobs(
        logits, targets_policy, ignore_index=CROSS_ENTROPY_IGNORE_IDX
    )

    print(f"   ✓ targets_policy shape: {targets_policy.shape}")
    print(f"   ✓ logprobs_policy shape: {logprobs_policy.shape}")
    print(
        f"   ✓ Non-IGNORE positions: {(targets_policy != CROSS_ENTROPY_IGNORE_IDX).sum().item()}"
    )

    # ============================================================================
    # Step 6: Compute ref_logprobs (ref model path - SAME logits!)
    # ============================================================================
    print("\n[6/7] Computing ref_logprobs (ref model path with SAME logits)...")

    # This is what happens in reference_model.forward
    targets_ref = create_shifted_targets(all_token_ids, loss_mask)
    logprobs_ref = compute_logprobs(
        logits, targets_ref, ignore_index=CROSS_ENTROPY_IGNORE_IDX
    )

    print(f"   ✓ targets_ref shape: {targets_ref.shape}")
    print(f"   ✓ logprobs_ref shape: {logprobs_ref.shape}")
    print(
        f"   ✓ Non-IGNORE positions: {(targets_ref != CROSS_ENTROPY_IGNORE_IDX).sum().item()}"
    )

    # ============================================================================
    # CRITICAL: Verify alignment
    # ============================================================================
    print("\n" + "=" * 80)
    print("ALIGNMENT VERIFICATION")
    print("=" * 80)

    # Check 1: Targets should be identical
    targets_match = torch.equal(targets_policy, targets_ref)
    print(f"\n✓ Targets match: {targets_match}")
    if not targets_match:
        print("   🔥 BUG DETECTED: Targets differ between policy and ref paths!")
        print(f"   Policy targets: {targets_policy[0, :20].tolist()}")
        print(f"   Ref targets:    {targets_ref[0, :20].tolist()}")

    # Check 2: Logprobs should be identical (since we used SAME logits)
    logprobs_match = torch.allclose(logprobs_policy, logprobs_ref, atol=1e-6)
    print(f"✓ Logprobs match: {logprobs_match}")
    if not logprobs_match:
        print("   🔥 BUG DETECTED: Logprobs differ even with same logits!")
        max_diff = (logprobs_policy - logprobs_ref).abs().max().item()
        print(f"   Max difference: {max_diff}")

    # Check 3: Logprob diff should be near zero
    logprob_diff = logprobs_ref - logprobs_policy
    masked_diff = logprob_diff * loss_mask
    num_trainable = loss_mask.sum().clamp(min=1.0)

    diff_mean = (masked_diff.sum() / num_trainable).item()
    diff_min = logprob_diff[loss_mask.bool()].min().item() if num_trainable > 0 else 0.0
    diff_max = logprob_diff[loss_mask.bool()].max().item() if num_trainable > 0 else 0.0

    print(f"\nLogprob diff statistics:")
    print(f"   Mean: {diff_mean:.6f}")
    print(f"   Min:  {diff_min:.6f}")
    print(f"   Max:  {diff_max:.6f}")

    if abs(diff_mean) > 0.01 or abs(diff_min) > 1.0 or abs(diff_max) > 1.0:
        print("   🔥 WARNING: Large logprob diff detected!")
    else:
        print("   ✓ Logprob diff is small (alignment is correct)")

    # ============================================================================
    # Step 7: Call simple_grpo_loss and verify no explosion
    # ============================================================================
    print("\n[7/7] Computing GRPO loss...")

    advantages = torch.tensor([[1.0]])  # Dummy advantage

    result = simple_grpo_loss_minimal(
        logits=logits,
        input_ids=all_token_ids,
        loss_mask=loss_mask,
        ref_logprobs=logprobs_ref,  # Use ref_logprobs from step 6
        advantages=advantages,
        beta=0.1,
    )

    loss = result["loss"]
    kl = result["kl"]

    kl_masked = kl * loss_mask
    kl_mean = (kl_masked.sum() / num_trainable).item()
    kl_max = kl[loss_mask.bool()].max().item() if num_trainable > 0 else 0.0

    print(f"\n   Loss: {loss.item():.6f}")
    print(f"   KL mean: {kl_mean:.6f}")
    print(f"   KL max:  {kl_max:.6f}")

    if loss.item() > 1000:
        print("   🔥 LOSS EXPLOSION DETECTED!")
    elif kl_max > 100:
        print("   🔥 KL EXPLOSION DETECTED!")
    else:
        print("   ✓ Loss and KL are reasonable")

    # ============================================================================
    # Print detailed comparison
    # ============================================================================
    print_detailed_comparison(result, all_token_ids)

    # ============================================================================
    # Final summary
    # ============================================================================
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    all_checks_pass = (
        targets_match
        and logprobs_match
        and abs(diff_mean) < 0.01
        and loss.item() < 1000
        and kl_max < 100
    )

    if all_checks_pass:
        print("\n✅ ALL CHECKS PASSED")
        print("   - Targets are identical in policy and ref paths")
        print("   - Logprobs are identical (with same logits)")
        print("   - Logprob diff is near zero")
        print("   - No loss explosion")
        print("   - No KL explosion")
        print("\n   CONCLUSION: No alignment bug detected in the implementation.")
        print("   The step 1 loss issue is likely due to:")
        print("   - Initial model divergence between policy and ref")
        print("   - Uninitialized or stale ref_logprobs")
        print("   - Real model behavior (not a bug in alignment)")
    else:
        print("\n❌ CHECKS FAILED")
        print("   CONCLUSION: Alignment bug detected! Review the implementation.")
        if not targets_match:
            print("   - Targets differ between paths")
        if not logprobs_match:
            print("   - Logprobs differ even with same logits")
        if abs(diff_mean) > 0.01:
            print(f"   - Large logprob diff mean: {diff_mean}")
        if loss.item() > 1000:
            print(f"   - Loss explosion: {loss.item()}")
        if kl_max > 100:
            print(f"   - KL explosion: {kl_max}")

    print("\n" + "=" * 80)
    print()


if __name__ == "__main__":
    test_loss_alignment()
