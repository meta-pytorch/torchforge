#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
V6 Loss Mask Diagnostic - Directly test loss_mask creation with suffix tokens.

This script creates a simple episode with V6 TokenAccumulator and verifies:
1. Suffix tokens are properly handled in response_mask
2. loss_mask correctly shifts response_mask via torch.roll
3. Suffix positions have loss_mask=0.0 and targets=IGNORE
4. No suffix tokens leak into training

This addresses the KL explosion hypothesis from v6_loss_debugging_summary.md.
"""

import sys

sys.path.insert(0, "/home/felipemello/forge")

import torch
from debug.token_accumulator_fn_v6 import TokenAccumulator, ValidationMode
from forge.data.common import CROSS_ENTROPY_IGNORE_IDX
from forge.util.ops import create_shifted_targets
from vllm.transformers_utils.tokenizer import get_tokenizer


def test_loss_mask_with_suffix():
    """Test loss_mask creation with V6 suffix tokens."""
    print("\n" + "=" * 80)
    print("V6 LOSS MASK DIAGNOSTIC - Suffix Token Handling")
    print("=" * 80)

    # Setup
    tokenizer = get_tokenizer("Qwen/Qwen2.5-0.5B-Instruct")

    accumulator = TokenAccumulator(
        tokenizer=tokenizer,
        messages=[{"role": "system", "content": "Help"}],
        max_len=512,
        eos_id=tokenizer.eos_token_id,
        thinking=False,
        validation=ValidationMode.OFF,
    )

    print(f"\n✓ Setup complete")
    print(f"  Suffix tokens: {accumulator.suffix}")
    print(f"  Suffix decoded: {tokenizer.decode(accumulator.suffix)!r}")

    # Add single turn
    accumulator.add_user("Hi")
    response_text = "Hello!"
    response_tokens = tokenizer.encode(response_text, add_special_tokens=False)
    response_tokens.append(tokenizer.eos_token_id)

    accumulator.add_assistant(response_text, response_tokens)

    # Get episode data
    episode_data = accumulator.get_data()

    print(f"\n✓ Episode created")
    print(f"  Total tokens: {len(episode_data.token_ids)}")
    print(
        f"  Trainable (response_mask=True): {episode_data.response_mask.sum().item()}"
    )

    # Create loss_mask using torch.roll (same as main_v2.py line 1050)
    loss_mask_tensor = torch.roll(episode_data.response_mask, shifts=-1, dims=0).float()
    loss_mask_tensor[-1] = 0.0

    print(f"\n✓ loss_mask created via torch.roll")
    print(f"  Trainable (loss_mask=1.0): {loss_mask_tensor.sum().item()}")

    # Create targets
    targets = create_shifted_targets(
        episode_data.token_ids.unsqueeze(0), loss_mask_tensor.unsqueeze(0)
    ).squeeze(0)

    # Find suffix positions (trainable followed by non-trainable)
    suffix_positions = []
    for i in range(len(episode_data.token_ids) - 1):
        # EOS token: response_mask[i] = True (trainable)
        # Suffix token: response_mask[i+1] = False (not trainable)
        if episode_data.response_mask[i] and not episode_data.response_mask[i + 1]:
            suffix_positions.append(i + 1)

    print(f"\n✓ Suffix positions detected: {suffix_positions}")

    # Detailed token-by-token analysis
    print("\n" + "=" * 80)
    print("TOKEN-BY-TOKEN ANALYSIS")
    print("=" * 80)
    print(
        f"{'Idx':>4} {'Token':>10} {'Decoded':>15} {'Resp':>5} {'Loss':>5} {'Target':>10} {'Status':>20}"
    )
    print("-" * 80)

    for i in range(len(episode_data.token_ids)):
        tok_id = episode_data.token_ids[i].item()
        tok_str = tokenizer.decode([tok_id])[:12]  # Truncate for display
        resp_mask = episode_data.response_mask[i].item()
        loss_mask = loss_mask_tensor[i].item()
        target = targets[i].item()

        resp_str = "✓" if resp_mask else "·"
        loss_str = f"{loss_mask:.1f}"
        target_str = "IGNORE" if target == CROSS_ENTROPY_IGNORE_IDX else f"{target:6d}"

        # Determine status
        if i in suffix_positions:
            status = "SUFFIX"
            if loss_mask != 0.0:
                status += " 🔥 LEAK!"
            if target != CROSS_ENTROPY_IGNORE_IDX:
                status += " 🔥 TARGET!"
        elif resp_mask and loss_mask == 1.0:
            status = "trainable ✓"
        elif not resp_mask and loss_mask == 0.0:
            status = "not trainable"
        else:
            status = "🔥 MISMATCH!"

        # Highlight EOS tokens
        if tok_id == tokenizer.eos_token_id:
            tok_str = f"<EOS> ({tok_id})"

        print(
            f"{i:4d} {tok_id:10d} {tok_str:>15s} {resp_str:>5s} {loss_str:>5s} {target_str:>10s} {status:>20s}"
        )

    # Verification checks
    print("\n" + "=" * 80)
    print("VERIFICATION CHECKS")
    print("=" * 80)

    all_pass = True

    # Check 1: Suffix positions should have response_mask=False
    print("\n[Check 1] Suffix tokens have response_mask=False")
    for pos in suffix_positions:
        resp = episode_data.response_mask[pos].item()
        if resp:
            print(f"  🔥 FAIL: Position {pos} has response_mask=True (expected False)")
            all_pass = False
        else:
            print(f"  ✓ Position {pos}: response_mask=False")

    # Check 2: Suffix positions should have loss_mask=0.0
    print("\n[Check 2] Suffix tokens have loss_mask=0.0")
    for pos in suffix_positions:
        loss = loss_mask_tensor[pos].item()
        if loss != 0.0:
            print(f"  🔥 FAIL: Position {pos} has loss_mask={loss} (expected 0.0)")
            all_pass = False
        else:
            print(f"  ✓ Position {pos}: loss_mask=0.0")

    # Check 3: Suffix positions should have targets=IGNORE
    print("\n[Check 3] Suffix tokens have targets=IGNORE")
    for pos in suffix_positions:
        tgt = targets[pos].item()
        if tgt != CROSS_ENTROPY_IGNORE_IDX:
            print(
                f"  🔥 FAIL: Position {pos} has target={tgt} (expected {CROSS_ENTROPY_IGNORE_IDX})"
            )
            all_pass = False
        else:
            print(f"  ✓ Position {pos}: target=IGNORE")

    # Check 4: EOS tokens should be trainable
    print("\n[Check 4] EOS tokens are trainable")
    eos_positions = [
        i
        for i, tok in enumerate(episode_data.token_ids)
        if tok == tokenizer.eos_token_id
    ]
    for pos in eos_positions:
        resp = episode_data.response_mask[pos].item()
        # EOS should be trainable only if it's an assistant EOS (not system/user EOS)
        # For this test, we only have one assistant response, so check if it's trainable
        if pos in suffix_positions:
            # This EOS is followed by suffix, so it should be trainable
            if not resp:
                print(
                    f"  🔥 FAIL: Assistant EOS at {pos} has response_mask=False (expected True)"
                )
                all_pass = False
            else:
                print(f"  ✓ Assistant EOS at {pos}: response_mask=True")
        else:
            # System/user EOS - check if it's correctly not trainable
            if resp:
                print(f"  Note: EOS at {pos} is trainable (possibly system/user)")

    # Check 5: loss_mask[i] should equal response_mask[i+1] for all i < len-1
    print("\n[Check 5] loss_mask[i] = response_mask[i+1] (torch.roll correctness)")
    mismatches = []
    for i in range(len(episode_data.token_ids) - 1):
        expected = episode_data.response_mask[i + 1].float().item()
        actual = loss_mask_tensor[i].item()
        if expected != actual:
            mismatches.append((i, expected, actual))

    if mismatches:
        print(f"  🔥 FAIL: {len(mismatches)} positions have incorrect loss_mask")
        for i, exp, act in mismatches[:5]:  # Show first 5
            print(f"    Position {i}: expected {exp:.1f}, got {act:.1f}")
        all_pass = False
    else:
        print(f"  ✓ All positions correctly shifted")

    # Final summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    if all_pass:
        print("\n✅ ALL CHECKS PASSED")
        print("\n   V6 suffix token handling is CORRECT:")
        print("   - Suffix tokens have response_mask=False")
        print("   - Suffix tokens have loss_mask=0.0")
        print("   - Suffix tokens have targets=IGNORE")
        print("   - Suffix tokens will NOT contribute to loss")
        print("\n   CONCLUSION: Suffix tokens are NOT the cause of KL explosion.")
        print("   The issue must be due to:")
        print("   - Real model divergence between policy and ref")
        print("   - Numerical issues in specific training batches")
        print("   - Other factors not related to suffix handling")
    else:
        print("\n❌ CHECKS FAILED")
        print("\n   🔥 BUG DETECTED: Suffix tokens are leaking into loss!")
        print("   This could cause KL explosion if ref_model and policy")
        print("   compute different logprobs for suffix positions.")

    print("\n" + "=" * 80)
    print()


if __name__ == "__main__":
    test_loss_mask_with_suffix()
