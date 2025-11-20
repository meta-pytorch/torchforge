#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Standalone test for next-token prediction targets and training masks.

This script tests the alignment between tokens, targets, and masks for multi-turn conversations.
"""

from typing import List

import torch
from tabulate import tabulate


CROSS_ENTROPY_IGNORE_IDX = -100


def create_next_token_targets(
    all_token_ids: torch.Tensor,  # [seq_len]
    response_mask: torch.Tensor,  # [seq_len] bool
    eos_token_id: int,
    ignore_index: int = CROSS_ENTROPY_IGNORE_IDX,
) -> torch.Tensor:
    """
    Create next-token prediction targets with EOS masking for multi-turn.

    Args:
        all_token_ids: All conversation tokens [seq_len]
        response_mask: Boolean mask, True for trainable tokens
        eos_token_id: EOS token ID to mask (prevents predicting after EOS)
        ignore_index: Value to use for masked positions

    Returns:
        targets: Target tokens for next-token prediction [seq_len]
    """
    targets = torch.full_like(all_token_ids, ignore_index)

    # Shift: targets[i] should predict all_token_ids[i+1]
    targets[:-1] = all_token_ids[1:]

    # Mask targets for non-trainable tokens
    targets[~response_mask] = ignore_index

    # EOS is part of response_mask, but we should ignore the prediction
    targets[all_token_ids == eos_token_id] = ignore_index

    return targets


def test_exact_user_example():
    """
    Test the EXACT example from the user:

    Multi-turn sequence:
    - System message
    - User message
    - Agent says "Hello there" + EOS
    - User message
    - Agent says "I am bob" + EOS

    Only agent responses should be trainable.
    """
    print("\n" + "=" * 100)
    print("TEST: Multi-turn conversation with 'Hello there' and 'I am bob'")
    print("=" * 100)
    print()

    # Define token IDs (using readable numbers)
    # Let's say: EOS=100, typical tokens are < 100

    # Build the sequence step by step
    token_strs = [
        # System message
        "<|im_start|>",
        "system",
        "\n",
        "You",
        "are",
        "helpful",
        "<|im_end|>",
        # User message 1
        "<|im_start|>",
        "user",
        "\n",
        "Hi",
        "<|im_end|>",
        # Assistant response 1: "Hello there"
        "<|im_start|>",
        "assistant",
        "\n",
        "Hello",
        "there",
        "<|im_end|>",
        # User message 2
        "<|im_start|>",
        "user",
        "\n",
        "Who",
        "are",
        "you",
        "<|im_end|>",
        # Assistant response 2: "I am bob"
        "<|im_start|>",
        "assistant",
        "\n",
        "I",
        "am",
        "bob",
        "<|im_end|>",
    ]

    # Map to token IDs (simplified)
    token_map = {s: i + 1 for i, s in enumerate(set(token_strs))}
    token_map["<|im_end|>"] = 100  # EOS token

    tokens = [token_map[s] for s in token_strs]

    # Create mask: True only for assistant content tokens (not the prefix)
    # Pattern: <|im_start|> assistant \n [CONTENT TOKENS] <|im_end|>
    #          False        False      False [TRUE...]    TRUE (EOS)

    mask = []
    in_assistant = False
    for i, s in enumerate(token_strs):
        if s == "assistant":
            in_assistant = True
            mask.append(False)  # "assistant" token itself is not trainable
        elif in_assistant and s == "\n":
            mask.append(False)  # newline after "assistant" is not trainable
        elif in_assistant and s == "<|im_end|>":
            mask.append(
                True
            )  # EOS is marked as trainable (but will be excluded in targets)
            in_assistant = False
        elif in_assistant:
            mask.append(True)  # Actual content is trainable
        else:
            mask.append(False)  # System, user, prefixes are not trainable

    all_token_ids = torch.tensor(tokens, dtype=torch.long)
    response_mask = torch.tensor(mask, dtype=torch.bool)
    eos_token_id = 100

    targets = create_next_token_targets(all_token_ids, response_mask, eos_token_id)

    # Create training mask (what actually contributes to loss)
    # This should be: position i is trainable if token i+1 is trainable AND token i is not EOS
    training_mask = torch.zeros_like(response_mask, dtype=torch.float)
    for i in range(len(tokens) - 1):
        # Position i predicts token i+1
        # We train on position i if:
        # 1. Token i+1 is trainable (response_mask[i+1] == True)
        # 2. Token i is NOT EOS (don't predict after EOS)
        if response_mask[i + 1] and tokens[i] != eos_token_id:
            training_mask[i] = 1.0

    # Build the table
    table_data = []
    for i in range(len(tokens)):
        token_str = token_strs[i]
        token_id = tokens[i]

        # Response mask
        resp_mask_str = "✓" if mask[i] else "✗"

        # Target
        target_val = targets[i].item()
        if target_val == CROSS_ENTROPY_IGNORE_IDX:
            target_str = "IGNORE"
        else:
            target_str = f"{target_val}"
            # Find what token this is
            for s, tid in token_map.items():
                if tid == target_val:
                    target_str = f"{target_val} ({s})"
                    break

        # Training mask (what contributes to loss)
        train_mask_val = training_mask[i].item()
        train_mask_str = f"{train_mask_val:.1f}"

        # Notes
        notes = []
        if i < len(tokens) - 1:
            next_token = token_strs[i + 1]
            notes.append(f"→ {next_token}")

        table_data.append(
            [
                i,
                token_str,
                token_id,
                resp_mask_str,
                target_str,
                train_mask_str,
                " ".join(notes),
            ]
        )

    headers = [
        "Idx",
        "Token",
        "ID",
        "Response\nMask",
        "Target",
        "Training\nMask",
        "Predicts",
    ]
    print(tabulate(table_data, headers=headers, tablefmt="grid"))

    print("\n" + "=" * 100)
    print("KEY INSIGHTS FROM THIS EXAMPLE")
    print("=" * 100)
    print()
    print("1. RESPONSE_MASK vs TRAINING_MASK:")
    print("   - response_mask: Marks which tokens ARE responses (content + EOS)")
    print("   - training_mask: Marks which POSITIONS contribute to loss")
    print("   - They are NOT the same!")
    print()
    print("2. THE SHIFT:")
    print("   - Position i predicts token i+1")
    print("   - If token i+1 is trainable, then position i contributes to loss")
    print(
        "   - training_mask[i] = 1.0 if (response_mask[i+1] == True AND token[i] != EOS)"
    )
    print()
    print("3. WHY MASK IS 0.0/1.0 (not bool):")
    print(
        "   - Used in loss computation: loss = (per_token_loss * training_mask).sum()"
    )
    print("   - Float mask allows element-wise multiplication")
    print()
    print("4. EOS HANDLING:")
    print("   - EOS appears in response_mask (it's part of the response)")
    print("   - Position before EOS should predict EOS (training_mask=1.0)")
    print(
        "   - Position AT EOS should NOT train (training_mask=0.0, don't predict after EOS)"
    )
    print()

    # Show specific examples
    print("=" * 100)
    print("SPECIFIC EXAMPLES FROM THE TABLE")
    print("=" * 100)
    print()

    # Find "Hello" token
    hello_idx = token_strs.index("Hello")
    there_idx = token_strs.index("there")

    print(f"Position {hello_idx} (token='Hello'):")
    print(f"  - Predicts: '{token_strs[hello_idx + 1]}'")
    print(f"  - response_mask[{hello_idx}] = {mask[hello_idx]}")
    print(f"  - training_mask[{hello_idx}] = {training_mask[hello_idx].item()}")
    print(f"  - target[{hello_idx}] = {targets[hello_idx].item()}")
    print(f"  → Position {hello_idx} TRAINS to predict '{token_strs[hello_idx + 1]}'")
    print()

    # Find position before first EOS
    first_eos_idx = tokens.index(100)
    before_eos_idx = first_eos_idx - 1

    print(f"Position {before_eos_idx} (token='{token_strs[before_eos_idx]}'):")
    print(f"  - Predicts: '<|im_end|>' (EOS)")
    print(f"  - response_mask[{before_eos_idx}] = {mask[before_eos_idx]}")
    print(
        f"  - training_mask[{before_eos_idx}] = {training_mask[before_eos_idx].item()}"
    )
    print(
        f"  - target[{before_eos_idx}] = {targets[before_eos_idx].item()} (should be {eos_token_id})"
    )
    print(f"  → Position {before_eos_idx} TRAINS to predict EOS")
    print()

    print(f"Position {first_eos_idx} (token='<|im_end|>'):")
    print(f"  - Token IS EOS")
    print(f"  - response_mask[{first_eos_idx}] = {mask[first_eos_idx]}")
    print(f"  - training_mask[{first_eos_idx}] = {training_mask[first_eos_idx].item()}")
    print(f"  - target[{first_eos_idx}] = {targets[first_eos_idx].item()}")
    print(f"  → Position {first_eos_idx} does NOT train (don't predict after EOS)")
    print()

    print("=" * 100)
    print("HOW LOSS COMPUTATION WORKS")
    print("=" * 100)
    print()
    print("In the GRPO loss function:")
    print()
    print("  logprobs = compute_logprobs(logits, all_tokens)  # [seq_len]")
    print("  per_token_loss = -(logprobs * advantages)        # [seq_len]")
    print("  masked_loss = per_token_loss * training_mask     # [seq_len]")
    print("  loss = masked_loss.sum() / training_mask.sum()   # scalar")
    print()
    print("Only positions where training_mask=1.0 contribute to the loss!")
    print()
    print("This means:")
    print("  - System, user messages: training_mask=0.0 → no gradient")
    print("  - Assistant prefix: training_mask=0.0 → no gradient")
    print("  - Assistant content: training_mask=1.0 → gets gradient")
    print("  - Position after EOS: training_mask=0.0 → no gradient")
    print()

    print("=" * 100)
    print("SUMMARY: WHAT NEEDS TO BE FIXED")
    print("=" * 100)
    print()
    print("1. RENAME 'response_mask' to 'response_token_mask' for clarity")
    print("   - It marks which tokens ARE responses")
    print()
    print(
        "2. CREATE 'training_mask' (or 'loss_mask') derived from response_token_mask:"
    )
    print(
        "   - training_mask[i] = 1.0 if response_token_mask[i+1] and not is_eos(token[i])"
    )
    print("   - This is the mask used in loss computation")
    print()
    print("3. FIX compute_logprobs call:")
    print("   - Currently: compute_logprobs(logits, all_tokens, align=False)")
    print("   - Problem: logits[i] predicts token[i+1], not token[i]!")
    print("   - Solution: Shift properly or use targets")
    print()
    print("4. USE targets in loss computation (if created):")
    print("   - targets already has the shift built in")
    print("   - targets[i] = all_tokens[i+1] where trainable, else IGNORE")
    print("   - Can derive training_mask from: (targets != IGNORE).float()")
    print()

    return True


def test_simple_hello_bob():
    """
    Simplified version with just the tokens, no template.

    Sequence:
    - "prompt" "prompt"
    - "Hello" "there" EOS
    - "prompt" "prompt"
    - "I" "am" "bob" EOS
    """
    print("\n" + "=" * 100)
    print("TEST: Simplified 'Hello there' and 'I am bob' example")
    print("=" * 100)
    print()

    # Token strings
    token_strs = [
        "Prompt",
        "prompt",  # User message 1
        "Hello",
        "there",
        "EOS",  # Agent response 1
        "Prompt",
        "prompt",  # User message 2
        "I",
        "am",
        "bob",
        "EOS",  # Agent response 2
    ]

    # Token IDs
    tokens = [1, 2, 3, 4, 100, 5, 6, 7, 8, 9, 100]

    # response_mask: True for agent responses (including EOS)
    response_mask = [
        False,
        False,
        True,
        True,
        True,
        False,
        False,
        True,
        True,
        True,
        True,
    ]

    all_token_ids = torch.tensor(tokens, dtype=torch.long)
    response_mask_tensor = torch.tensor(response_mask, dtype=torch.bool)
    eos_token_id = 100

    targets = create_next_token_targets(
        all_token_ids, response_mask_tensor, eos_token_id
    )

    # Create CORRECT training mask
    # Position i is trainable if token[i+1] is trainable AND token[i] is not EOS
    training_mask = torch.zeros(len(tokens), dtype=torch.float)
    for i in range(len(tokens) - 1):
        if response_mask[i + 1] and tokens[i] != eos_token_id:
            training_mask[i] = 1.0

    # Build table
    table_data = []
    for i in range(len(tokens)):
        token_str = token_strs[i]
        token_id = tokens[i]

        resp_mask_str = "1" if response_mask[i] else "0"

        target_val = targets[i].item()
        if target_val == CROSS_ENTROPY_IGNORE_IDX:
            target_str = "IGNORE"
        else:
            if i + 1 < len(token_strs):
                target_str = f"{target_val} (→{token_strs[i+1]})"
            else:
                target_str = f"{target_val}"

        train_mask_str = f"{training_mask[i].item():.1f}"

        # Show what contributes to loss
        contributes = "YES" if training_mask[i].item() == 1.0 else "NO"

        table_data.append(
            [
                i,
                token_str,
                token_id,
                resp_mask_str,
                target_str,
                train_mask_str,
                contributes,
            ]
        )

    headers = [
        "Idx",
        "Token",
        "ID",
        "Resp\nMask",
        "Target\n(predicts)",
        "Train\nMask",
        "Loss?",
    ]
    print(tabulate(table_data, headers=headers, tablefmt="grid"))

    print("\n" + "=" * 100)
    print("OBSERVATIONS")
    print("=" * 100)
    print()
    print(f"Total tokens: {len(tokens)}")
    print(f"Response tokens (response_mask=1): {sum(response_mask)}")
    print(f"Training positions (training_mask=1): {int(training_mask.sum().item())}")
    print()
    print("Notice:")
    print("  - Response tokens: 7 (includes both EOS)")
    print("  - Training positions: 5 (excludes positions AT EOS and after EOS)")
    print("  - The difference: 2 EOS positions don't train")
    print()

    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 100)
    print("TESTING NEXT-TOKEN PREDICTION: TARGETS AND TRAINING MASKS")
    print("=" * 100)

    test_exact_user_example()
    test_simple_hello_bob()

    print("\n" + "=" * 100)
    print("ALL TESTS COMPLETED ✅")
    print("=" * 100)
    print()


if __name__ == "__main__":
    try:
        main()
    except ImportError:
        print("Installing tabulate...")
        import subprocess

        subprocess.check_call(["pip", "install", "-q", "tabulate"])
        main()
