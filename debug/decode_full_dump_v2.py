#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Comprehensive dump analysis - show detailed table for every token.
"""

import sys

sys.path.insert(0, "/home/felipemello/forge")

import torch
from vllm.transformers_utils.tokenizer import get_tokenizer


def analyze_dump_detailed(dump_path, seq_idx=0, max_tokens=None):
    """Analyze dump with detailed per-token breakdown."""
    print(f"\nLoading: {dump_path}")
    dump = torch.load(dump_path, map_location="cpu")
    tokenizer = get_tokenizer("Qwen/Qwen3-1.7B")

    # Extract tensors for this sequence
    input_ids = dump["input_ids"][seq_idx]
    targets = dump["targets"][seq_idx]
    loss_mask = dump["loss_mask"][seq_idx]
    logprobs = dump.get("logprobs", None)
    ref_logprobs = dump.get("ref_logprobs", None)
    advantages = dump.get("advantages", None)
    kl = dump.get("kl", None)

    # Get per-token data
    if logprobs is not None:
        logprobs = logprobs[seq_idx]
    if ref_logprobs is not None:
        ref_logprobs = ref_logprobs[seq_idx]
    if advantages is not None:
        advantages = advantages[seq_idx]
    if kl is not None:
        kl = kl[seq_idx]

    seq_len = len(input_ids)
    if max_tokens:
        seq_len = min(seq_len, max_tokens)

    print(f"\n{'='*120}")
    print(f"SEQUENCE {seq_idx} - DETAILED TOKEN ANALYSIS")
    print(f"{'='*120}")
    print(f"Total tokens: {len(input_ids)}")
    print(f"Trainable tokens: {loss_mask.sum().item():.0f}")
    print(f"{'='*120}")

    # Decode full sequence for context
    full_text = tokenizer.decode(input_ids.tolist())
    print(f"\n--- FULL DECODED TEXT ---")
    print(full_text[:1000])
    if len(full_text) > 1000:
        print(f"\n... (truncated, {len(full_text)} total chars)")
    print()

    # Build header
    header_parts = [
        ("Pos", 5),
        ("TokenID", 8),
        ("Decoded", 25),
        ("Target", 8),
        ("Mask", 5),
    ]

    if logprobs is not None:
        header_parts.append(("Policy_LP", 10))
    if ref_logprobs is not None:
        header_parts.append(("Ref_LP", 10))
    if logprobs is not None and ref_logprobs is not None:
        header_parts.append(("LP_Diff", 10))
    if kl is not None:
        header_parts.append(("KL", 10))
    if advantages is not None:
        header_parts.append(("Adv", 8))

    # Print header
    header_line = " | ".join(name.ljust(width) for name, width in header_parts)
    print("=" * len(header_line))
    print(header_line)
    print("=" * len(header_line))

    # Print each token
    for i in range(seq_len):
        tok_id = input_ids[i].item()
        tgt = targets[i].item()
        mask = loss_mask[i].item()

        # Decode token
        tok_str = tokenizer.decode([tok_id])

        # Truncate and escape special chars for display
        tok_str_display = repr(tok_str)[1:-1]  # Remove outer quotes
        if len(tok_str_display) > 23:
            tok_str_display = tok_str_display[:20] + "..."

        # Special token markers
        marker = ""
        if tok_id == 151667:
            marker = " <think>"
        elif tok_id == 151668:
            marker = " </think>"
        elif tok_id == 151645:
            marker = " <|im_end|>"
        elif tok_id == 151644:
            marker = " <|im_start|>"
        elif tok_id == 77091:
            marker = " [assistant]"
        elif tok_id == 151643:
            marker = " <|endoftext|>"

        # Add marker to display
        if marker:
            tok_str_display = f"{tok_str_display}{marker}"
            if len(tok_str_display) > 23:
                tok_str_display = tok_str_display[:23]

        # Build row
        row_parts = [
            f"{i}".ljust(5),
            f"{tok_id}".ljust(8),
            tok_str_display.ljust(25),
            f"{tgt}".ljust(8) if tgt != -100 else "IGNORE".ljust(8),
            f"{mask:.1f}".ljust(5),
        ]

        if logprobs is not None:
            row_parts.append(f"{logprobs[i].item():>9.4f}".ljust(10))
        if ref_logprobs is not None:
            row_parts.append(f"{ref_logprobs[i].item():>9.4f}".ljust(10))
        if logprobs is not None and ref_logprobs is not None:
            diff = ref_logprobs[i].item() - logprobs[i].item()
            row_parts.append(f"{diff:>9.4f}".ljust(10))
        if kl is not None:
            kl_val = kl[i].item()
            # Highlight huge KL values
            if abs(kl_val) > 100:
                row_parts.append(f"{kl_val:>9.2e} ⚠".ljust(10))
            else:
                row_parts.append(f"{kl_val:>9.4f}".ljust(10))
        if advantages is not None:
            # Advantages are per-sequence, so they're constant
            if i == 0:
                row_parts.append(f"{advantages.item():>7.3f}".ljust(8))
            else:
                row_parts.append(" " * 8)

        # Color code trainable tokens
        prefix = "✓" if mask == 1.0 else "·"
        print(f"{prefix} {' | '.join(row_parts)}")

        # Add section breaks at message boundaries
        if tok_id in [151645, 151644]:  # <|im_end|> or <|im_start|>
            print("-" * len(header_line))

    print("=" * len(header_line))

    # Summary statistics
    print(f"\n--- SUMMARY STATISTICS ---")
    print(f"Total tokens: {len(input_ids)}")
    print(f"Trainable tokens: {loss_mask.sum().item():.0f}")

    if logprobs is not None:
        trainable_mask = loss_mask.bool()
        if trainable_mask.any():
            print(f"\nPolicy logprobs (trainable only):")
            print(f"  Mean: {logprobs[trainable_mask].mean().item():.4f}")
            print(f"  Min:  {logprobs[trainable_mask].min().item():.4f}")
            print(f"  Max:  {logprobs[trainable_mask].max().item():.4f}")
            print(f"  Std:  {logprobs[trainable_mask].std().item():.4f}")

    if ref_logprobs is not None:
        if trainable_mask.any():
            print(f"\nRef logprobs (trainable only):")
            print(f"  Mean: {ref_logprobs[trainable_mask].mean().item():.4f}")
            print(f"  Min:  {ref_logprobs[trainable_mask].min().item():.4f}")
            print(f"  Max:  {ref_logprobs[trainable_mask].max().item():.4f}")
            print(f"  Std:  {ref_logprobs[trainable_mask].std().item():.4f}")

    if logprobs is not None and ref_logprobs is not None:
        if trainable_mask.any():
            diff = ref_logprobs[trainable_mask] - logprobs[trainable_mask]
            print(f"\nLogprob difference (ref - policy, trainable only):")
            print(f"  Mean: {diff.mean().item():.4f}")
            print(f"  Min:  {diff.min().item():.4f}")
            print(f"  Max:  {diff.max().item():.4f}")
            print(f"  Std:  {diff.std().item():.4f}")

    if kl is not None:
        if trainable_mask.any():
            print(f"\nKL divergence (trainable only):")
            kl_trainable = kl[trainable_mask]
            print(f"  Mean: {kl_trainable.mean().item():.4f}")
            print(f"  Min:  {kl_trainable.min().item():.4f}")
            print(f"  Max:  {kl_trainable.max().item():.4f}")
            print(f"  Std:  {kl_trainable.std().item():.4f}")

            # Check for huge values
            huge_kl = (kl_trainable.abs() > 100).sum().item()
            if huge_kl > 0:
                print(f"  ⚠️  {huge_kl} tokens with |KL| > 100!")

    if advantages is not None:
        print(f"\nAdvantage: {advantages.item():.6f}")

    # Check for anomalies
    print(f"\n--- ANOMALY DETECTION ---")
    if logprobs is not None and trainable_mask.any():
        very_negative_lp = (logprobs[trainable_mask] < -20).sum().item()
        if very_negative_lp > 0:
            print(f"⚠️  {very_negative_lp} trainable tokens with logprob < -20")

    if ref_logprobs is not None and trainable_mask.any():
        very_negative_ref = (ref_logprobs[trainable_mask] < -20).sum().item()
        if very_negative_ref > 0:
            print(f"⚠️  {very_negative_ref} trainable tokens with ref_logprob < -20")

    # Check targets
    trainable_targets = targets[trainable_mask]
    if trainable_mask.any():
        if (trainable_targets == -100).any():
            print(f"⚠️  Some trainable positions have target=-100 (IGNORE)!")


def main():
    # Analyze both dumps
    dumps = [
        ("/tmp/grpo_loss_debug_20251119_231139.pt", 0),
        ("/tmp/grpo_loss_debug_20251119_231131.pt", 1),
    ]

    for dump_path, seq_idx in dumps:
        try:
            analyze_dump_detailed(dump_path, seq_idx, max_tokens=None)
            print("\n" * 3)
        except Exception as e:
            print(f"\nError analyzing {dump_path} seq {seq_idx}: {e}")
            import traceback

            traceback.print_exc()


if __name__ == "__main__":
    main()
