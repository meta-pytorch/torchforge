#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Analyze V6 loss dump files to find the culprit tokens causing KL explosion.

Automatically loads the most recent dump files (V6 only, skips V5).
"""

import sys

sys.path.insert(0, "/home/felipemello/forge")

import glob
import os
from datetime import datetime

import torch
from vllm.transformers_utils.tokenizer import get_tokenizer


def find_recent_dumps(max_age_hours=2):
    """Find dump files created in the last N hours."""
    dump_files = glob.glob("/tmp/grpo_loss_debug_*.pt")

    recent_dumps = []
    now = datetime.now()

    for path in dump_files:
        # Extract timestamp from filename: grpo_loss_debug_YYYYMMDD_HHMMSS.pt
        basename = os.path.basename(path)
        timestamp_str = basename.replace("grpo_loss_debug_", "").replace(".pt", "")

        try:
            file_time = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
            age_hours = (now - file_time).total_seconds() / 3600

            if age_hours <= max_age_hours:
                recent_dumps.append((path, file_time, age_hours))
        except ValueError:
            continue

    # Sort by timestamp (newest first)
    recent_dumps.sort(key=lambda x: x[1], reverse=True)
    return recent_dumps


def analyze_dump(dump_path, tokenizer):
    """Analyze a single dump file and show culprit tokens."""
    print("\n" + "=" * 80)
    print(f"ANALYZING: {os.path.basename(dump_path)}")
    print("=" * 80)

    # Load dump
    dump = torch.load(dump_path, map_location="cpu")

    # Extract tensors
    input_ids = dump["input_ids"]
    targets = dump["targets"]
    loss_mask = dump["loss_mask"]
    logprobs = dump["logprobs"]
    ref_logprobs = dump["ref_logprobs"]
    kl = dump["kl"]

    batch_size, seq_len = input_ids.shape

    print(f"\nDump metadata:")
    print(f"  Trigger stat: {dump['trigger_stat']}")
    print(f"  Trigger value: {dump['trigger_value']:.2f}")
    print(f"  Beta: {dump['beta']}")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")

    # Find positions with masked KL
    masked_kl = kl * loss_mask

    # Statistics
    num_trainable = loss_mask.sum().item()
    kl_mean = (masked_kl.sum() / num_trainable).item() if num_trainable > 0 else 0.0

    print(f"\nKL statistics:")
    print(f"  Trainable positions: {int(num_trainable)}")
    print(f"  KL mean: {kl_mean:.2f}")

    # Analyze each sequence in batch
    for seq_idx in range(min(batch_size, 3)):  # Show first 3 sequences
        print("\n" + "-" * 80)
        print(f"SEQUENCE {seq_idx}")
        print("-" * 80)

        seq_kl = kl[seq_idx]
        seq_mask = loss_mask[seq_idx]
        seq_masked_kl = masked_kl[seq_idx]

        # Find top 10 positions with highest KL
        trainable_positions = torch.where(seq_mask > 0)[0]

        if len(trainable_positions) == 0:
            print("  No trainable positions!")
            continue

        trainable_kl_values = seq_masked_kl[trainable_positions]
        top_k = min(10, len(trainable_positions))
        top_kl_values, top_indices_in_trainable = torch.topk(trainable_kl_values, top_k)
        top_positions = trainable_positions[top_indices_in_trainable]

        print(f"\nTop {top_k} positions with highest KL:")
        print(
            f"{'Pos':>4} {'Input':>10} {'InToken':>15} {'Target':>10} {'TgtToken':>15} "
            f"{'LogProb':>10} {'RefLogP':>10} {'Diff':>8} {'KL':>12}"
        )
        print("-" * 120)

        for pos in top_positions:
            pos_idx = pos.item()

            inp_id = input_ids[seq_idx, pos_idx].item()
            inp_token = tokenizer.decode([inp_id])[:12]

            tgt_id = targets[seq_idx, pos_idx].item()
            if tgt_id == -100:
                tgt_token = "IGNORE"
            else:
                tgt_token = tokenizer.decode([tgt_id])[:12]

            lp = logprobs[seq_idx, pos_idx].item()
            ref_lp = ref_logprobs[seq_idx, pos_idx].item()
            diff = ref_lp - lp
            kl_val = seq_kl[pos_idx].item()

            flag = ""
            if kl_val > 1000:
                flag = " 🔥"

            print(
                f"{pos_idx:4d} {inp_id:10d} {inp_token:>15s} {tgt_id:10d} {tgt_token:>15s} "
                f"{lp:10.4f} {ref_lp:10.4f} {diff:8.4f} {kl_val:12.2f}{flag}"
            )

        # Find THE position with max KL
        max_kl_pos = torch.argmax(seq_masked_kl).item()
        max_kl_val = seq_masked_kl[max_kl_pos].item()

        print(f"\n🔥 MAXIMUM KL position: {max_kl_pos}")
        print(f"   KL value: {max_kl_val:.2f}")

        inp_id = input_ids[seq_idx, max_kl_pos].item()
        tgt_id = targets[seq_idx, max_kl_pos].item()
        lp = logprobs[seq_idx, max_kl_pos].item()
        ref_lp = ref_logprobs[seq_idx, max_kl_pos].item()
        diff = ref_lp - lp

        inp_token = tokenizer.decode([inp_id])
        tgt_token = tokenizer.decode([tgt_id]) if tgt_id != -100 else "IGNORE"

        print(f"   Input token: {inp_id} ({inp_token!r})")
        print(f"   Target token: {tgt_id} ({tgt_token!r})")
        print(f"   Policy logprob: {lp:.4f}")
        print(f"   Ref logprob: {ref_lp:.4f}")
        print(f"   Difference: {diff:.4f}")
        print(f"   exp({diff:.4f}) = {torch.exp(torch.tensor(diff)).item():.2e}")

        # Show context around max position
        context_start = max(0, max_kl_pos - 5)
        context_end = min(seq_len, max_kl_pos + 6)

        print(f"\n   Context (positions {context_start} to {context_end-1}):")
        context_tokens = input_ids[seq_idx, context_start:context_end].tolist()
        context_text = tokenizer.decode(context_tokens)
        print(f"   {context_text!r}")

        # Show token-by-token context
        print(f"\n   Token-by-token context:")
        for i in range(context_start, context_end):
            tok_id = input_ids[seq_idx, i].item()
            tok_str = tokenizer.decode([tok_id])
            mask = seq_mask[i].item()
            marker = ">>> " if i == max_kl_pos else "    "
            print(f"   {marker}[{i:3d}] {tok_id:6d} {tok_str!r:20s} (mask={mask:.1f})")


def main():
    print("\n" + "=" * 80)
    print("V6 LOSS DUMP ANALYZER - Automatic Recent Dumps")
    print("=" * 80)

    # Find recent dumps (last 2 hours)
    recent_dumps = find_recent_dumps(max_age_hours=2)

    if not recent_dumps:
        print("\n❌ No recent dump files found in /tmp/grpo_loss_debug_*.pt")
        print("   (Looking for files created in the last 2 hours)")
        return

    print(f"\n✓ Found {len(recent_dumps)} recent dump file(s):")
    for path, timestamp, age_hours in recent_dumps:
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"  - {os.path.basename(path)}")
        print(
            f"    Created: {timestamp.strftime('%Y-%m-%d %H:%M:%S')} ({age_hours:.1f} hours ago)"
        )
        print(f"    Size: {size_mb:.1f} MB")

    # Load tokenizer
    print("\n✓ Loading tokenizer...")
    tokenizer = get_tokenizer("Qwen/Qwen2.5-0.5B-Instruct")

    # Analyze each dump (most recent first)
    for path, timestamp, age_hours in recent_dumps[:5]:  # Limit to 2 most recent
        try:
            analyze_dump(path, tokenizer)
        except Exception as e:
            print(f"\n❌ Error analyzing {os.path.basename(path)}: {e}")
            import traceback

            traceback.print_exc()

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print()


if __name__ == "__main__":
    main()
