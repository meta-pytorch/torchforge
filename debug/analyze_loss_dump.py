#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Analyze the debug dump files from the loss function.
"""

import sys

import torch

# Load the most recent dump file
dump_file = (
    sys.argv[1] if len(sys.argv) > 1 else "/tmp/grpo_loss_debug_20251119_140858.pt"
)

print("=" * 80)
print(f"Loading dump file: {dump_file}")
print("=" * 80)

data = torch.load(dump_file, map_location="cpu")

# Print what triggered the dump
print(f"\n🔥 TRIGGER: {data['trigger_stat']} = {data['trigger_value']:.2f}")
print(f"   Beta: {data['beta']}")

# Print shapes
print("\n📊 Tensor Shapes:")
print(f"   logits:       {data['logits'].shape}")
print(f"   input_ids:    {data['input_ids'].shape}")
print(f"   targets:      {data['targets'].shape}")
print(f"   loss_mask:    {data['loss_mask'].shape}")
print(f"   logprobs:     {data['logprobs'].shape}")
print(f"   ref_logprobs: {data['ref_logprobs'].shape}")
print(f"   advantages:   {data['advantages'].shape}")

# Get basic stats
batch_size, seq_len = data["input_ids"].shape
num_trainable = data["loss_mask"].sum().item()

print(f"\n📈 Basic Stats:")
print(f"   Batch size: {batch_size}")
print(f"   Sequence length: {seq_len}")
print(f"   Trainable positions: {num_trainable}")

# Analyze targets
targets = data["targets"]
input_ids = data["input_ids"]
loss_mask = data["loss_mask"]
logprobs = data["logprobs"]
ref_logprobs = data["ref_logprobs"]
kl = data["kl"]

print(f"\n🎯 Targets Analysis:")
ignore_idx = -100
num_ignore = (targets == ignore_idx).sum().item()
num_valid = (targets != ignore_idx).sum().item()
print(f"   IGNORE positions: {num_ignore} ({100*num_ignore/(batch_size*seq_len):.1f}%)")
print(f"   Valid targets:    {num_valid} ({100*num_valid/(batch_size*seq_len):.1f}%)")
print(f"   Trainable (loss_mask=1): {num_trainable}")

# Check if targets align with loss_mask
targets_match_mask = ((targets != ignore_idx).float() == loss_mask).all()
print(f"   Targets match loss_mask: {targets_match_mask}")

if not targets_match_mask:
    print("   ⚠️  MISMATCH DETECTED!")
    mismatch_count = ((targets != ignore_idx).float() != loss_mask).sum().item()
    print(f"   Mismatched positions: {mismatch_count}")

# Analyze logprobs and ref_logprobs
print(f"\n📉 Logprobs Analysis (trainable positions only):")
trainable_mask = loss_mask.bool()

if num_trainable > 0:
    lp_train = logprobs[trainable_mask]
    ref_lp_train = ref_logprobs[trainable_mask]

    print(f"   Logprobs:")
    print(f"      Mean:  {lp_train.mean().item():.4f}")
    print(f"      Min:   {lp_train.min().item():.4f}")
    print(f"      Max:   {lp_train.max().item():.4f}")
    print(f"      Std:   {lp_train.std().item():.4f}")

    print(f"   Ref Logprobs:")
    print(f"      Mean:  {ref_lp_train.mean().item():.4f}")
    print(f"      Min:   {ref_lp_train.min().item():.4f}")
    print(f"      Max:   {ref_lp_train.max().item():.4f}")
    print(f"      Std:   {ref_lp_train.std().item():.4f}")

    # Logprob difference
    diff = ref_lp_train - lp_train
    print(f"   Logprob Diff (ref - policy):")
    print(f"      Mean:  {diff.mean().item():.4f}")
    print(f"      Min:   {diff.min().item():.4f}")
    print(f"      Max:   {diff.max().item():.4f}")
    print(f"      Std:   {diff.std().item():.4f}")

    # Check for extreme values
    extreme_diff = diff.abs() > 10
    if extreme_diff.any():
        print(
            f"   ⚠️  EXTREME DIFFS: {extreme_diff.sum().item()} positions with |diff| > 10"
        )
        print(f"      Max extreme: {diff.abs().max().item():.4f}")

# Analyze KL divergence
print(f"\n🔥 KL Divergence Analysis (trainable positions only):")
if num_trainable > 0:
    kl_train = kl[trainable_mask]

    print(f"   KL:")
    print(f"      Mean:  {kl_train.mean().item():.4f}")
    print(f"      Min:   {kl_train.min().item():.4f}")
    print(f"      Max:   {kl_train.max().item():.4f}")
    print(f"      Std:   {kl_train.std().item():.4f}")

    # Check for extreme KL
    extreme_kl = kl_train > 1000
    if extreme_kl.any():
        print(f"   🔥 EXTREME KL: {extreme_kl.sum().item()} positions with KL > 1000")
        print(f"      Max KL: {kl_train.max().item():.4f}")

# Find the worst position
print(f"\n🔍 Finding Worst Position:")
kl_flat = kl.view(-1)
worst_idx = kl_flat.argmax().item()
worst_batch = worst_idx // seq_len
worst_pos = worst_idx % seq_len

print(f"   Position: batch={worst_batch}, pos={worst_pos}")
print(f"   input_id:    {input_ids[worst_batch, worst_pos].item()}")
print(f"   target:      {targets[worst_batch, worst_pos].item()}")
print(f"   loss_mask:   {loss_mask[worst_batch, worst_pos].item()}")
print(f"   logprob:     {logprobs[worst_batch, worst_pos].item():.4f}")
print(f"   ref_logprob: {ref_logprobs[worst_batch, worst_pos].item():.4f}")
print(
    f"   diff:        {(ref_logprobs[worst_batch, worst_pos] - logprobs[worst_batch, worst_pos]).item():.4f}"
)
print(f"   KL:          {kl[worst_batch, worst_pos].item():.4f}")

# Show context around worst position
print(f"\n📝 Context around worst position (batch={worst_batch}):")
start = max(0, worst_pos - 5)
end = min(seq_len, worst_pos + 6)

print(
    f"   {'Pos':>4} {'Input':>8} {'Target':>8} {'Mask':>5} {'LogP':>10} {'RefLP':>10} {'Diff':>8} {'KL':>10}"
)
print(f"   {'-'*70}")
for i in range(start, end):
    inp = input_ids[worst_batch, i].item()
    tgt = targets[worst_batch, i].item()
    mask = loss_mask[worst_batch, i].item()
    lp = logprobs[worst_batch, i].item()
    ref_lp = ref_logprobs[worst_batch, i].item()
    diff = ref_lp - lp
    kl_val = kl[worst_batch, i].item()

    tgt_str = "IGNORE" if tgt == ignore_idx else f"{tgt:6d}"
    flag = " ← WORST" if i == worst_pos else ""

    print(
        f"   {i:4d} {inp:8d} {tgt_str:>8s} {mask:5.1f} {lp:10.4f} {ref_lp:10.4f} {diff:8.4f} {kl_val:10.4f}{flag}"
    )

# Check if ref_logprobs are all zeros (uninitialized?)
print(f"\n🔎 Checking for Uninitialized Values:")
ref_lp_all_zero = (ref_logprobs == 0).all()
ref_lp_mostly_zero = (ref_logprobs == 0).sum().item() / (batch_size * seq_len)
print(f"   Ref logprobs all zero: {ref_lp_all_zero}")
print(f"   Ref logprobs fraction zero: {ref_lp_mostly_zero:.2%}")

lp_all_zero = (logprobs == 0).all()
lp_mostly_zero = (logprobs == 0).sum().item() / (batch_size * seq_len)
print(f"   Policy logprobs all zero: {lp_all_zero}")
print(f"   Policy logprobs fraction zero: {lp_mostly_zero:.2%}")

# Check if targets are actually shifted correctly
print(f"\n🔄 Checking Target Shift Correctness:")
print("   First sequence, first 20 positions:")
print(
    f"   {'Pos':>4} {'Input[i]':>10} {'Input[i+1]':>10} {'Target[i]':>10} {'Match':>6}"
)
print(f"   {'-'*50}")
for i in range(min(20, seq_len - 1)):
    inp_i = input_ids[0, i].item()
    inp_next = input_ids[0, i + 1].item()
    tgt_i = targets[0, i].item()

    if tgt_i == ignore_idx:
        match = "N/A"
        tgt_str = "IGNORE"
    else:
        match = "✓" if inp_next == tgt_i else "✗"
        tgt_str = f"{tgt_i:8d}"

    print(f"   {i:4d} {inp_i:10d} {inp_next:10d} {tgt_str:>10s} {match:>6s}")

print("\n" + "=" * 80)
