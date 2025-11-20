#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Verify the EOS hypothesis by decoding tokens and checking response_mask.
"""

import sys

import torch

sys.path.insert(0, "/home/felipemello/forge")
from vllm.transformers_utils.tokenizer import get_tokenizer

# Load dump
dump_file = (
    sys.argv[1] if len(sys.argv) > 1 else "/tmp/grpo_loss_debug_20251119_140858.pt"
)

print("=" * 80)
print(f"Loading: {dump_file}")
print("=" * 80)

data = torch.load(dump_file, map_location="cpu")

# Get tokenizer
model_name = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = get_tokenizer(model_name)
eos_token_id = tokenizer.eos_token_id

print(f"\nEOS token ID: {eos_token_id}")

# Extract tensors
input_ids = data["input_ids"]
targets = data["targets"]
loss_mask = data["loss_mask"]
logprobs = data["logprobs"]
ref_logprobs = data["ref_logprobs"]
kl = data["kl"]

batch_size, seq_len = input_ids.shape
ignore_idx = -100

# ============================================================================
# Step 1: Reconstruct response_mask from loss_mask
# ============================================================================
print("\n" + "=" * 80)
print("STEP 1: Reconstructing response_mask from loss_mask")
print("=" * 80)

# loss_mask[i] = response_mask[i+1]
# So: response_mask[i+1] = loss_mask[i]
# Therefore: response_mask[i] = loss_mask[i-1]

response_mask = torch.zeros_like(loss_mask)
response_mask[:, 1:] = loss_mask[:, :-1]  # Shift back
response_mask[:, 0] = 0.0  # First position unknown, assume False

print(f"Reconstructed response_mask shape: {response_mask.shape}")
print(f"Response tokens (response_mask=1): {response_mask.sum().item()}")
print(f"Trainable positions (loss_mask=1): {loss_mask.sum().item()}")
print(f"Difference: {response_mask.sum().item() - loss_mask.sum().item()}")

# ============================================================================
# Step 2: Find all EOS positions
# ============================================================================
print("\n" + "=" * 80)
print("STEP 2: Finding all EOS positions")
print("=" * 80)

eos_positions = input_ids == eos_token_id
eos_count = eos_positions.sum().item()

print(f"Total EOS tokens: {eos_count}")

# Find EOS positions with loss_mask=1 (being trained on)
eos_trainable = eos_positions & (loss_mask == 1.0)
eos_trainable_count = eos_trainable.sum().item()

print(f"EOS positions with loss_mask=1: {eos_trainable_count}")
print(f"EOS positions with loss_mask=0: {eos_count - eos_trainable_count}")

if eos_trainable_count > 0:
    print(f"\n⚠️  BUG CONFIRMED: {eos_trainable_count} EOS positions have loss_mask=1!")

# ============================================================================
# Step 3: Check KL values at EOS positions
# ============================================================================
print("\n" + "=" * 80)
print("STEP 3: Analyzing KL at EOS positions")
print("=" * 80)

if eos_trainable_count > 0:
    kl_at_eos = kl[eos_trainable]
    diff_at_eos = (ref_logprobs - logprobs)[eos_trainable]

    print(f"KL at EOS positions (where loss_mask=1):")
    print(f"   Mean: {kl_at_eos.mean().item():.4f}")
    print(f"   Min:  {kl_at_eos.min().item():.4f}")
    print(f"   Max:  {kl_at_eos.max().item():.4f}")

    print(f"Logprob diff at EOS positions:")
    print(f"   Mean: {diff_at_eos.mean().item():.4f}")
    print(f"   Min:  {diff_at_eos.min().item():.4f}")
    print(f"   Max:  {diff_at_eos.max().item():.4f}")

    # Compare to non-EOS trainable positions
    non_eos_trainable = (loss_mask == 1.0) & (~eos_positions)
    if non_eos_trainable.sum() > 0:
        kl_non_eos = kl[non_eos_trainable]
        diff_non_eos = (ref_logprobs - logprobs)[non_eos_trainable]

        print(f"\nKL at NON-EOS trainable positions:")
        print(f"   Mean: {kl_non_eos.mean().item():.4f}")
        print(f"   Min:  {kl_non_eos.min().item():.4f}")
        print(f"   Max:  {kl_non_eos.max().item():.4f}")

        print(f"Logprob diff at NON-EOS trainable positions:")
        print(f"   Mean: {diff_non_eos.mean().item():.4f}")
        print(f"   Min:  {diff_non_eos.min().item():.4f}")
        print(f"   Max:  {diff_non_eos.max().item():.4f}")

        print(f"\n📊 Comparison:")
        print(f"   EOS KL mean:     {kl_at_eos.mean().item():.4f}")
        print(f"   Non-EOS KL mean: {kl_non_eos.mean().item():.4f}")
        print(
            f"   Ratio:           {kl_at_eos.mean().item() / (kl_non_eos.mean().item() + 1e-8):.2f}x"
        )

# ============================================================================
# Step 4: Decode and show problematic positions
# ============================================================================
print("\n" + "=" * 80)
print("STEP 4: Decoding problematic positions")
print("=" * 80)

# Find top 10 worst KL positions
kl_flat = kl.view(-1)
_, top_indices = torch.topk(kl_flat, k=min(10, kl_flat.numel()))

for rank, idx in enumerate(top_indices[:10]):
    idx = idx.item()
    batch = idx // seq_len
    pos = idx % seq_len

    # Skip if not trainable
    if loss_mask[batch, pos] == 0:
        continue

    kl_val = kl[batch, pos].item()

    print(f"\n--- Rank {rank+1}: KL = {kl_val:.2f} (batch={batch}, pos={pos}) ---")

    # Show context
    start = max(0, pos - 3)
    end = min(seq_len, pos + 4)

    print(
        f"  {'Pos':>4} {'Token':>8} {'Decoded':>15} {'RespMask':>8} {'LossMask':>8} {'Target':>8} {'KL':>8}"
    )
    print(f"  {'-'*75}")

    for i in range(start, end):
        token_id = input_ids[batch, i].item()
        resp_mask = response_mask[batch, i].item()
        loss_mk = loss_mask[batch, i].item()
        tgt = targets[batch, i].item()
        kl_i = kl[batch, i].item()

        # Decode token
        try:
            decoded = tokenizer.decode([token_id])
            # Clean up for display
            decoded = decoded.replace("\n", "\\n").replace("\r", "\\r")
            decoded = decoded[:15]  # Truncate
        except:
            decoded = "???"

        # Check if EOS
        is_eos = " [EOS]" if token_id == eos_token_id else ""
        flag = " ← HERE" if i == pos else ""

        tgt_str = "IGNORE" if tgt == ignore_idx else f"{tgt:6d}"

        print(
            f"  {i:4d} {token_id:8d} {decoded:>15s}{is_eos:6s} {resp_mask:8.1f} {loss_mk:8.1f} {tgt_str:>8s} {kl_i:8.2f}{flag}"
        )

# ============================================================================
# Step 5: Check what happens after EOS
# ============================================================================
print("\n" + "=" * 80)
print("STEP 5: What comes after EOS tokens?")
print("=" * 80)

# Find all EOS positions that are NOT at the end of sequence
eos_coords = torch.where(eos_positions)

print(f"Checking {len(eos_coords[0])} EOS positions...")

suspicious_count = 0
for batch, pos in zip(eos_coords[0][:20], eos_coords[1][:20]):  # Check first 20
    batch = batch.item()
    pos = pos.item()

    if pos >= seq_len - 1:
        continue  # Skip last position

    # Check next 3 tokens
    print(f"\nEOS at batch={batch}, pos={pos}:")

    for offset in range(4):
        if pos + offset >= seq_len:
            break

        i = pos + offset
        token_id = input_ids[batch, i].item()
        resp_mask = response_mask[batch, i].item()
        loss_mk = loss_mask[batch, i].item()

        try:
            decoded = tokenizer.decode([token_id])
            decoded = decoded.replace("\n", "\\n").replace("\r", "\\r")[:20]
        except:
            decoded = "???"

        is_eos_marker = "[EOS]" if token_id == eos_token_id else ""
        flag = ""

        if offset == 0:
            label = "AT EOS"
        elif offset == 1:
            label = "NEXT"
            if resp_mask == 1.0:
                flag = " ⚠️  RESPONSE_MASK=1 (BUG!)"
                suspicious_count += 1
        elif offset == 2:
            label = "NEXT+1"
        else:
            label = "NEXT+2"

        print(
            f"  {label:8s}: pos={i:3d} token={token_id:6d} {is_eos_marker:6s} '{decoded:20s}' resp={resp_mask:.0f} loss={loss_mk:.0f}{flag}"
        )

if suspicious_count > 0:
    print(f"\n🔥 FOUND {suspicious_count} SUSPICIOUS POSITIONS!")
    print(f"   These are tokens AFTER EOS that have response_mask=1")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

print(f"\n1. Total EOS tokens: {eos_count}")
print(f"2. EOS positions being trained (loss_mask=1): {eos_trainable_count}")
if eos_trainable_count > 0:
    print(f"   ⚠️  THIS IS THE BUG!")
    print(f"   We should NOT train at EOS positions (predicting what comes after EOS)")
print(f"3. Suspicious tokens after EOS with response_mask=1: {suspicious_count}")
if suspicious_count > 0:
    print(f"   ⚠️  Root cause: TokenAccumulator is marking post-EOS tokens as responses")

print("\n" + "=" * 80)
