#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Demonstrate how to use vLLM tokens directly (like VERL) with proper prefix handling.

Shows that prefix tokens come from the anchor/generation_prompt, NOT from re-tokenizing.
"""

import sys

sys.path.insert(0, "/home/felipemello/forge")

import torch
from vllm.transformers_utils.tokenizer import get_tokenizer

# Setup
tokenizer = get_tokenizer("Qwen/Qwen2.5-0.5B-Instruct")
eos_token_id = tokenizer.eos_token_id

# Initial messages
initial_messages = [{"role": "system", "content": "You are helpful."}]

# Simulate what happens during multi-turn conversation
print("=" * 80)
print("MULTI-TURN CONVERSATION WITH VLLM TOKENS (VERL STYLE)")
print("=" * 80)

# ============================================================================
# Initialize: Tokenize initial prompt
# ============================================================================
print("\n[INIT] Tokenizing initial prompt")

# Tokenize with generation prompt to get ready for first generation
prompt_with_gen = tokenizer.apply_chat_template(
    initial_messages,
    add_generation_prompt=True,
    tokenize=True,
)

# Also tokenize without generation prompt to know where it starts
prompt_without_gen = tokenizer.apply_chat_template(
    initial_messages,
    add_generation_prompt=False,
    tokenize=True,
)

generation_prompt_len = len(prompt_with_gen) - len(prompt_without_gen)

# Start with just the prompt (no generation prompt yet)
accumulated_tokens = prompt_without_gen.copy()
response_mask = [False] * len(accumulated_tokens)

print(f"Initial tokens: {accumulated_tokens}")
print(f"Response mask:  {response_mask}")
print(f"Generation prompt length: {generation_prompt_len}")

# ============================================================================
# Turn 1: User says "hi"
# ============================================================================
print("\n" + "=" * 80)
print("TURN 1: User says 'hi'")
print("=" * 80)

# Compute delta for user message
temp_messages = [*initial_messages, {"role": "user", "content": "hi"}]
temp_tokens = tokenizer.apply_chat_template(
    temp_messages, add_generation_prompt=False, tokenize=True
)
user_delta_1 = temp_tokens[len(accumulated_tokens) :]

accumulated_tokens.extend(user_delta_1)
response_mask.extend([False] * len(user_delta_1))

print(f"User delta: {user_delta_1}")
print(f"Decoded: {repr(tokenizer.decode(user_delta_1))}")
print(f"Total tokens: {len(accumulated_tokens)}")

# ============================================================================
# Turn 1: Agent responds "hi there!"
# ============================================================================
print("\n" + "=" * 80)
print("TURN 1: Agent responds 'hi there!' (using vLLM tokens)")
print("=" * 80)

# Simulate vLLM generation (returns tokens WITHOUT prefix, WITH EOS)
vllm_response_1_text = "hi there!"
vllm_response_1_tokens = tokenizer.encode(
    vllm_response_1_text, add_special_tokens=False
) + [eos_token_id]

print(f"vLLM returns: {vllm_response_1_tokens}")
print(f"Decoded: {repr(tokenizer.decode(vllm_response_1_tokens))}")

# Get generation prompt tokens (these go BEFORE vLLM tokens)
# We compute this from the anchor
anchor_without = tokenizer.apply_chat_template(
    [{"role": "system", "content": ""}, {"role": "user", "content": ""}],
    add_generation_prompt=False,
    tokenize=True,
)
anchor_with = tokenizer.apply_chat_template(
    [{"role": "system", "content": ""}, {"role": "user", "content": ""}],
    add_generation_prompt=True,
    tokenize=True,
)
generation_prompt_tokens = anchor_with[len(anchor_without) :]

print(f"\nGeneration prompt tokens: {generation_prompt_tokens}")
print(f"Decoded: {repr(tokenizer.decode(generation_prompt_tokens))}")

# Add generation prompt (NOT trainable)
accumulated_tokens.extend(generation_prompt_tokens)
response_mask.extend([False] * len(generation_prompt_tokens))

# Add vLLM tokens (trainable)
accumulated_tokens.extend(vllm_response_1_tokens)
response_mask.extend([True] * len(vllm_response_1_tokens))

print(f"\nAfter adding generation prompt + vLLM tokens:")
print(f"  Total tokens: {len(accumulated_tokens)}")
print(f"  Response tokens: {sum(response_mask)}")

# ============================================================================
# Turn 2: User says "hello"
# ============================================================================
print("\n" + "=" * 80)
print("TURN 2: User says 'hello'")
print("=" * 80)

# Update messages
messages_so_far = [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "hi"},
    {"role": "assistant", "content": vllm_response_1_text},
    {"role": "user", "content": "hello"},
]

# Compute delta
temp_tokens_2 = tokenizer.apply_chat_template(
    messages_so_far, add_generation_prompt=False, tokenize=True
)
user_delta_2 = temp_tokens_2[len(accumulated_tokens) :]

accumulated_tokens.extend(user_delta_2)
response_mask.extend([False] * len(user_delta_2))

print(f"User delta: {user_delta_2}")
print(f"Decoded: {repr(tokenizer.decode(user_delta_2))}")
print(f"Total tokens: {len(accumulated_tokens)}")

# ============================================================================
# Turn 2: Agent responds "hello"
# ============================================================================
print("\n" + "=" * 80)
print("TURN 2: Agent responds 'hello' (using vLLM tokens)")
print("=" * 80)

# Simulate vLLM
vllm_response_2_text = "hello"
vllm_response_2_tokens = tokenizer.encode(
    vllm_response_2_text, add_special_tokens=False
) + [eos_token_id]

print(f"vLLM returns: {vllm_response_2_tokens}")
print(f"Decoded: {repr(tokenizer.decode(vllm_response_2_tokens))}")

# Add generation prompt (same tokens as before)
accumulated_tokens.extend(generation_prompt_tokens)
response_mask.extend([False] * len(generation_prompt_tokens))

# Add vLLM tokens
accumulated_tokens.extend(vllm_response_2_tokens)
response_mask.extend([True] * len(vllm_response_2_tokens))

print(f"\nAfter adding generation prompt + vLLM tokens:")
print(f"  Total tokens: {len(accumulated_tokens)}")
print(f"  Response tokens: {sum(response_mask)}")

# ============================================================================
# Final verification
# ============================================================================
print("\n" + "=" * 80)
print("FINAL VERIFICATION")
print("=" * 80)

# Verify our accumulated tokens match ground truth
final_messages = [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "hi"},
    {"role": "assistant", "content": vllm_response_1_text},
    {"role": "user", "content": "hello"},
    {"role": "assistant", "content": vllm_response_2_text},
]

ground_truth = tokenizer.apply_chat_template(
    final_messages, add_generation_prompt=False, tokenize=True
)

print(f"Accumulated length: {len(accumulated_tokens)}")
print(f"Ground truth length: {len(ground_truth)}")
print(f"Match: {accumulated_tokens == ground_truth}")

if accumulated_tokens != ground_truth:
    print(f"\n⚠️  MISMATCH!")
    print(f"Accumulated: {accumulated_tokens}")
    print(f"Ground truth: {ground_truth}")
else:
    print(f"\n✅ PERFECT MATCH!")

# ============================================================================
# Show where prefixes are
# ============================================================================
print("\n" + "=" * 80)
print("TOKEN BREAKDOWN")
print("=" * 80)

# Decode full sequence
full_decoded = tokenizer.decode(accumulated_tokens)

print(f"\nFull sequence ({len(accumulated_tokens)} tokens):")
response_mask_tensor = torch.tensor(response_mask, dtype=torch.bool)

for i, (token, is_response) in enumerate(zip(accumulated_tokens, response_mask)):
    decoded = tokenizer.decode([token])
    # Clean for display
    decoded = decoded.replace("\n", "\\n").replace("\r", "\\r")
    if len(decoded) > 15:
        decoded = decoded[:15] + "..."

    marker = "RESP" if is_response else "    "
    eos_marker = " [EOS]" if token == eos_token_id else ""

    print(f"  {i:3d}: {token:6d} {decoded:20s} {marker}{eos_marker}")

# ============================================================================
# Check: No newlines after EOS with response_mask=True
# ============================================================================
print("\n" + "=" * 80)
print("CHECKING FOR BUG (tokens after EOS with response_mask=True)")
print("=" * 80)

bug_found = False
for i in range(len(accumulated_tokens) - 1):
    if accumulated_tokens[i] == eos_token_id and response_mask[i]:
        # Check next token
        if response_mask[i + 1]:
            print(f"🔥 BUG at position {i}!")
            print(f"  Token {i}: EOS with response_mask=True")
            print(f"  Token {i+1}: {accumulated_tokens[i+1]} with response_mask=True")
            bug_found = True

if not bug_found:
    print("✅ No bug found! No tokens after EOS have response_mask=True")

# ============================================================================
# Create loss_mask
# ============================================================================
print("\n" + "=" * 80)
print("CREATING LOSS_MASK")
print("=" * 80)

response_mask_tensor = torch.tensor(response_mask, dtype=torch.bool)
loss_mask = torch.roll(response_mask_tensor, shifts=-1, dims=0).float()
loss_mask[-1] = 0.0

# Check EOS positions
eos_positions = [i for i, t in enumerate(accumulated_tokens) if t == eos_token_id]
print(f"\nEOS positions: {eos_positions}")

for pos in eos_positions:
    print(f"  Position {pos}:")
    print(f"    response_mask: {response_mask[pos]}")
    print(f"    loss_mask:     {loss_mask[pos].item()}")
    if loss_mask[pos] == 1.0:
        print(f"    ⚠️  Training at EOS position!")
    else:
        print(f"    ✅ Not training at EOS position")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(
    f"""
Approach: Use vLLM tokens directly (VERL style)

Key points:
1. Generation prompt tokens come from anchor computation
2. They are added BEFORE vLLM response tokens
3. They have response_mask=False (not trainable)
4. vLLM tokens have response_mask=True (trainable)
5. No re-tokenization → no extra \\n tokens after EOS!

Result:
- Total tokens: {len(accumulated_tokens)}
- Response tokens: {sum(response_mask)}
- Matches ground truth: {accumulated_tokens == ground_truth}
- Bug (tokens after EOS): {bug_found}
"""
)
