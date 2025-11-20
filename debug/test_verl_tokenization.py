#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Test to understand how VERL handles tokens after EOS in apply_chat_template.
"""

import sys

sys.path.insert(0, "/home/felipemello/forge")

from vllm.transformers_utils.tokenizer import get_tokenizer

# Get Qwen tokenizer
tokenizer = get_tokenizer("Qwen/Qwen2.5-0.5B-Instruct")
eos_token_id = tokenizer.eos_token_id

print("=" * 80)
print("Testing VERL's Delta Tokenization Approach")
print("=" * 80)

# Base chat history (like VERL)
BASE_CHAT_HISTORY = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "I am a user."},
]

# Calculate base lengths
base_wo_gen = tokenizer.apply_chat_template(
    BASE_CHAT_HISTORY,
    add_generation_prompt=False,
    tokenize=True,
)
base_with_gen = tokenizer.apply_chat_template(
    BASE_CHAT_HISTORY,
    add_generation_prompt=True,
    tokenize=True,
)

print(f"\nBase lengths:")
print(f"  Without generation prompt: {len(base_wo_gen)}")
print(f"  With generation prompt:    {len(base_with_gen)}")
print(f"  Generation prompt length:  {len(base_with_gen) - len(base_wo_gen)}")

# Now add an assistant message
assistant_message = {"role": "assistant", "content": "Hello world"}

# VERL approach: tokenize [BASE_CHAT_HISTORY, assistant_message]
messages_with_assistant = [*BASE_CHAT_HISTORY, assistant_message]

full_with_assistant = tokenizer.apply_chat_template(
    messages_with_assistant,
    add_generation_prompt=False,
    tokenize=True,
)

# Extract delta (what VERL does)
# They slice from base_with_gen_len
delta_tokens = full_with_assistant[len(base_with_gen) :]

print(f"\nFull conversation with assistant:")
print(f"  Total length: {len(full_with_assistant)}")
print(f"  Delta tokens (from base_with_gen): {len(delta_tokens)}")

# Decode the delta
delta_text = tokenizer.decode(delta_tokens)
print(f"\nDelta decoded:")
print(f"  Text: {repr(delta_text)}")
print(f"  Tokens: {delta_tokens}")

# Check if EOS is in delta
if eos_token_id in delta_tokens:
    eos_idx = delta_tokens.index(eos_token_id)
    print(f"\nEOS found at position {eos_idx} in delta")
    print(f"  Tokens before EOS: {delta_tokens[:eos_idx]}")
    print(f"  EOS token: {delta_tokens[eos_idx]}")
    print(f"  Tokens after EOS: {delta_tokens[eos_idx+1:]}")

    if len(delta_tokens) > eos_idx + 1:
        after_eos_text = tokenizer.decode(delta_tokens[eos_idx + 1 :])
        print(f"  Decoded after EOS: {repr(after_eos_text)}")
else:
    print(f"\n⚠️  No EOS in delta tokens!")

# Now let's see what happens if we manually append EOS (like vLLM does)
print("\n" + "=" * 80)
print("Simulating vLLM Generation (with EOS)")
print("=" * 80)

# Simulate vLLM: returns tokens WITHOUT chat template suffix
vllm_tokens = tokenizer.encode("Hello world", add_special_tokens=False) + [eos_token_id]
print(f"\nvLLM tokens (content + EOS): {vllm_tokens}")
print(f"  Decoded: {repr(tokenizer.decode(vllm_tokens))}")

# Now when VERL adds this to conversation, what happens?
# They pass content_ids directly sometimes
print("\n" + "=" * 80)
print("VERL Approach 1: Using content_ids from vLLM")
print("=" * 80)

# When they have content_ids from vLLM, they just use them directly
# (see line 399-412 in schemas.py)
print(f"  content_ids from vLLM: {vllm_tokens}")
print(f"  These get added with loss_mask=True")
print(f"  Length: {len(vllm_tokens)}")

# Check if there's a newline after EOS
if len(vllm_tokens) > 0 and vllm_tokens[-1] == eos_token_id:
    print(f"  ✓ Last token is EOS")
else:
    print(f"  ✗ Last token is NOT EOS: {vllm_tokens[-1]}")

print("\n" + "=" * 80)
print("VERL Approach 2: Re-tokenizing with chat template")
print("=" * 80)

# If they don't have content_ids, they re-tokenize
# Let's see what happens
messages_for_retokenize = [
    *BASE_CHAT_HISTORY,
    {"role": "assistant", "content": "Hello world"},
]
full_retokenize = tokenizer.apply_chat_template(
    messages_for_retokenize,
    add_generation_prompt=False,
    tokenize=True,
)

delta_retokenize = full_retokenize[len(base_with_gen) :]
print(f"  Delta from re-tokenization: {delta_retokenize}")
print(f"  Length: {len(delta_retokenize)}")

# Compare with vLLM tokens
print(f"\n  Comparison:")
print(f"    vLLM tokens:        {vllm_tokens}")
print(f"    Re-tokenized delta: {delta_retokenize}")
print(f"    Match: {vllm_tokens == delta_retokenize}")

if vllm_tokens != delta_retokenize:
    print(f"\n  ⚠️  MISMATCH!")
    print(f"    Extra in delta: {delta_retokenize[len(vllm_tokens):]}")
    if len(delta_retokenize) > len(vllm_tokens):
        extra_text = tokenizer.decode(delta_retokenize[len(vllm_tokens) :])
        print(f"    Decoded extra: {repr(extra_text)}")

print("\n" + "=" * 80)
print("Conclusion")
print("=" * 80)

print(
    """
Key findings:
1. When VERL uses content_ids from vLLM directly, they get exactly what was generated
2. When VERL re-tokenizes with apply_chat_template, the chat template MAY add extra tokens
3. The delta approach slices from base_with_gen_prompt_end_pos, which EXCLUDES generation
   prompt but INCLUDES any suffix the chat template adds

VERL's solution:
- They primarily use content_ids from the generation engine (vLLM/SGLang)
- Only re-tokenize when content_ids is None
- When they do re-tokenize, they accept whatever the chat template produces
- Then use get_response_mask() to mask tokens after EOS

Our bug:
- We're re-tokenizing with apply_chat_template (delta approach)
- Chat template adds \\n after EOS
- We mark it as response_mask=True
- Then we train at EOS position (predicting the \\n)

Fix options:
1. Use vLLM tokens directly (don't re-tokenize) - like VERL approach 1
2. Strip after EOS when re-tokenizing - explicit fix
3. Mask EOS positions in loss_mask - defensive fix
"""
)
