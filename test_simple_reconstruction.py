#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Simple test: Reconstruct conversation using vLLM tokens directly.
No dummy messages needed!
"""

import asyncio
import sys

from transformers import AutoTokenizer

sys.path.insert(0, "/home/felipemello/forge")

from forge.actors.generator import Generator
from vllm.engine.arg_utils import EngineArgs
from vllm.sampling_params import SamplingParams


async def main():
    # Load tokenizer
    model_path = "Qwen/Qwen3-1.7B"
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    tokenizer.enable_thinking = (
        True  # CRITICAL: Prevent auto-wrapper in generation prompt
    )

    print(f"Model: {model_path}")
    print(f"EOS token: {tokenizer.eos_token} (id={tokenizer.eos_token_id})\n")

    # Setup generator
    engine_args = EngineArgs(
        model=model_path,
        tensor_parallel_size=1,
        max_model_len=2048,
        enable_prefix_caching=True,
    )

    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=100,
        logprobs=1,
    )

    generator = await Generator.options(
        procs=1,
        num_replicas=1,
        with_gpus=True,
    ).as_service(
        engine_args=engine_args,
        sampling_params=sampling_params,
    )

    print("✅ Generator ready\n")

    # Build conversation
    messages = [
        {
            "role": "system",
            "content": "You are an expert BlackJack player. Output only 'HIT' or 'STAND'.",
        },
        {"role": "user", "content": "Hand: 15, Dealer: 10"},
    ]

    # Generate prompt with enable_thinking=True
    prompt_text = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=True,  # No auto-wrapper!
    )

    print("=" * 80)
    print("GENERATION")
    print("=" * 80)
    print(f"\nPrompt text:\n{repr(prompt_text)}\n")

    # Generate
    completions = await generator.generate.route(
        prompt_text, sampling_params=sampling_params
    )
    completion = completions[0]

    print(f"Response text:\n{repr(completion.text)}\n")
    print(f"Stop reason: {completion.stop_reason}")

    # Get tokens
    prompt_ids = completion.prompt_ids.tolist()
    token_ids = completion.token_ids.tolist()

    print(f"\nprompt_ids length: {len(prompt_ids)}")
    print(f"token_ids length: {len(token_ids)}")

    # Check if truncated
    is_truncated = len(token_ids) > 0 and token_ids[-1] != tokenizer.eos_token_id
    print(f"Is truncated: {is_truncated}")

    print("\n" + "=" * 80)
    print("RECONSTRUCTION (Simple Approach)")
    print("=" * 80)

    # Reconstruct: prompt_ids + token_ids (+ EOS if truncated)
    if is_truncated:
        print("\n✅ Truncated response - adding EOS")
        full_conversation = prompt_ids + token_ids + [tokenizer.eos_token_id]
    else:
        print("\n✅ Complete response - EOS already included")
        full_conversation = prompt_ids + token_ids

    print(f"\nFull conversation length: {len(full_conversation)}")

    # Decode
    decoded_full = tokenizer.decode(full_conversation)
    print(f"\nDecoded conversation:\n{decoded_full}")

    # Verify
    messages_with_response = messages + [
        {"role": "assistant", "content": completion.text}
    ]
    expected_tokens = tokenizer.apply_chat_template(
        messages_with_response,
        add_generation_prompt=False,
        tokenize=True,
        enable_thinking=True,
    )

    print("\n" + "=" * 80)
    print("VERIFICATION")
    print("=" * 80)
    print(f"\nReconstructed length: {len(full_conversation)}")
    print(f"Expected length: {len(expected_tokens)}")

    if full_conversation == expected_tokens:
        print("\n✅✅✅ PERFECT MATCH!")
        print("✅ No dummy messages needed!")
        print("✅ Just use: prompt_ids + token_ids (+ EOS if truncated)")
    else:
        print("\n❌ MISMATCH")
        # Find first difference
        for i in range(min(len(full_conversation), len(expected_tokens))):
            if full_conversation[i] != expected_tokens[i]:
                print(f"\nFirst diff at position {i}:")
                print(f"  Reconstructed: {full_conversation[max(0, i-5):i+10]}")
                print(f"  Expected: {expected_tokens[max(0, i-5):i+10]}")
                break

        if len(full_conversation) != len(expected_tokens):
            print(
                f"\nLength mismatch: {abs(len(full_conversation) - len(expected_tokens))} tokens"
            )

    # Cleanup
    await generator.shutdown()
    print("\n✅ Done")


if __name__ == "__main__":
    asyncio.run(main())
