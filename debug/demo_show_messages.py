# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Demo script to showcase show_messages() with multi-turn conversations.

This demonstrates the colorized token-level view that shows:
- Message structure (role, token range, trainability)
- Full message content
- Trainable vs non-trainable tokens highlighted
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from debug.token_accumulator_fn_v6 import TokenAccumulator
from vllm.transformers_utils.tokenizer import get_tokenizer


def mock_vllm_response(tokenizer, text, include_eos=True):
    """Simulate vLLM generation."""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if include_eos:
        tokens.append(tokenizer.eos_token_id)
    return tokens


def demo_multi_turn_conversation():
    """Demo: Multi-turn conversation with show_messages()"""
    print("=" * 80)
    print("MULTI-TURN CONVERSATION DEMO")
    print("=" * 80)

    tokenizer = get_tokenizer("Qwen/Qwen3-1.7B")

    acc = TokenAccumulator(
        tokenizer=tokenizer,
        messages=[{"role": "system", "content": "You are a helpful AI assistant."}],
        max_len=2048,
        eos_id=tokenizer.eos_token_id,
        thinking=False,  # Use thinking=False for this demo
    )

    print(f"\nInitial state:")
    print(f"  Tokens: {len(acc._tokens)}")
    print(f"  Budget: {acc.budget}")
    print(f"  Gen prompt length: {acc.gen_prompt_len}")
    print(f"  Suffix: {acc.suffix} (decoded: {tokenizer.decode(acc.suffix)!r})")

    # Turn 1
    print("\n" + "-" * 80)
    print("TURN 1: User asks about Python")
    print("-" * 80)

    acc.add_user("What is Python?")
    response_tokens = mock_vllm_response(
        tokenizer,
        "Python is a high-level programming language known for its simplicity.",
    )
    acc.add_assistant(
        "Python is a high-level programming language known for its simplicity.",
        response_tokens,
    )

    # Turn 2
    print("\n" + "-" * 80)
    print("TURN 2: User asks a follow-up")
    print("-" * 80)

    acc.add_user("Can you give me a simple example?")
    response_tokens = mock_vllm_response(
        tokenizer, "Sure! Here's a simple example:\n\nprint('Hello, World!')"
    )
    acc.add_assistant(
        "Sure! Here's a simple example:\n\nprint('Hello, World!')", response_tokens
    )

    # Turn 3
    print("\n" + "-" * 80)
    print("TURN 3: User says thanks")
    print("-" * 80)

    acc.add_user("Thanks!")
    response_tokens = mock_vllm_response(
        tokenizer, "You're welcome! Feel free to ask if you have more questions."
    )
    acc.add_assistant(
        "You're welcome! Feel free to ask if you have more questions.", response_tokens
    )

    # Show the complete conversation with colorized tokens
    print("\n\n")
    print("#" * 80)
    print("# SHOW_MESSAGES() OUTPUT")
    print("#" * 80)
    acc.show_messages()

    # Show final stats
    print("\n" + "=" * 80)
    print("FINAL STATISTICS")
    print("=" * 80)
    print(f"Total tokens: {len(acc._tokens)}/{acc.max_len}")
    print(f"Trainable tokens: {sum(acc._mask)}")
    print(f"Non-trainable tokens: {len(acc._mask) - sum(acc._mask)}")
    print(f"Trainable percentage: {100 * sum(acc._mask) / len(acc._mask):.1f}%")
    print(f"Truncated: {acc.truncated}")


def demo_simple_conversation():
    """Demo: Simple single-turn conversation"""
    print("\n\n")
    print("=" * 80)
    print("SIMPLE SINGLE-TURN DEMO")
    print("=" * 80)

    tokenizer = get_tokenizer("Qwen/Qwen3-1.7B")

    acc = TokenAccumulator(
        tokenizer=tokenizer,
        messages=[{"role": "system", "content": "You are helpful."}],
        max_len=2048,
        eos_id=tokenizer.eos_token_id,
        thinking=True,  # Use thinking=True for this demo
    )

    acc.add_user("What is 2+2?")
    response_tokens = mock_vllm_response(tokenizer, "The answer is 4.")
    acc.add_assistant("The answer is 4.", response_tokens)

    print("\n")
    acc.show_messages()


if __name__ == "__main__":
    demo_multi_turn_conversation()
    demo_simple_conversation()
