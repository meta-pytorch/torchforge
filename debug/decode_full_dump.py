#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Decode full messages from dump to understand why think tags are missing.
"""

import sys

sys.path.insert(0, "/home/felipemello/forge")

import torch
from vllm.transformers_utils.tokenizer import get_tokenizer


def decode_full_episode(dump_path, seq_idx=0):
    """Decode a full episode from dump."""
    print(f"\nLoading: {dump_path}")
    dump = torch.load(dump_path, map_location="cpu")
    tokenizer = get_tokenizer("Qwen/Qwen3-1.7B")  # FIX: Use correct tokenizer!

    input_ids = dump["input_ids"][seq_idx]
    loss_mask = dump["loss_mask"][seq_idx]
    targets = dump["targets"][seq_idx]

    print(f"\n{'='*80}")
    print(f"SEQUENCE {seq_idx}")
    print(f"{'='*80}")

    # Decode full sequence
    full_text = tokenizer.decode(input_ids.tolist())
    print("\nFULL DECODED TEXT:")
    print("-" * 80)
    print(full_text)
    print("-" * 80)

    # Find all assistant positions
    assistant_token = 77091
    assistant_positions = (input_ids == assistant_token).nonzero(as_tuple=True)[0]

    print(f"\nFound {len(assistant_positions)} assistant message(s)")

    # Decode each assistant message
    for idx, pos in enumerate(assistant_positions):
        pos = pos.item()
        print(f"\n{'='*80}")
        print(f"ASSISTANT MESSAGE {idx} (starts at position {pos})")
        print(f"{'='*80}")

        # Find the extent of this message (until next special token or end)
        # Look for next <|im_start|> (151644) or <|im_end|> (151645) or end
        start = pos
        end = len(input_ids)

        for i in range(pos + 1, len(input_ids)):
            if input_ids[i].item() in [151644, 151645]:
                # Found next message boundary, but include the <|im_end|> if it's there
                if input_ids[i].item() == 151645:
                    end = i + 1
                else:
                    end = i
                break

        # Decode this message
        msg_tokens = input_ids[start:end].tolist()
        msg_text = tokenizer.decode(msg_tokens)

        print(f"\nDecoded message ({end - start} tokens):")
        print("-" * 80)
        print(msg_text)
        print("-" * 80)

        # Show token breakdown
        print(f"\nToken breakdown:")
        for i in range(start, min(end, start + 30)):  # Show first 30 tokens
            tok_id = input_ids[i].item()
            tok_str = tokenizer.decode([tok_id])
            mask = loss_mask[i].item()
            tgt = targets[i].item()

            # Special markers
            marker = ""
            if tok_id == 151667:
                marker = " ← <think>"
            elif tok_id == 151668:
                marker = " ← </think>"
            elif tok_id == 151645:
                marker = " ← <|im_end|>"
            elif tok_id == 198:
                marker = " ← \\n"
            elif tok_id == 271:
                marker = " ← \\n\\n"

            trainable = "✓" if mask == 1.0 else "·"
            print(
                f"  [{i:3d}] {trainable} {tok_id:6d} {tok_str!r:20s} (tgt={tgt:6d}){marker}"
            )

        if end - start > 30:
            print(f"  ... ({end - start - 30} more tokens)")


def main():
    # Analyze both dumps, focusing on sequences that failed
    dumps = [
        ("/tmp/grpo_loss_debug_20251119_231139.pt", 0),  # First dump, seq 0
        (
            "/tmp/grpo_loss_debug_20251119_231131.pt",
            1,
        ),  # Second dump, seq 1 (61M explosion)
    ]

    for dump_path, seq_idx in dumps:
        try:
            decode_full_episode(dump_path, seq_idx)
        except Exception as e:
            print(f"\nError: {e}")
            import traceback

            traceback.print_exc()


if __name__ == "__main__":
    main()
