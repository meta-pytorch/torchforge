# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""This is convenience script meant for hydrating the HuggingFace cache.

This is meant for downloading the model weights and tokenizer to the cache, i.e. for
OilFS.

Example:

python .meta/mast/hydrate_cache.py --model-id Qwen/Qwen3-32B

"""
import argparse
import os
import sys

from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser(
        description="Hydrate HuggingFace cache for a specific model"
    )
    parser.add_argument(
        "--model-id",
        type=str,
        required=True,
        help="HuggingFace model ID (e.g., Qwen/Qwen3-8B)",
    )
    args = parser.parse_args()

    # Ensure HF_HOME is set
    hf_home = os.environ.get("HF_HOME")
    if not hf_home:
        print(
            "ERROR: HF_HOME environment variable must be set. "
            "You will likely want to run export HF_HOME=/mnt/wsfuse/teamforge/hf."
        )
        sys.exit(1)

    print(f"Using HF_HOME: {hf_home}")
    print(f"Downloading {args.model_id}...")

    # This will pull tokenizer + config + all weight shards
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_id, trust_remote_code=True)

    print("Download complete. Cache hydrated.")


if __name__ == "__main__":
    main()
