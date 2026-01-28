# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Generate baseline golden output files using the current Generator implementation.

These golden files serve as a baseline for verifying that new implementations
produce identical outputs. Uses deterministic sampling (temperature=0) for
reproducibility.

NOTE: Golden output artifacts are checked into git. Keep the number of prompts
and MAX_TOKENS small to avoid bloating the repository. Current artifacts are
~20KB total.

Usage:
    python tests/integration_tests/generate_golden_outputs.py

The script will generate golden files in tests/integration_tests/fixtures/golden_outputs/
"""

import asyncio
from pathlib import Path

import torch
from forge.actors.generator import Generator


# Configuration - matches test_vllm_policy_correctness.py
MODEL_NAME = "facebook/opt-125m"
MAX_MODEL_LEN = 512
GPU_MEMORY_UTILIZATION = 0.1
ENFORCE_EAGER = True
ENABLE_PREFIX_CACHING = True
TENSOR_PARALLEL_SIZE = 1

# Deterministic sampling
MAX_TOKENS = 50
TEMPERATURE = 0.0
TOP_P = 1.0
N_SAMPLES = 1

TEST_PROMPTS = [
    "Hello, how are you?",
    "What is 2+2?",
    "Tell me a joke.",
    "Explain machine learning briefly.",
    "What color is the sky?",
]


async def generate_golden_outputs():
    """Generate golden outputs using the current Generator."""
    golden_dir = Path(__file__).parent / "fixtures" / "golden_outputs"
    golden_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating golden outputs to: {golden_dir}")
    print(f"Model: {MODEL_NAME}")

    generator = None
    try:
        generator = await Generator.options(
            procs=1, num_replicas=1, with_gpus=True
        ).as_service(
            engine_args={
                "model": MODEL_NAME,
                "tensor_parallel_size": TENSOR_PARALLEL_SIZE,
                "enforce_eager": ENFORCE_EAGER,
                "max_model_len": MAX_MODEL_LEN,
                "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,
                "enable_prefix_caching": ENABLE_PREFIX_CACHING,
            },
            sampling_params={
                "n": N_SAMPLES,
                "max_tokens": MAX_TOKENS,
                "temperature": TEMPERATURE,
                "top_p": TOP_P,
                "logprobs": 1,
            },
        )

        print("Generator ready. Generating outputs...\n")

        for i, prompt in enumerate(TEST_PROMPTS):
            print(f"[{i + 1}/{len(TEST_PROMPTS)}] Prompt: {prompt[:50]}...")

            result = await generator.generate.route(prompt)
            completion = result[0]

            # Serialize entire Completion object
            golden_path = golden_dir / f"completion_{i}.pt"
            torch.save(completion, golden_path)
            print(f"    Saved: {golden_path}")

        metadata = {
            "model": MODEL_NAME,
            "max_tokens": MAX_TOKENS,
            "temperature": TEMPERATURE,
            "prompts": TEST_PROMPTS,
        }
        torch.save(metadata, golden_dir / "metadata.pt")

        print("\nGolden output generation complete!")

    finally:
        if generator is not None:
            await generator.shutdown()


if __name__ == "__main__":
    asyncio.run(generate_golden_outputs())
