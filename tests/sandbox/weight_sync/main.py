# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Weight Sync Sandbox

A minimal test environment focused exclusively on testing the weight synchronization
mechanism between RLTrainer and Generator.

This sandbox:
- Initializes both trainer and generator with the same model
- Runs ONE training step to create a weight delta
- Tests push_weights() performance and correctness
- Tests update_weights() performance and correctness
- Verifies the sync with a forward pass

Usage:
    python -m tests.sandbox.weight_sync.main --config tests/sandbox/weight_sync/qwen3_1_7b.yaml
"""

import asyncio
import os
import time

import torch
import torchstore as ts
from forge.actors._torchstore_utils import rdma_enabled
from forge.actors.generator import Generator
from forge.actors.trainer import RLTrainer
from forge.controller.provisioner import init_provisioner, shutdown
from forge.observability.metric_actors import get_or_create_metric_logger
from forge.types import LauncherConfig, ProvisionerConfig
from forge.util.config import parse
from omegaconf import DictConfig
from vllm.transformers_utils.tokenizer import get_tokenizer

# Suppress resource_tracker warnings about shared memory cleanup
# These occur because shared memory is cleaned up by one process while the
# resource tracker in another process tries to clean it up again. The cleanup
# is working correctly - these are just harmless race condition warnings.
os.environ["PYTHONWARNINGS"] = "ignore::UserWarning:multiprocessing.resource_tracker"


def simple_grpo_loss(
    logits: torch.Tensor,
    response: torch.Tensor,
    ref_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    padding_mask: torch.Tensor,
    beta: float = 0.1,
) -> torch.Tensor:
    """
    Simplified loss function for weight sync testing.
    Just performs basic tensor operations to create a weight delta.
    """
    # Extract dimensions
    local_batch_size, response_len = response.shape
    vocab_size = logits.size(-1)
    full_seq_len = logits.size(1)

    # Extract only the response portion from logits
    request_len = full_seq_len - response_len
    response_logits = logits[:, request_len:, :]

    # Flatten logits and response for cross-entropy
    logits_flat = response_logits.reshape(-1, vocab_size)
    response_flat = response.reshape(-1)

    # Basic cross-entropy loss
    loss = torch.nn.functional.cross_entropy(
        logits_flat, response_flat, reduction="none"
    ).view(local_batch_size, response_len)

    # Apply padding mask and reduce
    masked_loss = loss * padding_mask
    loss = masked_loss.sum() / padding_mask.sum().clamp(min=1.0)

    return loss


def generate_random_batch(
    local_batch_size: int,
    request_len: int,
    response_len: int,
    vocab_size: int = 32000,
    device: str = "cuda",
    dp_size: int = 1,
):
    """
    Generate random input and target tensors for a single training step.
    Creates one batch per data parallel rank.
    """
    inputs = []
    targets = []

    # Create one batch for each data parallel rank
    for _ in range(dp_size):
        request = torch.randint(
            1,
            vocab_size,
            (local_batch_size, request_len),
            dtype=torch.long,
            device=device,
        )
        response = torch.randint(
            1,
            vocab_size,
            (local_batch_size, response_len),
            dtype=torch.long,
            device=device,
        )

        # Create padding mask
        padding_mask = torch.rand((local_batch_size, response_len), device=device) > 0.1

        ref_logprobs = (
            -torch.abs(torch.randn((local_batch_size, response_len), device=device))
            - 1.0
        )
        advantages = torch.randn((local_batch_size, 1), device=device)
        input_tokens = torch.cat([request, response], dim=1)
        inputs.append({"tokens": input_tokens})
        targets.append(
            {
                "response": response,
                "ref_logprobs": ref_logprobs,
                "advantages": advantages,
                "padding_mask": padding_mask,
            }
        )

    return inputs, targets


async def main(cfg: DictConfig):
    """
    Weight sync sandbox main function.

    Tests the complete weight synchronization pipeline:
    1. Initialize trainer and generator
    2. Run one training step
    3. Push weights from trainer
    4. Update weights in generator
    5. Verify with forward pass
    """

    # Extract configuration
    local_batch_size = cfg.get("local_batch_size", None)
    assert local_batch_size is not None, "local_batch_size must be specified"

    request_len = cfg.get("max_req_tokens", 64)
    response_len = cfg.get("max_res_tokens", 64)
    model_name = cfg.get("model")

    # Get vocab size from tokenizer
    print(f"Loading tokenizer for model: {model_name}")
    tokenizer = get_tokenizer(model_name)
    vocab_size = tokenizer.vocab_size
    print(f"Detected vocab size: {vocab_size}")

    # Get data parallel size
    dp_size = cfg.get("replay_buffer", {}).get("dp_size", 1)
    if dp_size is None:
        trainer_dp_degree = cfg.trainer.parallelism.get("data_parallel_shard_degree", 1)
        dp_size = trainer_dp_degree if trainer_dp_degree != -1 else 1

    # ---- Global setups ---- #
    provisioner = None
    if cfg.get("provisioner", None) is not None:
        provisioner = await init_provisioner(
            ProvisionerConfig(launcher_config=LauncherConfig(**cfg.provisioner))
        )
    else:
        provisioner = await init_provisioner()

    metric_logging_cfg = cfg.get("metric_logging", {})
    mlogger = await get_or_create_metric_logger(process_name="Controller")
    await mlogger.init_backends.call_one(metric_logging_cfg)

    # Initialize torchstore
    await ts.initialize(strategy=ts.ControllerStorageVolumes())

    print("\n" + "=" * 80)
    print("WEIGHT SYNC SANDBOX")
    print("=" * 80)
    print(f"Model: {model_name}")
    print(f"Local batch size: {local_batch_size}")
    print(
        f"Sequence length: {request_len + response_len} ({request_len} + {response_len})"
    )
    print(f"Data parallel size: {dp_size}")
    print(f"RDMA available: {rdma_enabled()}")
    print(f"Sync mode: {'Direct (RDMA)' if rdma_enabled() else 'DCP (Filesystem)'}")
    print("=" * 80 + "\n")

    # Initialize trainer and generator
    print("Initializing trainer and generator...")
    init_start = time.time()

    trainer, policy = await asyncio.gather(
        RLTrainer.options(**cfg.actors.trainer).as_actor(
            **cfg.trainer, loss=simple_grpo_loss
        ),
        Generator.options(**cfg.actors.policy).as_actor(**cfg.policy),
    )

    init_time = time.time() - init_start
    print(f"✓ Initialization complete ({init_time:.2f}s)\n")

    # Run one training step to create weight delta
    print("[1/4] Running single training step to create weight delta...")
    step_start = time.time()

    inputs, targets = generate_random_batch(
        local_batch_size=local_batch_size,
        request_len=request_len,
        response_len=response_len,
        vocab_size=vocab_size,
        dp_size=dp_size,
    )

    await trainer.train_step.call(inputs, targets)
    step_time = time.time() - step_start
    print(f"✓ Training step complete ({step_time:.2f}s)\n")

    # Test push_weights
    print("[2/4] Testing push_weights() to torchstore...")
    push_start = time.time()

    await trainer.push_weights.call(policy_version=1)

    push_time = time.time() - push_start
    print(f"✓ Pushed weights to torchstore ({push_time:.2f}s)\n")

    # Test update_weights
    print("[3/4] Testing update_weights() from torchstore...")
    update_start = time.time()

    await policy.update_weights.call(version=1)

    update_time = time.time() - update_start
    print(f"✓ Updated weights in generator ({update_time:.2f}s)\n")

    # Verify with forward pass
    print("[4/4] Verification: Running forward pass with updated weights...")
    verify_start = time.time()

    test_prompt = "Write a short poem"
    result = await policy.generate.call(prompt=test_prompt)
    # Unwrap the ValueMesh
    _, result = next(result.items())

    verify_time = time.time() - verify_start
    print(f"✓ Forward pass successful ({verify_time:.2f}s)")
    print("\nSample output:")
    print(f"Prompt: {test_prompt}")
    print(f"Response: {result[0].text[:100]}...\n")

    # Summary
    print("=" * 80)
    print("WEIGHT SYNC TEST COMPLETE")
    print("=" * 80)
    print(f"Push time:         {push_time:.2f}s")
    print(f"Update time:       {update_time:.2f}s")
    print(f"Total sync time:   {push_time + update_time:.2f}s")
    print(
        f"Sync mode used:    {'Direct (RDMA)' if rdma_enabled() else 'DCP (Filesystem)'}"
    )
    print("=" * 80 + "\n")

    # Cleanup
    print("Shutting down...")
    await shutdown()
    print("✓ Shutdown complete.")


if __name__ == "__main__":

    @parse
    def _main(cfg):
        asyncio.run(main(cfg))

    _main()
