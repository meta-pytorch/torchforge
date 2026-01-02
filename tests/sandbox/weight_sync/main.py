# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Weight Sync Sandbox

A minimal test environment focused exclusively on testing the weight synchronization
mechanism between RLTrainer and Generator.

Usage:
    python -m tests.sandbox.weight_sync.main --config tests/sandbox/weight_sync/qwen3_1_7b.yaml
"""

import asyncio
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
    local_batch_size = cfg.get("local_batch_size", None)
    assert local_batch_size is not None, "local_batch_size must be specified"

    request_len = cfg.get("max_req_tokens", 64)
    response_len = cfg.get("max_res_tokens", 64)
    model_name = cfg.get("model")

    print(f"Loading tokenizer for model: {model_name}")
    tokenizer = get_tokenizer(model_name)
    vocab_size = tokenizer.vocab_size
    print(f"Detected vocab size: {vocab_size}")

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

    print("=" * 80)
    print(f"Model: {model_name}")
    print(f"Local batch size: {local_batch_size}")
    print(
        f"Sequence length: {request_len + response_len} ({request_len} + {response_len})"
    )
    print(f"Data parallel size: {dp_size}")
    print(f"Is RDMA available? {rdma_enabled()}")
    print("=" * 80 + "\n")

    # Initialize trainer and generator
    print("Initializing trainer and generator...")
    init_start = time.time()

    trainer, generator = await asyncio.gather(
        RLTrainer.options(**cfg.actors.trainer).as_actor(
            **cfg.trainer,
            loss=lambda *args, **kwargs: torch.tensor(
                1.0, requires_grad=True, device="cuda"
            ),
        ),
        Generator.options(**cfg.actors.generator).as_actor(**cfg.generator),
    )

    init_time = time.time() - init_start
    print(f"Finished initialization in ({init_time:.2f}s)")

    # Run one training step to create weight delta
    print("Running single training step...")
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
    print(f"Finished train step in ({step_time:.2f}s)\n")

    # Test push_weights
    print("Pushing weights from trainer to store...")
    push_start = time.time()

    await trainer.push_weights.call(policy_version=1)

    push_time = time.time() - push_start
    print(f"Finished weights push in ({push_time:.2f}s)\n")

    # Test update_weights
    print("Updating generator weights from store...")
    update_start = time.time()

    await generator.update_weights.call(version=1)

    update_time = time.time() - update_start
    print(f"Updated generator weights ({update_time:.2f}s)\n")

    # TODO - ideally we have the capability to check forward passes between
    # the trainer/generator to verify correctness. This would require adding
    # forward capabilities to both trainer/generator actors.

    # Summary
    print("=" * 80)
    print("Results")
    print("=" * 80)
    print(f"Push time:         {push_time:.2f}s")
    print(f"Update time:       {update_time:.2f}s")
    print(f"Total sync time:   {push_time + update_time:.2f}s")
    print("=" * 80 + "\n")

    # Cleanup
    print("Shutting down...")
    await shutdown()
    print("Shutdown complete.")


if __name__ == "__main__":

    @parse
    def _main(cfg):
        asyncio.run(main(cfg))

    _main()
