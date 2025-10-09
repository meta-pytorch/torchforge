# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""To run:
export HF_HUB_DISABLE_XET=1
python -m tests.sandbox.vllm.main --config tests/sandbox/vllm/llama3_8b.yaml
"""

import asyncio

import os

from forge.actors.generator import Generator
from forge.cli.config import parse

from forge.controller.provisioner import init_provisioner, shutdown

from forge.data_models.completion import Completion
from forge.observability.metric_actors import get_or_create_metric_logger
from forge.types import LauncherConfig, ProvisionerConfig
from omegaconf import DictConfig

os.environ["HYPERACTOR_MESSAGE_DELIVERY_TIMEOUT_SECS"] = "600"
os.environ["HYPERACTOR_CODE_MAX_FRAME_LENGTH"] = "1073741824"


async def run(cfg: DictConfig):
    if cfg.get("provisioner", None) is not None:
        await init_provisioner(
            ProvisionerConfig(launcher_config=LauncherConfig(**cfg.provisioner))
        )
    metric_logging_cfg = cfg.get("metric_logging", {"console": {"log_per_rank": False}})
    mlogger = await get_or_create_metric_logger()
    await mlogger.init_backends.call_one(metric_logging_cfg)

    if (prompt := cfg.get("prompt")) is None:
        gd = cfg.policy.get("sampling_config", {}).get("guided_decoding", False)
        prompt = "What is 3+5?" if gd else "Tell me a joke"

    print("Spawning service...")
    policy = await Generator.options(**cfg.services.policy).as_service(**cfg.policy)

    import time

    print("Requesting generation...")
    n = 100
    start = time.time()
    response_outputs: list[Completion] = await asyncio.gather(
        *[policy.generate.route(prompt=prompt) for _ in range(n)]
    )
    end = time.time()

    print(f"Generation of {n} requests completed in {end - start:.2f} seconds.")
    print(
        f"Generation with procs {cfg.services.policy.procs}, replicas {cfg.services.policy.num_replicas}"
    )

    print(f"\nGeneration Results (last one of {n} requests):")
    print("=" * 80)
    for batch, response in enumerate(response_outputs[-1]):
        print(f"Sample {batch + 1}:")
        print(f"User: {prompt}")
        print(f"Assistant: {response.text}")
        print("-" * 80)

    print("\nShutting down...")
    await policy.shutdown()
    await shutdown()


@parse
def recipe_main(cfg: DictConfig) -> None:
    asyncio.run(run(cfg))


if __name__ == "__main__":
    recipe_main()
