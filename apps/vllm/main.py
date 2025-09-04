# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""To run:
export HF_HUB_DISABLE_XET=1
python -m apps.vllm.main --config apps/vllm/llama3_8b.yaml
"""

import asyncio
import sys

from forge.actors.policy import Policy
from forge.cli.config import parse
from forge.controller.service import ServiceConfig, shutdown_service, spawn_service

from omegaconf import DictConfig
from vllm.outputs import RequestOutput

async def run(cfg: DictConfig):

    if "prompt" in cfg and cfg["prompt"] is not None:
        prompt = cfg["prompt"]
    else:
        gd = cfg.policy.get("sampling_overrides", {}).get("guided_decoding", False)
        prompt = "What is 3+5?" if gd else "Tell me a joke"

    print("Spawning service...")

    policy = await spawn_service(
        ServiceConfig(**cfg.service_config), Policy, **cfg.policy
    )

    async with policy.session():
        print("Requesting generation...")
        response_output: RequestOutput = await policy.generate.choose(prompt=prompt)

        print("\nGeneration Results:")
        print("=" * 80)
        for batch, response in enumerate(response_output.outputs):
            print(f"Sample {batch + 1}:")
            print(f"User: {prompt}")
            print(f"Assistant: {response.text}")
            print("-" * 80)

        print("\nShutting down...")

    await shutdown_service(policy)


@parse
def recipe_main(cfg: DictConfig) -> None:
    asyncio.run(run(cfg))


if __name__ == "__main__":
    sys.exit(recipe_main())
