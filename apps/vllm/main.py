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

import os

from forge.actors.policy import Policy
from forge.cli.config import parse
from forge.controller.provisioner import shutdown

from omegaconf import DictConfig
from src.forge.data.utils import exclude_service
from vllm.outputs import RequestOutput

os.environ["HYPERACTOR_MESSAGE_DELIVERY_TIMEOUT_SECS"] = "600"
os.environ["HYPERACTOR_CODE_MAX_FRAME_LENGTH"] = "1073741824"


async def run(cfg: DictConfig):

    if (prompt := cfg.get("prompt")) is None:
        gd = cfg.policy.get("sampling_config", {}).get("guided_decoding", False)
        prompt = "What is 3+5?" if gd else "Tell me a joke"

    print("Spawning service...")
    policy = await Policy.options(**cfg.policy.service).as_service(
        **exclude_service(cfg.policy)
    )

    try:
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

    finally:
        print("\nShutting down...")
        await policy.shutdown()
        await shutdown()

@parse
def recipe_main(cfg: DictConfig) -> None:
    asyncio.run(run(cfg))


if __name__ == "__main__":
    recipe_main()
