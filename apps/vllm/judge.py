# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""To run:
export HF_HUB_DISABLE_XET=1
python -m apps.vllm.judge --config apps/vllm/llama3_8b.yaml
"""

import asyncio

import os

from forge.actors.generative_judge import LLMJudge
from forge.cli.config import parse
from forge.controller.provisioner import shutdown

from omegaconf import DictConfig

os.environ["HYPERACTOR_MESSAGE_DELIVERY_TIMEOUT_SECS"] = "600"
os.environ["HYPERACTOR_CODE_MAX_FRAME_LENGTH"] = "1073741824"


async def run(cfg: DictConfig):
    prompt = "What is the capital of Japan?"
    responses = ["Aardvark", "Durian", "Tokyo"]

    print("Spawning service...")
    judge_service_config = cfg.services.pop("judge")
    judge = await LLMJudge.options(**judge_service_config).as_service(policy_cfg=cfg)

    print(f"Prompt: {prompt}")
    print(f"Responses: {responses}\n")
    print("Requesting generation ...")
    evaluations: list[str] = await judge.generate.choose(
        prompt=prompt,
        responses=responses,
    )

    print("\nGeneration Results:")
    print("=" * 80)
    for batch, evaluation in enumerate(evaluations):
        print(f"Sample {batch + 1}")
        print(f"Evaluation: {evaluation}")
        print("-" * 80)

    print("\nShutting down...")
    await judge.shutdown()
    await shutdown()


@parse
def recipe_main(cfg: DictConfig) -> None:
    asyncio.run(run(cfg))


if __name__ == "__main__":
    recipe_main()
