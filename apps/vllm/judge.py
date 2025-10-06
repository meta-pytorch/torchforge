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

from forge.actors.judge import EvaluationMode, Judge
from forge.cli.config import parse
from forge.controller.provisioner import shutdown

from forge.observability.metric_actors import get_or_create_metric_logger
from omegaconf import DictConfig

os.environ["HYPERACTOR_MESSAGE_DELIVERY_TIMEOUT_SECS"] = "600"
os.environ["HYPERACTOR_CODE_MAX_FRAME_LENGTH"] = "1073741824"


async def run(cfg: DictConfig):
    metric_logging_cfg = cfg.get("metric_logging", {"console": {"log_per_rank": False}})
    mlogger = await get_or_create_metric_logger()
    await mlogger.init_backends.call_one(metric_logging_cfg)

    prompt = "What is the capital of Japan?"
    responses = ["Aardvark", "Durian", "Tokyo"]

    print("Spawning service...")
    judge = await Judge.options(**cfg.services.policy).as_service(**cfg.policy)

    print(f"Prompt: {prompt}")
    print(f"Responses: {responses}\n")
    print("Evaluating responses...")
    best_response_evaluations: list[str] = await judge.evaluate.route(
        prompt=prompt, responses=responses, evaluation_mode=EvaluationMode.BEST_RESPONSE
    )
    response_check_evaluations: list[str] = await judge.evaluate.route(
        prompt=prompt,
        responses=responses,
        evaluation_mode=EvaluationMode.RESPONSE_CHECK,
    )

    print("\nGeneration Results:")
    print("=" * 80)
    for batch, (best, fact) in enumerate(
        zip(best_response_evaluations, response_check_evaluations)
    ):
        print(f"Sample {batch + 1}")
        print(f"Evaluation (BEST_RESPONSE): {best}")
        print(f"Evaluation (RESPONSE_CHECK): {fact}")
        print("-" * 80)

    print("\nShutting down...")
    await judge.shutdown()
    await shutdown()


@parse
def recipe_main(cfg: DictConfig) -> None:
    asyncio.run(run(cfg))


if __name__ == "__main__":
    recipe_main()
