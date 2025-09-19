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

from typing import Callable

from forge.actors.generative_judge import GenerativeJudge
from forge.actors.policy import Policy
from forge.cli.config import parse
from forge.controller.provisioner import shutdown

from omegaconf import DictConfig
from vllm.outputs import RequestOutput

os.environ["HYPERACTOR_MESSAGE_DELIVERY_TIMEOUT_SECS"] = "600"
os.environ["HYPERACTOR_CODE_MAX_FRAME_LENGTH"] = "1073741824"

from vllm.transformers_utils.tokenizer import get_tokenizer


def basic_selector_wrapper(model: str) -> Callable[[str, list[str]], str]:
    """
    Note: This is not a "good" prompt setup, it just demonstrates how to make one
    """

    def _wrapper(prompt: str, responses: list[str]) -> str:
        tokenizer = get_tokenizer(model)
        system_prompt = f"""
        You are an expert evaluator. Evaluate the responses provided and return
        a single integer indicating which response is the most factually correct.
        Each response is formatted as [Response #<N>], where <N> represents the
        selection. Do not explain your reasoning, just provide a number.

        Here is the prompt that generated the responses: {prompt}.
        """
        response_str = "\n".join(
            [f"[Response #{i+1}] {resp}" for i, resp in enumerate(responses)]
        )
        as_chat = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": response_str},
        ]
        formatted_request = tokenizer.apply_chat_template(
            as_chat, tokenize=False, add_generation_prompt=True
        )
        return formatted_request

    return _wrapper


def unroll_response(response: RequestOutput) -> list[str]:
    return [output.text for output in response.outputs]


async def run(cfg: DictConfig):
    prompt = "What is the capital of Japan?"
    responses = ["Aardvark", "Durian", "Tokyo"]

    print("Spawning service...")
    policy = await Policy.options(**cfg.services.policy).as_service(**cfg.policy)
    evaluate = GenerativeJudge(
        policy,
        prompt_wrapper=basic_selector_wrapper(cfg.policy.engine_config.model),
        output_postprocessor=unroll_response,
    )

    print(f"Prompt: {prompt}")
    print(f"Responses: {responses}\n")

    try:
        async with policy.session():
            print("Requesting generation ...")
            evaluations = await evaluate.generate(
                prompt=prompt,
                responses=responses,
            )

            print("\nGeneration Results:")
            print("=" * 80)
            for batch, evaluation in enumerate(evaluations):
                print(f"Sample {batch + 1}")
                print(f"Evaluation: {evaluation}")
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
