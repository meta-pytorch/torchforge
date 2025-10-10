# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
This file provides examples of how LLM Inference can be used as reward calculators,
verifiers, judges, or graders in algorithms like PPO and GRPO.

Specifically this example focuses on leveraging LLM inference using `Generator`.


To run:
export HF_HUB_DISABLE_XET=1
python -m tests.sandbox.vllm.judge --config tests/sandbox/vllm/qwen3_4b.yaml
"""


import asyncio
import os

import re
from dataclasses import dataclass

from forge.actors.policy import Policy as Generator
from forge.cli.config import parse
from forge.controller.provisioner import shutdown

from forge.observability.metric_actors import get_or_create_metric_logger
from monarch.actor import endpoint
from omegaconf import DictConfig

os.environ["HYPERACTOR_MESSAGE_DELIVERY_TIMEOUT_SECS"] = "600"
os.environ["HYPERACTOR_CODE_MAX_FRAME_LENGTH"] = "1073741824"


@dataclass
class CorrectnessJudge(Generator):
    """
    Basic Judge that evaluates the correctness of a response to a prompt.
    Specifically, this judge is prompted towards math problems.

    Input is formatting based on https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507

    Note: This is not a perfect prompt, but demonstrates a basic one
    """

    @endpoint
    async def evaluate(self, prompt: str, response: str) -> list[str]:
        CORRECT = "\\boxed{CORRECT}"
        INCORRECT = "\\boxed{INCORRECT}"

        task = (
            f"You are an expert math professor. Given a prompt and response, "
            f"evaluate whether the response accurately answers the prompt. "
            f"Keep your explanations brief and don't overthink. Put your thoughts "
            f"inside <think>...</think> tags and provide your final evaluation after "
            f"the tags as {CORRECT} or {INCORRECT}.\n"
            f"---\n"
            f"# Prompt: {prompt}\n"
            f"---\n"
            f"# Response: {response}\n"
        )

        formatted = [
            {"role": "user", "content": task},
        ]
        tokenizer = self.processor.tokenizer.tokenizer
        wrapped_prompt = tokenizer.apply_chat_template(
            formatted, tokenize=False, add_generation_prompt=True
        )
        verdict: List[Completion] = await self._generate(wrapped_prompt)
        response = verdict[0].text

        # Find the last occurrence of either CORRECT or INCORRECT in the response
        # to not catch iterated verdicts while thinking
        match = None
        pattern = f"({re.escape(CORRECT)})"
        for m in re.finditer(pattern, response):
            match = m

        if match and match.group(1):
            return 1.0
        return 0.0


async def run(cfg: DictConfig):
    metric_logging_cfg = cfg.get("metric_logging", {"console": {"log_per_rank": False}})
    mlogger = await get_or_create_metric_logger()
    await mlogger.init_backends.call_one(metric_logging_cfg)

    # From https://huggingface.co/datasets/openai/gsm8k/viewer/main/train?views%5B%5D=main_train&row=0
    prompt = "Natalia sold clips to 48 of her friends in April, and then she sold half \
    as many clips in May. How many clips did Natalia sell altogether in April and May?"

    correct_response = "Natalia sold 48/2 = <<48/2=24>>24 clips in May.  Natalia sold 48+24 = \
    <<48+24=72>>72 clips altogether in April and May.  #### 72"

    # Intentionally incorrect response
    wrong_response = "Natalia sold 48/2 = <<48/2=24>>24 clips in May.  Natalia sold 48-24 = \
    <<48-24=24>>24 clips altogether in April and May.  #### 24"

    # Intentionally unrelated response
    unrelated_response = "Turtles have shells"

    print("Spawning service...")
    judge = await CorrectnessJudge.options(**cfg.services.judge).as_service(**cfg.judge)
    verdicts = await asyncio.gather(
        judge.evaluate.route(prompt, correct_response),
        judge.evaluate.route(prompt, wrong_response),
        judge.evaluate.route(prompt, unrelated_response),
    )

    print(f"Prompt: {prompt}")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    responses = [correct_response, wrong_response, unrelated_response]
    for response, verdict in zip(responses, verdicts):
        print(f"Response: {response}")
        print(f"Verdict: {verdict}")
        print("")

    print("\nShutting down...")
    await judge.shutdown()
    await shutdown()


@parse
def recipe_main(cfg: DictConfig) -> None:
    asyncio.run(run(cfg))


if __name__ == "__main__":
    recipe_main()
