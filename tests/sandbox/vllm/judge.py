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

# from forge.actors.judge import EvaluationMode, Judge
from forge.actors.policy import Policy
from forge.cli.config import parse
from forge.controller.provisioner import shutdown

from forge.observability.metric_actors import get_or_create_metric_logger
from monarch.actor import endpoint
from omegaconf import DictConfig

os.environ["HYPERACTOR_MESSAGE_DELIVERY_TIMEOUT_SECS"] = "600"
os.environ["HYPERACTOR_CODE_MAX_FRAME_LENGTH"] = "1073741824"


async def run(cfg: DictConfig):
    metric_logging_cfg = cfg.get("metric_logging", {"console": {"log_per_rank": False}})
    mlogger = await get_or_create_metric_logger()
    await mlogger.init_backends.call_one(metric_logging_cfg)

    # prompt = "What is the capital of Japan?"
    # responses = ["Aardvark", "Durian", "Tokyo"]

    # From https://huggingface.co/Skywork/Skywork-Reward-V2-Llama-3.1-8B
    # prompt = "Jane has 12 apples. She gives 4 apples to her friend Mark, \
    # then buys 1 more apple, and finally splits all her apples equally among \
    # herself and her 2 siblings. How many apples does each person get?"
    # response1 = "1. Jane starts with 12 apples and gives 4 to Mark. 12 - 4 = 8. \
    # Jane now has 8 apples.  2. Jane buys 1 more apple. 8 + 1 = 9. Jane now has 9 \
    # apples.  3. Jane splits the 9 apples equally among herself and her 2 siblings \
    # (3 people in total). 9 ÷ 3 = 3 apples each. Each person gets 3 apples."
    # response2 = "1. Jane starts with 12 apples and gives 4 to Mark. 12 - 4 = 8. \
    # Jane now has 8 apples.  2. Jane buys 1 more apple. 8 + 1 = 9. Jane now has 9 \
    # apples.  3. Jane splits the 9 apples equally among her 2 siblings (2 people in \
    # total). 9 ÷ 2 = 4.5 apples each. Each person gets 4 apples."

    # From https://huggingface.co/datasets/openai/gsm8k/viewer/main/train?views%5B%5D=main_train&row=0
    prompt = "Natalia sold clips to 48 of her friends in April, and then she sold half \
    as many clips in May. How many clips did Natalia sell altogether in April and May?"

    response1 = "Natalia sold 48/2 = <<48/2=24>>24 clips in May.  Natalia sold 48+24 = \
    <<48+24=72>>72 clips altogether in April and May.  #### 72"

    # Intentionally incorrect response
    response2 = "Natalia sold 48/2 = <<48/2=24>>24 clips in May.  Natalia sold 48-24 = \
    <<48-24=24>>24 clips altogether in April and May.  #### 24"

    class Judge2(Policy):
        @endpoint
        async def evaluate(self, prompt: str, response: str) -> list[str]:
            system_prompt = f""" As an expert mathematician, and given the following question:
                    {prompt}
                     Answer with a Positive or Negative. Evaluate if the following response is correct"""
            as_chat = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": response},
            ]
            tokenizer = self.processor.tokenizer.tokenizer
            wrapped_prompt = tokenizer.apply_chat_template(
                as_chat, tokenize=False, add_generation_prompt=True
            )

            # wrapped_prompt = self._math_check(prompt, response)
            # wrapped_prompt = self._response_check(prompt, response)
            verdict: List[Completion] = await self._generate(wrapped_prompt)
            print(verdict[0].text)
            return 1.0 if "Positive" in verdict[0].text else 0.0

    print("Spawning service...")
    # judge = await Judge.options(**cfg.services.policy).as_service(**cfg.policy)
    judge2 = await Judge2.options(**cfg.services.judge).as_service(**cfg.judge)
    print("resp 1", await judge2.evaluate.route(prompt, response1))
    print("resp 2", await judge2.evaluate.route(prompt, response2))

    print("\nShutting down...")
    await judge2.shutdown()
    await shutdown()


@parse
def recipe_main(cfg: DictConfig) -> None:
    asyncio.run(run(cfg))


if __name__ == "__main__":
    recipe_main()
