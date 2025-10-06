# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""To run:
export HF_HUB_DISABLE_XET=1
python -m apps.vllm.judge --config apps/vllm/llama3_8b.yaml
"""

# flake8: noqa

import asyncio

from forge.actors.judge import RewardModelJudge


async def run():

    prompt = "Jane has 12 apples. She gives 4 apples to her friend Mark, then buys 1 more apple, and finally splits all her apples equally among herself and her 2 siblings. How many apples does each person get?"
    response1 = "1. Jane starts with 12 apples and gives 4 to Mark. 12 - 4 = 8. Jane now has 8 apples.  2. Jane buys 1 more apple. 8 + 1 = 9. Jane now has 9 apples.  3. Jane splits the 9 apples equally among herself and her 2 siblings (3 people in total). 9 ÷ 3 = 3 apples each. Each person gets 3 apples."
    response2 = "1. Jane starts with 12 apples and gives 4 to Mark. 12 - 4 = 8. Jane now has 8 apples.  2. Jane buys 1 more apple. 8 + 1 = 9. Jane now has 9 apples.  3. Jane splits the 9 apples equally among her 2 siblings (2 people in total). 9 ÷ 2 = 4.5 apples each. Each person gets 4 apples."

    responses = [response1, response2]

    conv1 = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response1},
    ]
    conv2 = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response2},
    ]

    print(f"Prompt: {prompt}")
    print(f"Responses: {responses}\n")
    print("Evaluating responses...")

    judge = RewardModelJudge("Skywork/Skywork-Reward-V2-Llama-3.1-8B", num_labels=1)
    judge.evaluate(responses=[conv1, conv2])

    # best_response_evaluations: list[str] = await judge.evaluate.route(
    #     prompt=prompt,
    #     responses=responses,
    #     ground_truth=ground_truth,
    #     evaluation_mode=EvaluationMode.BEST_RESPONSE,
    # )

    # print("\nGeneration Results:")
    # print("=" * 80)
    # for batch, (best, fact) in enumerate(
    #     zip(best_response_evaluations, response_check_evaluations)
    # ):
    #     print(f"Sample {batch + 1}")
    #     print(f"Evaluation (BEST_RESPONSE): {best}")
    #     print(f"Evaluation (RESPONSE_CHECK): {fact}")
    #     print("-" * 80)

    # print("\nShutting down...")


def recipe_main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    recipe_main()
