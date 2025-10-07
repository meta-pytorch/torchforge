# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""To run:
export HF_HUB_DISABLE_XET=1 python -m apps.vllm.rewardjudge
"""

from forge.actors.judge import RewardModelJudge


def run():
    # metric_logging_cfg = cfg.get("metric_logging", {"console": {"log_per_rank": False}})
    # mlogger = await get_or_create_metric_logger()
    # await mlogger.init_backends.call_one(metric_logging_cfg)

    prompt = "Jane has 12 apples. She gives 4 apples to her friend Mark, \
    then buys 1 more apple, and finally splits all her apples equally among \
    herself and her 2 siblings. How many apples does each person get?"
    response1 = "1. Jane starts with 12 apples and gives 4 to Mark. 12 - 4 = 8. \
    Jane now has 8 apples.  2. Jane buys 1 more apple. 8 + 1 = 9. Jane now has 9 \
    apples.  3. Jane splits the 9 apples equally among herself and her 2 siblings \
    (3 people in total). 9 ÷ 3 = 3 apples each. Each person gets 3 apples."
    response2 = "1. Jane starts with 12 apples and gives 4 to Mark. 12 - 4 = 8. \
    Jane now has 8 apples.  2. Jane buys 1 more apple. 8 + 1 = 9. Jane now has 9 \
    apples.  3. Jane splits the 9 apples equally among her 2 siblings (2 people in \
    total). 9 ÷ 2 = 4.5 apples each. Each person gets 4 apples."

    responses = [response1, response2]

    print(f"Prompt: {prompt}")
    print(f"Responses: {responses}\n")
    print("Evaluating responses...")

    model = "Skywork/Skywork-Reward-V2-Llama-3.1-8B"
    judge = RewardModelJudge(model)
    scores = judge.evaluate(prompt=prompt, responses=responses)

    print("\nGeneration Results:")
    print("=" * 80)
    for batch, (response, score) in enumerate(zip(responses, scores)):
        print(f"Sample {batch + 1}")
        print(f"Response: {response}")
        print(f"Score: {score}")
        print("-" * 80)


if __name__ == "__main__":
    run()
