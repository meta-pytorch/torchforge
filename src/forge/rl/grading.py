# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Callable

from forge.controller.actor import ForgeActor
from forge.observability.metrics import record_metric, Reduce
from monarch.actor import endpoint


@dataclass
class RewardActor(ForgeActor):
    reward_functions: list[Callable]

    @endpoint
    async def evaluate_response(
        self, prompt: str, response: str, target: str
    ) -> (dict[str, float], float):
        total_rewards = 0.0
        reward_breakdown = {}  # reward breakdown by function
        for reward_fn in self.reward_functions:
            reward = reward_fn(prompt, response, target)
            total_rewards += reward

            # Get a name for the reward function (works for classes, functions, lambdas)
            reward_fn_name = getattr(
                reward_fn, "__name__", reward_fn.__class__.__name__
            )
            reward_breakdown[reward_fn_name] = reward

            # log per fn reward and avg total
            record_metric(
                f"reward/evaluate_response/avg_{reward_fn_name}_reward",
                reward,
                Reduce.MEAN,
            )
            record_metric(
                f"reward/evaluate_response/std_{reward_fn_name}_reward",
                reward,
                Reduce.STD,
            )

            record_metric(
                "reward/evaluate_response/avg_total_reward",
                reward,
                Reduce.MEAN,
            )

        avg_reward: float = total_rewards / len(self.reward_functions)
        return reward_breakdown, avg_reward
