# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""A working example showcasing a practical example of forge with RL.

Run this with:
    python -m examples.rl_main
"""

import asyncio
from dataclasses import dataclass
from functools import partial

from forge.rl.config import GeneratorConfig
from forge.rl.entities import ReplayBuffer, ToyPolicy
from forge.rl.environments import ToyEnvironment
from forge.rl.generator import Generator
from forge.rl.rewards import ToyRewarder

from monarch.proc_mesh import proc_mesh


@dataclass
class Config:
    pass


async def main():
    print("Starting RL example with toy environment and rewarder...")
    print("ToyRewarder: reward = next_state_value + 1")

    # Process allocation
    policy_procs = await proc_mesh(gpus=1)
    replay_procs = await proc_mesh(gpus=1)
    generator_procs = await proc_mesh(gpus=1)
    rewarder_procs = await proc_mesh(gpus=1)

    # Actor instantiation
    replay_buffer = await replay_procs.spawn("replay_buffer", ReplayBuffer)
    policy = await policy_procs.spawn("policy", ToyPolicy, action_range=(-2.0, 2.0))
    rewarder = await rewarder_procs.spawn("rewarder", ToyRewarder)

    generator_config = GeneratorConfig(
        max_generator_steps=5,
    )

    generator = await generator_procs.spawn(
        "generator",
        Generator,
        config=generator_config,
        policy=policy,
        replay_buffer=replay_buffer,
        environment_creator=partial(ToyEnvironment, max_steps=5, rewarder=rewarder),
    )

    print("Running a test episode...")

    # Run a single episode
    trajectory = await generator.run_episode.choose()

    print("Episode completed!")
    print(f"Trajectory length: {len(trajectory.states) if trajectory.states else 0}")
    print(f"Total reward: {sum(trajectory.rewards) if trajectory.rewards else 0}")

    # Print trajectory details to verify data flow
    print(
        "\nTrajectory Details (verifying ToyRewarder: reward = next_state_value + 1):"
    )
    if trajectory.states and trajectory.actions and trajectory.rewards:
        for i, (state, action, reward) in enumerate(
            zip(trajectory.states, trajectory.actions, trajectory.rewards)
        ):
            # Extract state value to show reward calculation
            state_value = float(state.data[0]) if state.data is not None else 0.0
            expected_next_value = state_value + (
                float(action.data[0]) if action.data is not None else 1.0
            )
            expected_reward = expected_next_value + 1.0

            print(f"Step {i}:")
            print(f"  State: {state.text}")
            print(f"  Action: {action.text}")
            print(f"  Expected next_state_value: {expected_next_value}")
            print(f"  Expected reward (next_state + 1): {expected_reward}")
            print(f"  Actual reward: {reward}")
            print(
                "  ✓ Correct!"
                if abs(reward - expected_reward) < 0.01
                else "  ✗ Mismatch!"
            )
            print()


if __name__ == "__main__":
    asyncio.run(main())
