# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Implementation of a generic generator.

A "generator" in this context refers to the orchestrator that coordinates
1) policy, 2) environments, 3) rewarders, and 4) replay buffers.

"""

from typing import Callable

from monarch.actor_mesh import Actor, ActorMeshRef, endpoint

from forge.rl.config import GeneratorConfig
from forge.rl.interfaces import (
    EnvironmentInterface,
    ForgeTrajectory,
    PolicyInterface,
    ReplayBufferInterface,
)


class Generator(Actor):
    """Generates trajectories for the training loop."""

    def __init__(
        self,
        config: GeneratorConfig,
        policy: ActorMeshRef[PolicyInterface],
        replay_buffer: ActorMeshRef[ReplayBufferInterface],
        environment_creator: Callable[[], EnvironmentInterface],
    ):
        self.config = config
        self.replay_buffer = replay_buffer
        self.environment_creator = environment_creator
        # maybe this is just the policy endpoint with a router?
        self.policy = policy
        self.environment = self.environment_creator()

    @endpoint
    async def run_episode(self):
        """Runs a single episode and writes it to the Replay buffer."""
        state, info = self.environment.reset()

        # Initialize trajectory storage
        trajectory = ForgeTrajectory()

        step = 0
        max_steps: int | None = self.config.max_generator_steps
        should_run = lambda: True if max_steps is None else step < max_steps

        while should_run():
            # Get action from policy
            action = await self.policy.generate.choose(state)

            # Store current state and action
            if trajectory.states is not None:
                trajectory.states.append(state)
            if trajectory.actions is not None:
                trajectory.actions.append(action)

            # Take step in environment
            # Note that this is the exact API that gym uses.
            (
                next_state,
                reward,
                terminated,
                truncated,
                next_info,
            ) = self.environment.step(action)

            # Store reward and done flag
            if trajectory.rewards is not None:
                trajectory.rewards.append(reward)
            if trajectory.dones is not None:
                trajectory.dones.append(terminated or truncated)
            if trajectory.infos is not None:
                trajectory.infos.append(next_info.__dict__ if next_info else {})

            # Update state for next iteration
            state = next_state

            if terminated or truncated:
                break

            step += 1

        # Write trajectory to replay buffer
        await self.replay_buffer.extend.call_one(trajectory)

        return trajectory
