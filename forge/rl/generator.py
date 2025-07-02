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
from forge.rl.entities import ForgeTrajectory, Policy, ReplayBuffer
from forge.rl.environments import ForgeEnvironment


class Generator(Actor):
    """Generates trajectories for the training loop."""

    def __init__(
        self,
        config: GeneratorConfig,
        policy: ActorMeshRef[Policy],
        replay_buffer: ReplayBuffer,
        environment_creator: Callable[[], ForgeEnvironment],
    ):
        self.config = config
        self.replay_buffer = replay_buffer
        self.environment_creator = environment_creator
        # maybe this is just the policy endpoint with a router?
        self.policy = policy

    def initialize(self):
        """Initializes the generator."""
        self.environment = self.environment_creator()

    @endpoint
    async def run_episode(self):
        """Runs a single episode and writes it to the Replay buffer."""
        state, info = self.environment.reset()

        step = 0
        max_steps: int | None = self.config.max_generator_steps
        should_run = lambda: True if max_steps is None else step < max_steps

        while should_run():
            action = self.policy.generate.choose(state)
            observation, reward, terminated, truncated, info = self.environment.step(
                action
            )
            if terminated or truncated:
                break

            step += 1
        # trajectory creation
        trajectory = ForgeTrajectory()
        self.replay_buffer.extend.call_one(trajectory)
