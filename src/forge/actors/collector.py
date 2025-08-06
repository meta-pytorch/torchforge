# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Implementation of a generic collector.

A "collector" in this context refers to the orchestrator that coordinates
1) policy, 2) environments, 3) rewarders, and 4) replay buffers.

"""

from typing import Callable

from forge.data.replay_buffer import ReplayBuffer

from forge.interfaces import Policy

from forge.types import Trajectory

from monarch.actor import Actor, endpoint


class Collector(Actor):
    """Collects trajectories for the training loop."""

    def __init__(
        self,
        max_collector_steps: int,
        policy: Policy,
        replay_buffer: ReplayBuffer,
        environment_creator: Callable,
    ):
        self.max_collector_steps = max_collector_steps
        self.replay_buffer = replay_buffer
        self.environment_creator = environment_creator
        # maybe this is just the policy endpoint with a router?
        self.policy = policy
        self.environment = self.environment_creator()

    @endpoint
    async def run_episode(self) -> Trajectory:
        """Runs a single episode and writes it to the Replay buffer."""
        state = self.environment.reset()

        # Initialize trajectory storage
        trajectory = Trajectory()

        step = 0
        max_steps = self.max_collector_steps
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
            state = self.environment.step(action)
            step += 1

        # Write trajectory to replay buffer
        await self.replay_buffer.extend.call_one(trajectory)

        return trajectory
