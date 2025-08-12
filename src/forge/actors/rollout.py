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
from forge.types import State

from monarch.actor import Actor, endpoint


class RolloutActor(Actor):
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
    async def run_episode(self) -> list[State]:
        """Runs a single episode and writes it to the Replay buffer."""
        state = self.environment.reset()

        # Keep track of all states just for reporting purposes
        all_states = []
        while True:
            state = await self.policy.generate.choose(state)
            state = self.environment.step(state)
            if state.truncated or state.terminated:
                # Rewards?
                # Transforms?
                all_states.append(state)
                await self.replay_buffer.add.call_one(state)
                break
        return all_states
