# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Initialize, add, sample, size/len, evict, clear
Uniform sampling (without replacement)
Batching
Able to restart training from exact location
Reproducible
Proposed implementation
Python list to back the buffer
Evict based on policy_version on a Rollout (essentially just if a Rollout is too off policy, we get rid of it)
Simple collation function for batching, not exposed directly to user since only doing text data
Set RNG in sampler, exposed to user
Stateful functionality for resuming from failure
"""

import random
from typing import Any

from forge.types import Trajectory
from monarch.actor import Actor, endpoint

from torchdata.stateful_dataloader import Stateful


class ReplayBuffer(Actor, Stateful):
    """Simple in-memory replay buffer implementation."""

    def __init__(self, batch_size: int, max_policy_age: int) -> None:
        self.buffer: list[Trajectory] = []
        self.batch_size = batch_size
        self.max_policy_age = max_policy_age
        self.sampler = random.choices

    @endpoint
    async def add(self, trajectory: Trajectory) -> None:
        self.buffer.append(trajectory)

    @endpoint
    async def sample(
        self, curr_policy_version: int, batch_size: int | None
    ) -> list[Trajectory] | None:
        """Sample from the replay buffer.

        Args:
            curr_policy_version (int): The current policy version.
            batch_size (int, optional): Number of trajectories to sample. If none, defaults to batch size
                passed in at initialization.

        Returns:
            A list of sampled trajectories or None if there are not enough trajectories in the buffer.
        """
        bsz = batch_size if batch_size is not None else self.batch_size

        if bsz > len(self.buffer):
            return None

        await self.evict(curr_policy_version)

        return self.sampler(self.buffer, k=bsz)

    @endpoint
    async def evict(self, curr_policy_version: int) -> None:
        """Evict trajectories from the replay buffer if they are too old based on the current policy version
        and the max policy age allowed.

        Args:
            curr_policy_version (int): The current policy version.
        """
        self.buffer = [
            trajectory
            for trajectory in self.buffer
            if (curr_policy_version - trajectory.policy_version) < self.max_policy_age
        ]

    def len(self) -> int:
        return len(self.buffer)

    def clear(self) -> None:
        """Clear the replay buffer immediately - dropping all trajectories."""
        self.buffer.clear()

    def state_dict(self) -> dict[str, Any]:
        return {"buffer": self.buffer}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.buffer = state_dict["buffer"]
