# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import random
from dataclasses import dataclass
from typing import Any

from monarch.actor import endpoint

from forge.controller import ForgeActor
from forge.types import Trajectory


@dataclass
class ReplayBuffer(ForgeActor):
    """Simple in-memory replay buffer implementation."""

    batch_size: int
    max_policy_age: int
    seed: int | None = None

    @endpoint
    async def setup(self) -> None:
        self.buffer: list[Trajectory] = []
        if self.seed is None:
            self.seed = random.randint(0, 2**32)
        random.seed(self.seed)
        self.sampler = random.sample

    @endpoint
    async def add(self, trajectory: Trajectory) -> None:
        self.buffer.append(trajectory)

    @endpoint
    async def sample(
        self, curr_policy_version: int, batch_size: int | None = None
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

        # Evict old trajectories
        self._evict(curr_policy_version)

        if bsz > len(self.buffer):
            print("Not enough trajectories in the buffer.")
            return None

        # TODO: Make this more efficient
        idx_to_sample = self.sampler(range(len(self.buffer)), k=bsz)
        sorted_idxs = sorted(
            idx_to_sample, reverse=True
        )  # Sort in desc order to avoid shifting idxs
        sampled_trajectories = [self.buffer.pop(i) for i in sorted_idxs]
        return sampled_trajectories

    @endpoint
    async def evict(self, curr_policy_version: int) -> None:
        """Evict trajectories from the replay buffer if they are too old based on the current policy version
        and the max policy age allowed.

        Args:
            curr_policy_version (int): The current policy version.
        """
        self._evict(curr_policy_version)

    def _evict(self, curr_policy_version: int) -> None:
        self.buffer = [
            trajectory
            for trajectory in self.buffer
            if (curr_policy_version - trajectory.policy_version) <= self.max_policy_age
        ]

    @endpoint
    async def _getitem(self, idx: int) -> Trajectory:
        return self.buffer[idx]

    @endpoint
    async def _numel(self) -> int:
        """Number of elements (trajectories) in the replay buffer."""
        return len(self.buffer)

    @endpoint
    async def clear(self) -> None:
        """Clear the replay buffer immediately - dropping all trajectories."""
        self.buffer.clear()

    @endpoint
    async def state_dict(self) -> dict[str, Any]:
        return {
            "buffer": self.buffer,
            "rng_state": random.getstate(),
            "seed": self.seed,
        }

    @endpoint
    async def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.buffer = state_dict["buffer"]
        random.setstate(state_dict["rng_state"])
