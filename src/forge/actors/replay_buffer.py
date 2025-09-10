# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import random
from dataclasses import dataclass
from typing import Any

from forge.controller import ForgeActor

from monarch.actor import endpoint


@dataclass
class ReplayBuffer(ForgeActor):
    """Simple in-memory replay buffer implementation."""

    batch_size: int
    max_policy_age: int
    dp_size: int = 1
    seed: int | None = None

    @endpoint
    async def setup(self) -> None:
        self.buffer: list = []
        if self.seed is None:
            self.seed = random.randint(0, 2**32)
        random.seed(self.seed)
        self.sampler = random.sample

    @endpoint
    async def add(self, episode) -> None:
        self.buffer.append(episode)

    @endpoint
    async def sample(self, curr_policy_version: int, batch_size: int | None = None):
        """Sample from the replay buffer.

        Args:
            curr_policy_version (int): The current policy version.
            batch_size (int, optional): Number of episodes to sample. If none, defaults to batch size
                passed in at initialization.

        Returns:
            A list of sampled episodes with shape (dp_size, bsz, ...) or None if there are not enough episodes in the buffer.
        """
        bsz = batch_size if batch_size is not None else self.batch_size
        total_samples = self.dp_size * bsz

        # Evict old episodes
        self._evict(curr_policy_version)

        if total_samples > len(self.buffer):
            return None

        # TODO: Make this more efficient
        idx_to_sample = self.sampler(range(len(self.buffer)), k=total_samples)
        sampled_episodes = [self.buffer[i] for i in idx_to_sample]

        # Evict sampled episodes (descending order so pops are safe)
        for i in sorted(idx_to_sample, reverse=True):
            self.buffer.pop(i)

        # Reshape into (dp_size, bsz, ...)
        reshaped_episodes = [
            sampled_episodes[dp_idx * bsz : (dp_idx + 1) * bsz]
            for dp_idx in range(self.dp_size)
        ]
        return reshaped_episodes

    @endpoint
    async def evict(self, curr_policy_version: int) -> None:
        """Evict episodes from the replay buffer if they are too old based on the current policy version
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
    async def _getitem(self, idx: int):
        return self.buffer[idx]

    @endpoint
    async def _numel(self) -> int:
        """Number of elements (episodes) in the replay buffer."""
        return len(self.buffer)

    @endpoint
    async def clear(self) -> None:
        """Clear the replay buffer immediately - dropping all episodes."""
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
