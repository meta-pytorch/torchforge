# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import random
from dataclasses import dataclass
from typing import Any, Callable

from monarch.actor import endpoint

from forge.controller import ForgeActor

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclass
class ReplayBuffer(ForgeActor):
    """Simple in-memory replay buffer implementation."""

    batch_size: int
    max_policy_age: int
    dp_size: int = 1
    seed: int | None = None
    collate: Callable = lambda batch: batch

    def __post_init__(self):
        super().__init__()

    @endpoint
    async def setup(self) -> None:
        self.buffer: list = []
        if self.seed is None:
            self.seed = random.randint(0, 2**32)
        random.seed(self.seed)
        self.sampler = random.sample

    @endpoint
    async def add(self, episode: "Episode") -> None:
        self.buffer.append(episode)

    @endpoint
    async def sample(
        self, curr_policy_version: int, batch_size: int | None = None
    ) -> tuple[tuple[Any, ...], ...] | None:
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

        # TODO: prefetch samples in advance
        idx_to_sample = self.sampler(range(len(self.buffer)), k=total_samples)
        # Pop episodes in descending order to avoid shifting issues
        popped = [self.buffer.pop(i) for i in sorted(idx_to_sample, reverse=True)]

        # Reorder popped episodes to match the original random sample order
        sorted_idxs = sorted(idx_to_sample, reverse=True)
        idx_to_popped = dict(zip(sorted_idxs, popped))
        sampled_episodes = [idx_to_popped[i] for i in idx_to_sample]

        # Reshape into (dp_size, bsz, ...)
        reshaped_episodes = [
            sampled_episodes[dp_idx * bsz : (dp_idx + 1) * bsz]
            for dp_idx in range(self.dp_size)
        ]

        return self.collate(reshaped_episodes)

    @endpoint
    async def evict(self, curr_policy_version: int) -> None:
        """Evict episodes from the replay buffer if they are too old based on the current policy version
        and the max policy age allowed.

        Args:
            curr_policy_version (int): The current policy version.
        """
        self._evict(curr_policy_version)

    def _evict(self, curr_policy_version: int) -> None:
        buffer_len_before_evict = len(self.buffer)
        self.buffer = [
            trajectory
            for trajectory in self.buffer
            if (curr_policy_version - trajectory.policy_version) <= self.max_policy_age
        ]
        buffer_len_after_evict = len(self.buffer)

        logger.debug(
            f"maximum policy age: {self.max_policy_age}, current policy version: {curr_policy_version}, "
            f"{buffer_len_before_evict - buffer_len_after_evict} episodes expired, {buffer_len_after_evict} episodes left"
        )

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
        logger.debug("replay buffer cleared")

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
