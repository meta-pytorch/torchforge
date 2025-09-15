# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import random
import uuid
from dataclasses import dataclass
from typing import Any

from forge.controller import ForgeActor
from forge.interfaces import StoreInterface

from monarch.actor import endpoint


@dataclass
class ReplayBuffer(ForgeActor):
    """Simple in-memory replay buffer implementation."""

    backend: StoreInterface
    batch_size: int
    max_policy_age: int
    dp_size: int = 1
    seed: int | None = None

    def __post_init__(self):
        if self.seed is None:
            self.seed = random.randint(0, 2**32)
        random.seed(self.seed)
        self.sampler = random.sample

    @endpoint
    async def add(self, episode) -> None:
        key = f"rb_{uuid.uuid4().hex}"
        await self.backend.put(key, episode)

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
        # TODO: _evict() before keys() isn't concurrency-safe; may need async lock or refactor. See PR #147.
        await self._evict(curr_policy_version)

        keys = await self.backend.keys()

        total_available = await self.backend.numel()
        if total_samples > total_available:
            return None

        # TODO: Make this more efficient
        idx_to_sample = self.sampler(range(len(keys)), k=total_samples)

        # Fetch and remove the sampled episodes
        sampled_episodes = [await self.backend.pop(keys[i]) for i in idx_to_sample]

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
        await self._evict(curr_policy_version)

    async def _evict(self, curr_policy_version: int) -> None:
        keys = await self.backend.keys()
        for key in keys:
            episode = await self.backend.get(key)
            # TODO: Could store keys as policy_version+uuid to evict without fetching each episode
            if (curr_policy_version - episode.policy_version) > self.max_policy_age:
                await self.backend.delete(key)

    @endpoint
    async def _getitem(self, key: str):
        return await self.backend.get(key)

    @endpoint
    async def _numel(self) -> int:
        """Number of elements (episodes) in the replay buffer."""
        return await self.backend.numel()

    @endpoint
    async def clear(self) -> None:
        """Clear the replay buffer immediately - dropping all episodes."""
        await self._clear()

    async def _clear(self) -> None:
        await self.backend.delete_all()

    @endpoint
    async def state_dict(self) -> dict[str, Any]:
        keys = await self.backend.keys()
        episodes = [(k, await self.backend.get(k)) for k in keys]
        return {
            "buffer": episodes,
            "rng_state": random.getstate(),
            "seed": self.seed,
        }

    @endpoint
    async def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        await self._clear()
        for k, ep in state_dict["buffer"]:
            await self.backend.put(k, ep)
        random.setstate(state_dict["rng_state"])
        self.seed = state_dict["seed"]
