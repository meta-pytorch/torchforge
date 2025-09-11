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

    store: StoreInterface
    batch_size: int
    max_policy_age: int
    dp_size: int = 1
    seed: int | None = None

    @endpoint
    async def setup(self) -> None:
        if self.seed is None:
            self.seed = random.randint(0, 2**32)
        random.seed(self.seed)
        self.sampler = random.sample

    @endpoint
    async def add(self, episode) -> None:
        await self._add(episode)

    async def _add(self, episode) -> None:
        key = f"rb_ep_{await self.store.numel()}_{uuid.uuid4().hex}"
        await self.store.put(key, episode)

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
        await self._evict(curr_policy_version)

        total_available = await self.store.numel()
        if total_samples > total_available:
            return None

        keys = await self.store.keys()

        # TODO: Make this more efficient
        idx_to_sample = self.sampler(range(len(keys)), k=total_samples)
        # Pop episodes in descending order to avoid shifting issues
        sorted_idxs = sorted(idx_to_sample, reverse=True)
        popped = [await self.store.pop(keys[i]) for i in sorted_idxs]

        # Reorder popped episodes to match the original random sample order
        idx_to_popped = dict(zip(sorted_idxs, popped))
        sampled_episodes = [idx_to_popped[i] for i in idx_to_sample]

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
        keys = await self.store.keys()
        for key in keys:
            episode = await self.store.get(key)
            if (curr_policy_version - episode.policy_version) > self.max_policy_age:
                await self.store.delete(key)

    @endpoint
    async def _getitem(self, key: str):
        return await self.store.get(key)

    @endpoint
    async def _numel(self) -> int:
        """Number of elements (episodes) in the replay buffer."""
        return await self.store.numel()

    @endpoint
    async def clear(self) -> None:
        """Clear the replay buffer immediately - dropping all episodes."""
        await self._clear()

    async def _clear(self) -> None:
        await self.store.delete_all()

    @endpoint
    async def state_dict(self) -> dict[str, Any]:
        keys = await self.store.keys()
        episodes = [await self.store.get(k) for k in keys]
        return {
            "buffer": episodes,
            "rng_state": random.getstate(),
            "seed": self.seed,
        }

    @endpoint
    async def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        await self._clear()
        for ep in state_dict["buffer"]:
            await self._add(ep)
        random.setstate(state_dict["rng_state"])
        self.seed = state_dict["seed"]
