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
from forge.data.raw_buffer import SimpleRawBuffer
from forge.data.stateful_sampler import RandomStatefulSampler

from forge.interfaces import StatefulSampler

from monarch.actor import endpoint


@dataclass
class ReplayBuffer(ForgeActor):
    """Simple in-memory replay buffer implementation."""

    batch_size: int
    max_policy_age: int
    seed: int | None = None

    @endpoint
    async def setup(self, *, sampler: StatefulSampler | None = None) -> None:
        self._buffer = SimpleRawBuffer[int, Any]()
        if self.seed is None:
            self.seed = random.randint(0, 2**32)
        if sampler is None:
            sampler = RandomStatefulSampler(seed=self.seed)

        self._sampler = sampler

    @endpoint
    async def add(self, episode) -> None:
        # I think key should be provided by the caller, but let's just generate a random one for now
        # Note that this means add() is not deterministic, however the original implementation using list
        # isn't actually deterministic either because it depends on the order of add() being called.
        # Alternatively, add a field in Trajectory as the id of the trajectory.
        key = uuid.uuid4().int
        self._buffer.add(key, episode)

    @endpoint
    async def sample(self, curr_policy_version: int, batch_size: int | None = None):
        """Sample from the replay buffer.

        Args:
            curr_policy_version (int): The current policy version.
            batch_size (int, optional): Number of episodes to sample. If none, defaults to batch size
                passed in at initialization.

        Returns:
            A list of sampled episodes or None if there are not enough episodes in the buffer.
        """
        bsz = batch_size if batch_size is not None else self.batch_size

        # Evict old episodes
        self._evict(curr_policy_version)

        if bsz > len(self._buffer):
            return None

        keys_to_sample = self._sampler.sample_keys(self._buffer, num=bsz)
        sampled_episodes = [self._buffer.pop(k) for k in keys_to_sample]
        return sampled_episodes

    @endpoint
    async def evict(self, curr_policy_version: int) -> None:
        """Evict episodes from the replay buffer if they are too old based on the current policy version
        and the max policy age allowed.

        Args:
            curr_policy_version (int): The current policy version.
        """
        self._evict(curr_policy_version)

    def _evict(self, curr_policy_version: int) -> None:
        keys_to_delete = []
        for key, episode in self._buffer:
            if curr_policy_version - episode.policy_version > self.max_policy_age:
                keys_to_delete.append(key)
        for key in keys_to_delete:
            self._buffer.pop(key)

    @endpoint
    async def _numel(self) -> int:
        """Number of elements (episodes) in the replay buffer."""
        return len(self._buffer)

    @endpoint
    async def clear(self) -> None:
        """Clear the replay buffer immediately - dropping all episodes."""
        self._buffer.clear()

    @endpoint
    async def state_dict(self) -> dict[str, Any]:
        return {
            "buffer": self._buffer,
            "sampler_state": self._sampler.state_dict(),
            "seed": self.seed,
        }

    @endpoint
    async def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._buffer = state_dict["buffer"]
        self._sampler.set_state_dict(state_dict["sampler_state"])
        self.seed = state_dict["seed"]
