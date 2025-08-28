# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import random
from typing import Any, Generic, List, Mapping, TypeVar

from forge.interfaces import BufferView, StatefulSampler

K = TypeVar("K")
V = TypeVar("V")


class RandomStatefulSampler(StatefulSampler[K, V], Generic[K, V]):
    """A simple stateful sampler that uses Python's random.sample for deterministic sampling.

    This sampler maintains an internal random state that can be saved and restored,
    allowing for reproducible sampling behavior. It uses random.sample to select
    keys from the buffer without replacement.
    """

    def __init__(self, seed: int | None = None):
        """Initialize the sampler with an optional random seed.

        Args:
            seed: Optional seed for the random number generator. If None,
                  the sampler will use Python's default random state.
        """
        if seed is None:
            self._random = random.Random()
        self._random = random.Random(seed)

    def sample_keys(self, buffer: BufferView[K, V], num: int) -> List[K]:
        """Sample keys from the buffer using random.sample.

        Args:
            buffer: The buffer to sample from
            num: Number of keys to sample

        Returns:
            A list of sampled keys. If num is greater than the buffer size,
            returns all available keys.
        """
        # Get all keys from the buffer
        all_keys = list(buffer.keys())

        # If requesting more samples than available, return all keys
        if num >= len(all_keys):
            return all_keys

        # Use random.sample for sampling without replacement
        return self._random.sample(all_keys, num)

    def state_dict(self):
        """Return the state dict of the sampler.

        Returns:
            A dictionary containing the random number generator state.
        """
        return {"random_state": self._random.getstate()}

    def set_state_dict(self, state_dict: Mapping[str, Any]):
        """Set the state dict of the sampler.

        Args:
            state_dict: Dictionary containing the random state to restore.
        """
        if "random_state" in state_dict:
            self._random.setstate(state_dict["random_state"])
        else:
            raise ValueError("Missing 'random_state' in state dict")
