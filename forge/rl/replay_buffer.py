# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Concrete replay buffer implementations."""

import random

from monarch.actor_mesh import endpoint

from forge.rl.interfaces import ForgeTrajectory, ReplayBufferInterface


# Silly replay buffer implementation for testing.
# One nice thing if we implement our own Replay buffer is that
# we can wrap RDMA calls / torchstore calls here.
class ReplayBuffer(ReplayBufferInterface):
    """Simple in-memory replay buffer implementation."""

    def __init__(self):
        self.buffer: list[ForgeTrajectory] = []

    @endpoint
    async def extend(self, sample: ForgeTrajectory):
        """Add a trajectory to the replay buffer."""
        self.buffer.append(sample)

    @endpoint
    async def sample(self):
        """Sample from the replay buffer."""
        if not self.buffer:
            return None
        return random.choice(self.buffer)

    @endpoint
    async def len(self) -> int:
        """Return the length of the replay buffer."""
        return len(self.buffer)

    @endpoint
    async def is_empty(self) -> bool:
        """Check if the replay buffer is empty."""
        return len(self.buffer) == 0
