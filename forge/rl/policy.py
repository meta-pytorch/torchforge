# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Concrete policy implementations."""

import random

import torch
from monarch.actor_mesh import endpoint

from forge.rl.interfaces import ForgeRequest, PolicyInterface


class ToyPolicy(PolicyInterface):
    """A simple toy policy for testing."""

    def __init__(self, action_range: tuple[float, float] = (-1.0, 1.0)):
        super().__init__()
        self.action_range = action_range

    @endpoint
    async def generate(self, request: ForgeRequest) -> ForgeRequest:
        """Generate a simple random action."""
        # Generate a random action within the specified range
        action_value = random.uniform(self.action_range[0], self.action_range[1])

        action = ForgeRequest(
            data=torch.tensor([action_value]),
            text=f"Action: {action_value:.2f}",
            metadata={"action_value": action_value, "policy_type": "toy"},
        )

        return action

    @endpoint
    async def update_weights(self):
        """No-op for toy policy."""
        pass
