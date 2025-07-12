# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Concrete policy implementations."""

import torch
from monarch.actor import endpoint

from forge.rl.environments import Action, Observation
from forge.rl.environments.toy import ToyAction
from forge.rl.interfaces import PolicyInterface


class ToyPolicy(PolicyInterface):
    """A simple toy policy for testing."""

    def __init__(self, action_range: tuple[float, float] = (-1.0, 1.0)):
        super().__init__()
        self.action_range = action_range

    @endpoint
    async def generate(self, request: Observation) -> Action:
        """Generate a simple random action."""
        # Generate a random action within the specified range
        action_value = (
            torch.rand(1).item() * (self.action_range[1] - self.action_range[0])
            + self.action_range[0]
        )
        action = ToyAction(
            data=torch.tensor([action_value]),
        )
        return action

    @endpoint
    async def update_weights(self):
        """No-op for toy policy."""
        pass
