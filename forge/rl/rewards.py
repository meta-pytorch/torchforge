# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from monarch.actor_mesh import Actor, endpoint

from forge.rl.entities import ForgeRequest


class Rewarder(Actor):
    """Base class for reward computation."""

    @endpoint
    async def compute_reward(
        self, state: ForgeRequest, action: ForgeRequest, next_state: ForgeRequest
    ) -> float:
        """Compute reward for a state-action-next_state transition."""
        return 0.0


class ToyRewarder(Rewarder):
    """A very simple toy rewarder for testing data flow."""

    @endpoint
    async def compute_reward(
        self, state: ForgeRequest, action: ForgeRequest, next_state: ForgeRequest
    ) -> float:
        """Simple reward: next_state_value + 1."""

        # Extract the state value from next_state
        if next_state.data is not None and len(next_state.data) > 0:
            state_value = float(next_state.data[0])
            return state_value + 1.0

        # Default reward if no data
        return 1.0
