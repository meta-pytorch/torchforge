# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import random
from dataclasses import dataclass

import torch
from monarch.actor_mesh import Actor, endpoint


@dataclass
class ForgeRequest:
    """A request containing state/observation data."""

    data: torch.Tensor | None = None
    text: str | None = None
    metadata: dict | None = None


@dataclass
class ForgeTrajectory:
    """A trajectory containing a sequence of states, actions, rewards, etc."""

    states: list[ForgeRequest] | None = None
    actions: list[ForgeRequest] | None = None
    rewards: list[float] | None = None
    dones: list[bool] | None = None
    infos: list[dict] | None = None

    def __post_init__(self):
        if self.states is None:
            self.states = []
        if self.actions is None:
            self.actions = []
        if self.rewards is None:
            self.rewards = []
        if self.dones is None:
            self.dones = []
        if self.infos is None:
            self.infos = []


@dataclass
class ForgeEnvInfo:
    """Environment info returned with observations."""

    episode_id: str | None = None
    step_count: int = 0
    metadata: dict | None = None


# ==== Generic RL entities ====
class Trainer(Actor):
    def step(self):
        pass


class Policy(Actor):
    @endpoint
    async def generate(self, request: ForgeRequest) -> ForgeRequest:
        # e.g. run vLLM
        raise NotImplementedError("Subclasses must implement generate method")

    @endpoint
    async def update_weights(self):
        """Updates the weights of the policy."""
        pass


class ToyPolicy(Policy):
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


class EnvironmentTransformActor(Actor):
    @endpoint
    async def reward(self, request: ForgeRequest) -> float:
        # e.g. run reward model
        raise NotImplementedError("Subclasses must implement reward method")


class ReplayBuffer(Actor):
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
        import random

        return random.choice(self.buffer)

    @endpoint
    async def len(self) -> int:
        """Return the length of the replay buffer."""
        return len(self.buffer)

    @endpoint
    async def is_empty(self) -> bool:
        """Check if the replay buffer is empty."""
        return len(self.buffer) == 0
