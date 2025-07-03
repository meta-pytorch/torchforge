# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Abstract interfaces for RL components.

TODO: Revisit the organization later - we may want to restructure how we organize
concrete implementations vs interfaces, and consider whether we want separate files
for each component type or group related interfaces together.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from monarch.actor_mesh import Actor, endpoint


# ==== Data Structures ====


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


# ==== Abstract Interfaces ====


class PolicyInterface(Actor, ABC):
    """Abstract interface for policies."""

    @endpoint
    @abstractmethod
    async def generate(self, request: ForgeRequest) -> ForgeRequest:
        """Generate an action given a state/request."""
        pass

    @endpoint
    @abstractmethod
    async def update_weights(self):
        """Update the policy weights."""
        pass


class ReplayBufferInterface(Actor, ABC):
    """Abstract interface for replay buffers."""

    @endpoint
    @abstractmethod
    async def extend(self, sample: ForgeTrajectory):
        """Add a trajectory to the replay buffer."""
        pass

    @endpoint
    @abstractmethod
    async def sample(self) -> ForgeTrajectory | None:
        """Sample from the replay buffer."""
        pass

    @endpoint
    @abstractmethod
    async def len(self) -> int:
        """Return the length of the replay buffer."""
        pass

    @endpoint
    @abstractmethod
    async def is_empty(self) -> bool:
        """Check if the replay buffer is empty."""
        pass


class RewarderInterface(Actor, ABC):
    """Abstract interface for reward computation."""

    @endpoint
    @abstractmethod
    async def compute_reward(
        self, state: ForgeRequest, action: ForgeRequest, next_state: ForgeRequest
    ) -> float:
        """Compute reward for a state-action-next_state transition."""
        pass


class EnvironmentInterface(ABC):
    """Abstract interface for environments."""

    @abstractmethod
    def reset(self) -> tuple[ForgeRequest, ForgeEnvInfo]:
        """Reset the environment and return initial state and info."""
        pass

    @abstractmethod
    def step(
        self, action: ForgeRequest
    ) -> tuple[ForgeRequest, float, bool, bool, ForgeEnvInfo]:
        """Take a step in the environment and return (obs, reward, terminated, truncated, info)."""
        pass


class TrainerInterface(Actor, ABC):
    """Abstract interface for trainers."""

    @abstractmethod
    def step(self):
        """Perform a training step."""
        pass
