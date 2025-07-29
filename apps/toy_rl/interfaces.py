# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from typing import Any

from forge.interfaces import Transform

from forge.types import Action, Observation, State, Trajectory
from monarch.actor_mesh import Actor, endpoint


class Environment(ABC):
    """Abstract base class for environments.

    Args:
        transform: Optional transform that modifies observations, typically to add rewards.
                  Can be a Transform instance or a callable for backward compatibility.
    """

    def __init__(
        self,
        transform: Transform | None = None,
    ):
        self.transform = transform

    @abstractmethod
    def reset(self) -> Observation:
        """Reset the environment and return an initial observation."""
        pass

    @abstractmethod
    def step(self, action: Any) -> Observation:
        """Take a step in the environment and return an observation."""
        pass

    @property
    @abstractmethod
    def state(self) -> State:
        """Get the current state of the environment."""
        pass

    def _apply_transform(self, observation: Observation) -> Observation:
        """Apply the transform to an observation if one is provided."""
        if self.transform is not None:
            return self.transform(observation)
        return observation


class Policy(Actor, ABC):
    """Abstract interface for policies."""

    @endpoint
    @abstractmethod
    async def generate(self, request: Observation) -> Action:
        """Generate an action given a state/request."""
        pass

    @endpoint
    @abstractmethod
    async def update_weights(self):
        """Update the policy weights."""
        pass


class ReplayBuffer(Actor, ABC):
    """Abstract interface for replay buffers."""

    @endpoint
    @abstractmethod
    async def extend(self, sample: Trajectory):
        """Add a trajectory to the replay buffer."""
        pass

    @endpoint
    @abstractmethod
    async def sample(self, batch_size: int) -> list[Trajectory] | None:
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
