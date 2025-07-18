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

These abstract interfaces are useful to define the API for each component. This
plays nicely with the `stack` APIs, and also makes it so that we can write
a generic orchestrator that can work with any component that implements these.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from monarch.actor import Actor, endpoint

from forge.rl.environments.base import Action, Observation, State


# ==== Data Structures ====
# Of course, these data structures can be replaced by almost anything.


@dataclass
class Trajectory:
    """A trajectory containing a sequence of states, actions, etc."""

    states: list[Observation] = field(default_factory=list)
    actions: list[Action] = field(default_factory=list)


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
    async def generate(self, request: Observation) -> Action:
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


class RewarderInterface(Actor, ABC):
    """Abstract interface for reward computation."""

    @endpoint
    @abstractmethod
    async def compute_reward(
        self, state: State, action: Action, next_state: State
    ) -> float:
        """Compute reward for a state-action-next_state transition."""
        pass


class TrainerInterface(Actor, ABC):
    """Abstract interface for trainers."""

    @abstractmethod
    def step(self):
        """Perform a training step."""
        pass
