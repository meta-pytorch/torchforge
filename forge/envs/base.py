# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import abc

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Union


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


@dataclass(kw_only=True)
class Observation:
    """Base class for environment observations.

    Contract:
    - Should contain all information needed by an agent to make decisions
    - Should be serializable/deserializable
    - Should be immutable (or treated as such)
    Args:
        done: Whether the episode/conversation is complete
        reward: Optional reward signal (can be boolean, int, or float)
        metadata: Additional data that doesn't affect agent decisions but may be useful
                 for transforms, logging, evaluation, etc.
    """

    done: bool = False
    reward: Optional[Union[bool, int, float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(kw_only=True)
class Action:
    """Base class for environment actions.

    Contract:
    - Should contain all information needed to execute a step in the environment
    - Should be serializable/deserializable
    - Should be immutable (or treated as such)

    Args:
        metadata: Additional data that may be useful for logging, debugging, or transforms
    """

    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(kw_only=True)
class State:
    """Base class for environment state.

    Contract:
    - Should contain all information needed to restore the environment
    - Should be serializable/deserializable
    - May contain information not exposed in observations

    Args:
        metadata: Additional state information that may be useful for debugging or analysis
    """

    metadata: Dict[str, Any] = field(default_factory=dict)


class Transform(abc.ABC):
    """Abstract base class for observation transforms.

    Transforms are first-class citizens that can modify observations,
    typically to add rewards, compute metrics, or modify state.

    They follow a functional interface where they take an observation
    and return a (potentially modified) observation.
    """

    @abc.abstractmethod
    def __call__(self, observation: Observation) -> Observation:
        """Transform an observation.

        Args:
            observation: The input observation to transform

        Returns:
            The transformed observation (may be the same instance if no changes)
        """
        pass


class Environment(abc.ABC):
    """Abstract base class for environments.

    Args:
        transform: Optional transform that modifies observations, typically to add rewards.
                  Can be a Transform instance or a callable for backward compatibility.
    """

    def __init__(
        self,
        transform: Optional[
            Union[Transform, Callable[[Observation], Observation]]
        ] = None,
    ):
        self.transform = transform

    @abc.abstractmethod
    def reset(self) -> Observation:
        """Reset the environment and return an initial observation."""
        pass

    @abc.abstractmethod
    def step(self, action: Any) -> Observation:
        """Take a step in the environment and return an observation."""
        pass

    @property
    @abc.abstractmethod
    def state(self) -> State:
        """Get the current state of the environment."""
        pass

    def _apply_transform(self, observation: Observation) -> Observation:
        """Apply the transform to an observation if one is provided."""
        if self.transform is not None:
            return self.transform(observation)
        return observation
