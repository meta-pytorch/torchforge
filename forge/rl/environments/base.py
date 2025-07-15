# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Union


@dataclass
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


@dataclass
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


@dataclass
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
        """Initialize the environment with an optional transform."""
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
        """Apply the transform to an observation and ensure type consistency.

        This method handles transforms that return:
        1. A new Observation instance (same or different subclass)
        2. None (indicating no changes)

        The method ensures the returned observation is of the same type as the input,
        preserving the original observation's class while incorporating any changes
        from the transform.

        Args:
            observation: The observation to transform

        Returns:
            The transformed observation with the same type as the input

        Examples:
            >>> env = MyEnvironment()
            >>> obs = env.reset()
            >>>
            >>> # Example 1: Transform returns None (no changes)
            >>> def no_change_transform(o):
            ...     return None
            >>> env.transform = no_change_transform
            >>> transformed = env._apply_transform(obs)  # Returns original observation
            >>>
            >>> # Example 2: Transform returns a different observation type
            >>> def type_changing_transform(o):
            ...     return DifferentObservation(done=True)
            >>> env.transform = type_changing_transform
            >>> transformed = env._apply_transform(obs)  # Original obs type with done=True
        """
        if self.transform is None:
            return observation

        result = self.transform(observation)

        if result is None:
            return observation

        if not isinstance(result, type(observation)):
            return self._convert_observation_type(observation, result)

        return result

    def _convert_observation_type(
        self, target_obs: Observation, source_obs: Observation
    ) -> Observation:
        """Convert an observation to the target observation type.

        Creates a new observation of the target type and copies all matching
        attributes from the source observation.

        Args:
            target_obs: The observation whose type should be preserved
            source_obs: The observation containing new values

        Returns:
            A new observation of the target type with values from the source

        Examples:
            >>> target = MyObservation(done=False, reward=None)
            >>> source = DifferentObservation(done=True, reward=1.0, extra="ignored")
            >>> new_obs = env._convert_observation_type(target, source)
            >>> print(type(new_obs), new_obs.done, new_obs.reward)
            <class 'MyObservation'> True 1.0
        """
        new_obs = type(target_obs)()

        for attr_name, attr_value in vars(target_obs).items():
            if hasattr(source_obs, attr_name):
                setattr(new_obs, attr_name, getattr(source_obs, attr_name))
            else:
                setattr(new_obs, attr_name, attr_value)

        return new_obs
