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
from dataclasses import dataclass
from typing import Any


@dataclass
class Observation:
    """Base class for environment observations.

    Contract:
    - Should contain all information needed by an agent to make decisions
    - Should be serializable/deserializable
    - Should be immutable (or treated as such)
    """

    pass


@dataclass
class Action:
    """Base class for environment actions.

    Contract:
    - Should contain all information needed to execute a step in the environment
    - Should be serializable/deserializable
    - Should be immutable (or treated as such)
    """

    pass


@dataclass
class State:
    """Base class for environment state.

    Contract:
    - Should contain all information needed to restore the environment
    - Should be serializable/deserializable
    - May contain information not exposed in observations
    """

    pass


class Environment(abc.ABC):
    """Abstract base class for environments."""

    @abc.abstractmethod
    def reset(self) -> Observation:
        """Reset the environment and return an initial observation."""
        pass

    @abc.abstractmethod
    def step(self, action: Any) -> tuple[Observation, float]:
        """Take a step in the environment and return an observation."""
        pass

    @property
    @abc.abstractmethod
    def state(self) -> State:
        """Get the current state of the environment."""
        pass
