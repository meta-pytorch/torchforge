# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
from typing import Any, Literal, TypedDict

import torch

Role = Literal[
    "system",  # Origin is system prompt
    "user",  # Origin is user
    "assistant",  # Origin is the model output
    "ipython",  # Origin is return from a tool call
    "tool",  # Origin is return from a tool call
]


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
    reward: bool | int | float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


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

    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Trajectory:
    """A trajectory containing a sequence of states, actions, etc."""

    policy_version: int
    states: list[Observation] = field(default_factory=list)
    actions: list[Action] = field(default_factory=list)

    def __post_init__(self):
        assert self.policy_version >= 0


class Message(TypedDict):
    role: Role
    content: str | dict[str, Any]
    tools: dict[str, Any] | None


@dataclass
class Conversation:
    messages: list[Message]
    input_ids: torch.Tensor
    mask: torch.Tensor
    label: torch.Tensor
    # Extra things for RL
    logprobs: dict[str, torch.Tensor] | None = None
    rewards: dict[str, Any] | None = None
    policy_version: int | None = None


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

    metadata: dict[str, Any] = field(default_factory=dict)
