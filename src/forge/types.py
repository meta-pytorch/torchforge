# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Any, Literal

import torch

Role = Literal[
    "system",  # Origin is system prompt
    "user",  # Origin is user
    "assistant",  # Origin is the model output
    "ipython",  # Origin is return from a tool call
    "tool",  # Origin is return from a tool call
]


@dataclass
class Message:
    """Generic message class in OpenAI chat completion format.

    role: 'system', 'user', 'assistant', etc.
    content: Either a string message or a dictionary denoting multimodal content.

    Example::
        >>> prompt = Message(role='user', content="Tell me a joke.")

    Full spec for accepted content format can be found here: <>
    """

    role: Role
    content: str | dict[str, Any]


@dataclass
class State:
    observations: list[Message]
    input_ids: torch.Tensor | None = None
    mask: torch.Tensor | None = None
    target: torch.Tensor | None = None
    truncated: bool = False
    terminated: bool = False
    transforms: dict[str, torch.Tensor] | None = None
    rewards: dict[str, Any] | None = None
    policy_version: int | None = None
    info: dict[str, Any] | None = None
