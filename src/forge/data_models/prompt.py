# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any


class Role(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    NONE = "none"


@dataclass
class Message:
    """A single message in a conversation."""

    chunks: Sequence[str]
    role: Role


@dataclass
class Prompt:
    """A multi-turn prompt (conversation history)."""

    # Multi-turn messages, each turn is a message.
    messages: Sequence[Message]

    @classmethod
    def from_prompt(
        cls, prompt: str, system_instruction: str | None = None
    ) -> "Prompt":
        messages = prompt_to_messages(prompt, system_instruction)
        return Prompt(
            messages=messages,
        )


def prompt_to_messages(
    prompt: str, system_instruction: str | None = None
) -> Sequence[Message]:
    """Convert a prompt to a sequence of messages."""
    messages = []
    if system_instruction is not None:
        messages.append(Message(chunks=[system_instruction], role=Role.SYSTEM))
    messages.append(
        Message(chunks=[prompt], role=Role.USER),
    )
    return messages


def to_prompt(prompt: str, system_instruction: str | None = None) -> Prompt:
    """Converts a prompt to a sequence of messages."""
    return Prompt(
        messages=prompt_to_messages(prompt, system_instruction),
    )
