# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from typing import Any, Callable

from monarch.actor import Actor, endpoint

from forge.types import Message, State


# class Transform(ABC):
#     """Abstract base class for observation transforms.

#     Transforms are first-class citizens that can modify observations,
#     typically to add rewards, compute metrics, or modify state.

#     They follow a functional interface where they take an observation
#     and return a (potentially modified) observation.
#     """

#     @abstractmethod
#     def __call__(self, observation: Observation) -> Observation:
#         """Transform an observation.

#         Args:
#             observation: The input observation to transform

#         Returns:
#             The transformed observation (may be the same instance if no changes)
#         """
#         pass


class Environment(ABC):
    """Abstract base class for environments.

    Args:
        transform: Optional transform that modifies observations, typically to add rewards.
                  Can be a Transform instance or a callable for backward compatibility.
    """

    def __init__(
        self,
        transform: Callable | None = None,
    ):
        self.transform = transform

    @abstractmethod
    def reset(self) -> State:
        """Reset the environment and return an initial observation."""
        pass

    @abstractmethod
    def step(self, state: State) -> State:
        """Take a step in the environment and return an observation."""
        pass

    # @property
    # @abstractmethod
    # def state(self) -> State:
    #     """Get the current state of the environment."""
    #     pass

    # def _apply_transform(self, observation: Observation) -> Observation:
    #     """Apply the transform to an observation if one is provided."""
    #     if self.transform is not None:
    #         return self.transform(observation)
    #     return observation


class Policy(Actor, ABC):
    """Abstract interface for policies."""

    @endpoint
    @abstractmethod
    async def generate(self, state: State) -> State:
        """Generate an action given a state/request."""
        pass

    @endpoint
    @abstractmethod
    async def update_weights(self):
        """Update the policy weights."""
        pass


class BaseTokenizer(ABC):
    """
    Abstract token encoding model that implements ``encode`` and ``decode`` methods.
    See :class:`~torchtune.modules.transforms.tokenizers.SentencePieceBaseTokenizer` and
    :class:`~torchtune.modules.transforms.tokenizers.TikTokenBaseTokenizer` for example implementations of this protocol.
    """

    @abstractmethod
    def encode(self, text: str, **kwargs: dict[str, Any]) -> list[int]:
        """
        Given a string, return the encoded list of token ids.

        Args:
            text (str): The text to encode.
            **kwargs (dict[str, Any]): kwargs.

        Returns:
            list[int]: The encoded list of token ids.
        """
        pass

    @abstractmethod
    def decode(self, token_ids: list[int], **kwargs: dict[str, Any]) -> str:
        """
        Given a list of token ids, return the decoded text, optionally including special tokens.

        Args:
            token_ids (list[int]): The list of token ids to decode.
            **kwargs (dict[str, Any]): kwargs.

        Returns:
            str: The decoded text.
        """
        pass


class ModelTokenizer(ABC):
    """
    Abstract tokenizer that implements model-specific special token logic in
    the ``tokenize_messages`` method. See :class:`~torchtune.models.llama3.Llama3Tokenizer`
    for an example implementation of this protocol.
    """

    special_tokens: dict[str, int]
    max_seq_len: int | None

    @abstractmethod
    def tokenize_messages(
        self, messages: list[Message], **kwargs: dict[str, Any]
    ) -> tuple[list[int], list[bool]]:
        """
        Given a list of messages, return a list of tokens and list of masks for
        the concatenated and formatted messages.

        Args:
            messages (list[Message]): The list of messages to tokenize.
            **kwargs (dict[str, Any]): kwargs.

        Returns:
            tuple[list[int], list[bool]]: The list of token ids and the list of masks.
        """
        pass
