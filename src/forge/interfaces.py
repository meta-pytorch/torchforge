# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from typing import Any, Mapping

from forge.types import Message, Observation, Scalar


class Transform(ABC):
    """Abstract base class for observation transforms.

    Transforms are first-class citizens that can modify observations,
    typically to add rewards, compute metrics, or modify state.

    They follow a functional interface where they take an observation
    and return a (potentially modified) observation.
    """

    @abstractmethod
    def __call__(self, observation: Observation) -> Observation:
        """Transform an observation.

        Args:
            observation: The input observation to transform

        Returns:
            The transformed observation (may be the same instance if no changes)
        """
        pass


class BaseTokenizer(ABC):
    """
    Abstract token encoding model that implements ``encode`` and ``decode`` methods.
    See :class:`forge.data.HuggingFaceModelTokenizer for an example implementation of this protocol.
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
    the ``tokenize_messages`` method. See :class:`forge.data.HuggingFaceModelTokenizer`
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


class MetricLogger(ABC):
    """Abstract metric logger."""

    @abstractmethod
    def is_log_step(self, name: str, step: int) -> bool:
        """Returns true if the current step is a logging step.

        Args:
            name (str): metric name (for checking the freq for this metric)
            step (int): current step
        """
        pass

    @abstractmethod
    def log(self, name: str, data: Scalar, step: int) -> None:
        """Log scalar data if this is a logging step.

        Args:
            name (str): tag name used to group scalars
            data (Scalar): scalar data to log
            step (int): step value to record
        """
        pass

    @abstractmethod
    def log_dict(self, metrics: Mapping[str, Scalar], step: int) -> None:
        """Log multiple scalar values if this is a logging step.

        Args:
            metrics (Mapping[str, Scalar]): dictionary of tag name and scalar value
            step (int): step value to record
        """
        pass

    def __del__(self) -> None:
        self.close()

    def close(self) -> None:
        """
        Close log resource, flushing if necessary.
        This will automatically be called via __del__ when the instance goes out of scope.
        Logs should not be written after `close` is called.
        """


class Reward(ABC):
    """Abstract base class for reward models."""

    @abstractmethod
    def __call__(self, observation: Observation) -> float:
        """Compute a reward for an observation."""
        pass
