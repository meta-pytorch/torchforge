# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from typing import Any, List, Mapping

from forge.controller import ForgeActor

from forge.types import Action, Message, Observation, Scalar, State

from monarch.actor import endpoint


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


class Policy(ForgeActor, ABC):
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


class StoreInterface(ABC):
    """
    Abstract base class for a KV store. This closely follows the interface of
    torchstore.
    """

    # TODO: support this in torchstore.
    @abstractmethod
    async def numel(self, prefix=None) -> int:
        """Return the number of keys starting with the given prefix.
        The prefix matching follows reverse domain name notation convention.

        Args:
        prefix (str): The prefix to match against stored keys.
            For example, "xyz" matches "xyz.abc.def" but "xy" does not.
            Note: None is the prefix of all keys, while "" is the prefix of keys
            starting with "." and "" itself.

        Returns:
            int: The number of keys matching the prefix in the store.
        """
        pass

    @abstractmethod
    async def keys(self, prefix=None) -> List[str]:
        """Return an iterable of all keys in the store matching the given prefix.
        The prefix matching follows reverse domain name notation convention.

        Args:
        prefix (str): The prefix to match against stored keys.
            For example, "xyz" matches "xyz.abc.def" but "xy" does not.
            Note: None is the prefix of all keys, while "" is the prefix of keys
            starting with "." and "" itself.

        Returns:
            Iterable[K]: An iterable containing all keys in the buffer.
        """
        pass

    @abstractmethod
    async def put(self, key: str, value: Any) -> None:
        """
        Add a key-value pair to the buffer.

        Args:
            key (K): The key to store the value under
            val (V): The value to store in the buffer

        Returns:
            None
        """
        pass

    @abstractmethod
    async def get(self, key: str) -> Any:
        """
        Get a key-value pair from the store.

        Args:
            key (K): The key to get

        Returns:
            V: The value stored under the key

        Raises:
            KeyError: If the key does not exist in the store
        """
        pass

    @abstractmethod
    async def exists(self, key: str) -> bool:
        """
        Check if a key exists in the store.
        """
        pass

    # TODO: support this in torchstore.
    @abstractmethod
    async def pop(self, key: str) -> Any:
        """
        Get a key-value pair from the store, and delete it from the store.

        Args:
            key (K): The key to get

        Returns:
            V: The value stored under the key

        Raises:
            KeyError: If the key does not exist in the store
        """

    # TODO: support this in torchstore.
    @abstractmethod
    async def delete(self, key: str) -> None:
        """
        Delete a key-value pair from the store.

        Args:
            key (K): The key to delete

        Returns:
            None

        Raises:
            KeyError: If the key does not exist in the store
        """
        pass

    # TODO: support this in torchstore.
    @abstractmethod
    async def delete_all(self, prefix=None) -> int:
        """
        Delete all key-value pairs from the store matching the given prefix.
        The prefix matching follows reverse domain name notation convention.

        Args:
            prefix (str): The prefix to match against stored keys.
                For example, "xyz" matches "xyz.abc.def" but "xy" does not.
                Note: None is the prefix of all keys, while "" is the prefix of keys
                starting with "." and "" itself.

        Returns:
            int: The number of keys deleted from the store.
        """
        pass


# TODO
# class RLLoss(ABC):

# class SFTLoss(ABC): # inherit from titan loss
# from torchtitan.components.loss import LossFunction
