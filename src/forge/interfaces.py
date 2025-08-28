# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from typing import Any, Generic, Iterable, Mapping, TypeVar

from forge.types import Action, Message, Observation, Scalar, State

from monarch.actor import Actor, endpoint

K = TypeVar("K")
V = TypeVar("V")


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


class Policy(Actor, ABC):
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


class BufferView(ABC, Generic[K, V]):
    """Abstract base class for a view into a buffer with key-value pairs.

    This class defines the interface for accessing elements in a buffer
    through dictionary-like operations. It supports generic key and value types.
    """

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of key-value pairs in the buffer.

        Returns:
            int: The number of items in the buffer.
        """
        pass

    @abstractmethod
    def __getitem__(self, key: K) -> V:
        """Retrieve a value from the buffer using the specified key.

        Args:
            key (K): The key to look up in the buffer.

        Returns:
            V: The value associated with the key.

        Raises:
            KeyError: If the key is not found in the buffer.
        """
        pass

    @abstractmethod
    def __iter__(self) -> Iterable[tuple[K, V]]:
        """Return an iterator over the key-value pairs in the buffer.

        Returns:
            Iterable[tuple[K, V]]: An iterator yielding (key, value) tuples.
        """
        pass

    @abstractmethod
    def keys(self) -> Iterable[K]:
        """Return an iterable of all keys in the buffer.

        Returns:
            Iterable[K]: An iterable containing all keys in the buffer.
        """
        pass


class RawBuffer(BufferView[K, V], ABC):
    """Abstract interface for the underlying storage backend (raw buffer) of a ReplayBuffer."""

    @abstractmethod
    def add(self, key: K, val: V) -> None:
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
    def pop(self, key: K) -> V:
        """
        Remove and return a value from the buffer using the specified key.

        Args:
            key (K): The key to look up and remove from the buffer

        Returns:
            V: The value associated with the key before removal
        """
        pass

    @abstractmethod
    def clear(self) -> None:
        """
        Remove all key-value pairs from the buffer, effectively emptying it.

        This method should reset the buffer to its initial empty state.

        Returns:
            None
        """
        pass


class StatefulSampler(ABC, Generic[K, V]):
    """Abstract interface for stateful samplers with deterministic behavior given a state.

    This class defines the interface for samplers that maintain internal state and provide
    deterministic sampling behavior when the state is fixed.
    """

    @abstractmethod
    def sample_keys(self, buffer: BufferView[K, V], num: int) -> list[K]:
        """Return the keys of selected samples from the buffer.

        This method samples a specified number of keys from the provided buffer
        according to the sampler's internal sampling strategy. The sampling
        behavior is deterministic for a given internal state of the sampler.

        Args:
            buffer (BufferView[K, V]): The buffer to sample from, containing key-value pairs.
            num (int): Desired number of samples to retrieve from the buffer.
                      If num is greater than the buffer size, implementation may
                      return fewer samples or handle it according to the specific
                      sampling strategy.

        Returns:
            list[K]: A list of keys corresponding to the selected samples.
                    The length of this list will typically be equal to num,
                    unless the buffer contains fewer items.
        """
        pass

    @abstractmethod
    def state_dict(self) -> Mapping[str, Any]:
        """Return the state dict of the sampler.

        This method should capture all the internal state necessary to reproduce
        the sampler's behavior, such as random number generator states.

        Returns:
            dict: A dictionary containing the internal state of the sampler.
        """
        pass

    @abstractmethod
    def set_state_dict(self, state_dict):
        """Set the state dict of the sampler.

        Args:
            state_dict (dict): A dictionary containing the internal state to restore
                               the sampler to a specific configuration.
        """
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


# TODO
# class RLLoss(ABC):

# class SFTLoss(ABC): # inherit from titan loss
# from torchtitan.components.loss import LossFunction
