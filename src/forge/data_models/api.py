# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

from forge.data_models.completion import Completion

from forge.data_models.loss import LossOutput
from forge.data_models.minibatch import Minibatch
from forge.data_models.prompt import Prompt


# TODO: This file needs should NOT be in the data_models folder/package


class Store(ABC):
    """
    Abstract base class for a generic key-value store.

    This class defines the interface for a storage backend that can save and retrieve
    values using string keys. Subclasses should implement the actual storage logic,
    which could be in-memory, on disk, remote (e.g., RDMA, Redis), or any other backend.

    Example use cases include storing model weights, configuration objects, or any
    other data that needs to be accessed by key.

    Methods:
    put(key: str, value: Any) -> None
        Store a value under the specified key.

    get(key: str) -> Any
        Retrieve the value associated with the specified key.

    Subclasses must implement both methods.
    """

    @abstractmethod
    def put(self, key: str, value: Any) -> None:
        """Store a value under a key."""
        pass

    @abstractmethod
    def get(self, key: str) -> Any:
        """Retrieve a value by key."""
        pass


class WeightsBuffer:
    """
    Concrete class for managing model weights using a generic key-value Store backend.
    This class provides a simple interface to store and retrieve model weights
    (or references to them) by delegating the actual storage logic to a Store instance.
    The Store abstraction allows for flexible backends (e.g., in-memory, RDMA, file system, torchstore etc.)
    without changing the WeightBuffer interface.
    Example usage:
        store = MyCustomStoreBackend()
        buffer = WeightBuffer(store)
        buffer.put("model_weights", weights)
        latest_weights = buffer.get("model_weights")
    Args:
        store (Store): An instance of a Store backend to use for storage.
    """

    def __init__(self, store):
        """
        Initialize the WeightBuffer with a given Store backend.
        Args:
            store (Store): The storage backend to use.
        """
        self.store = store

    def put(self, key: str, weights):
        """
        Store the given weights under the specified key.
        Args:
            key (str): The key under which to store the weights.
            weights: The weights object or reference to store.
        """
        self.store.put(key, weights)

    def get(self, key: str):
        """
        Retrieve the weights stored under the specified key.
        Args:
            key (str): The key for which to retrieve the weights.
        Returns:
            The weights object or reference associated with the key.
        """
        return self.store.get(key)


class Trainer(ABC):
    """
    Abstract base class for a reinforcement learning (RL) trainer.
    This class defines the interface for any RL trainer implementation.
    It standardizes the methods required for gradient accumulation, applying updates,
    and snapshotting model weights. Subclasses should implement the actual logic
    for these operations, which may vary depending on the underlying model,
    framework, or distributed setup.
    """

    @abstractmethod
    def accummulate_gradients(self, minibatch: Minibatch) -> LossOutput:
        """
        Accumulate gradients for the given minibatch.
        This method is called once per minibatch during training. It should compute
        the gradients for the minibatch and accumulate them (without applying them yet).

        Args:
            minibatch (Minibatch): The minibatch of data to use for gradient computation.
        Returns:
            LossOutput: The computed loss and any additional outputs needed for logging or analysis.
        """
        pass

    @abstractmethod
    def apply_gradients(self) -> None:
        """
        Apply accumulated gradients to the model parameters.
        This method should update the model's parameters using the gradients that have
        been accumulated so far (e.g., by calling an optimizer step). After this call,
        the accumulated gradients should be cleared/reset.
        """
        pass

    @abstractmethod
    def snapshot_weights(self) -> WeightsBuffer:
        """
        Save the current model weights and return a buffer handle.
        This method should capture the current state of the model's weights and store
        them in a WeightBuffer (which may be local or remote, depending on the implementation).
        The returned buffer can be used to transfer weights between components or for checkpointing.
        Returns:
            WeightsBuffer: A handle or reference to the stored weights buffer.
        """
        pass


class Generator(ABC):
    """
    Abstract base class for a model generator in RL or sequence modeling workflows.
    This class defines the interface for any generator implementation, which is responsible
    for producing completions (e.g., text, actions) given a prompt, and for updating its
    internal model weights. Subclasses should implement the actual logic for generation
    and weight updates, which may vary depending on the underlying model or framework.
    """

    @abstractmethod
    def generate(self, prompt: Prompt) -> list[Completion]:
        """
        Generate completions given a prompt.
        This method should use the current model to produce one or more completions
        (e.g., text outputs, actions) based on the provided prompt.
        Args:
            prompt (Prompt): The input prompt or context for generation.
        Returns:
            list[Completion]: A list of generated completions corresponding to the prompt.
        """
        pass

    @abstractmethod
    def update_weights(self, weights_handle: WeightsBuffer) -> None:
        """
        Update the weights of the model using the provided weights buffer.
        This method should update the generator's internal model parameters using
        the weights stored in the given WeightsBuffer (which may be local or remote).
        Args:
            weights_handle (WeightsBuffer): A handle or reference to the weights buffer
                                            containing the new model weights.
        """
        pass
