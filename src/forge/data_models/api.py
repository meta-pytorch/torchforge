# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from typing import Dict, Sequence, Tuple

import torch

from forge.data_models.completion import Completion
from forge.data_models.loss import LossOutput
from forge.data_models.minibatch import Minibatch

from forge.data_models.prompt import Prompt
from forge.data_models.scored_completion import ScoredCompletion


# TODO: This file needs should not be in the data_models folder/package


class Trainer(ABC):
    @abstractmethod
    def accummulate_gradients(self, minibatch: Minibatch) -> LossOutput:
        """
        accummulate_gradients is called once per minibatch.
        """
        pass

    @abstractmethod
    def apply_gradients(self) -> None:
        """
        applying accumulated gradients to the model parameters.
        """
        pass

    @abstractmethod
    def snapshot_weights(
        self,
    ) -> Dict[str, Tuple[torch.Tensor, torch.dtype, torch.Size]]:  # TODO: RDMA buffer
        """
        applying accumulated gradients to the model parameters.

        the return type is a tuple of weights buffer, dtype, and shape of the original tensor.
        """
        # TODO: NEEDS fixing: the weights_handle should be remote handle, like RDMA Buffer handle
        pass


class Generator(ABC):
    @abstractmethod
    def generate(self, prompt: Prompt, **kwargs) -> list[Completion]:
        """
        Generate a completion given a prompt.
        Args:
            prompt: The input prompt.
            **kwargs: Additional model-specific generation parameters.
        Returns:
            str: The generated text.
        """
        pass

    @abstractmethod
    def update_weights(
        self, weights_handle: dict[str, tuple[torch.Tensor, torch.dtype, torch.Size]]
    ):
        """
        Update the weights of the model.
        Args:
            weights: A dictionary of weights to update.
        """
        # TODO: NEEDS fixing: the weights_handle should be remote handle, like RDMA Buffer handle
        pass


class Scorer(ABC):
    @abstractmethod
    def score(self, completion: Completion) -> ScoredCompletion:
        pass

    def score_batch(
        self, completions: Sequence[Completion]
    ) -> Sequence[ScoredCompletion]:
        """
        Optionally override for efficient batch scoring.
        """
        return [self.score(c) for c in completions]
