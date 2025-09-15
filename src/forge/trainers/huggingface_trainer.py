# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

from forge.data_models.api import Trainer, WeightsBuffer

from forge.data_models.distributed_metric import Fraction, SumDistributedMetric

from forge.data_models.loss import LossOutput
from forge.data_models.minibatch import Minibatch
from forge.stores.in_memory_store import InMemoryStore


class HuggingFaceTrainer(Trainer):
    def __init__(self, model_path: str):
        # TODO: model_path and other trainer related configs should be passed in as a config object
        super().__init__()
        self.model_name = model_path

        # TODO: Harded coded implementation for RFC. this needs to be injected via config
        self._store = InMemoryStore()
        self._weights_buffer = WeightsBuffer(self._store)

    def accummulate_gradients(self, minibatch: Minibatch) -> LossOutput:
        """
        Accumulate gradients for the given minibatch.
        """
        return LossOutput(
            loss=Fraction(
                SumDistributedMetric(torch.Tensor(1)), SumDistributedMetric(1.0)
            )
        )

    def apply_gradients(self) -> None:
        """
        Apply accumulated gradients to the model parameters.
        """
        pass

    def snapshot_weights(
        self,
    ) -> WeightsBuffer:
        """
        Save the current model weights using the provided WeightBuffer.
        Args:
            buffer (WeightBuffer): The buffer abstraction to use for storing weights.
        Returns:
            WeightsBuffer: A remote handle to the buffered weights.
        """
        return self._weights_buffer
