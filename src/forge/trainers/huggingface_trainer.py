# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, Tuple

import torch
from forge.data_models.api import Trainer

from forge.data_models.distributed_metric import Fraction, SumDistributedMetric

from forge.data_models.loss import LossOutput
from forge.data_models.minibatch import Minibatch


class HuggingFaceTrainer(Trainer):
    def __init__(self, model_path: str):
        super().__init__()
        self.model_name = model_path

    def accummulate_gradients(self, minibatch: Minibatch) -> LossOutput:
        return LossOutput(
            loss=Fraction(
                SumDistributedMetric(torch.Tensor(1)), SumDistributedMetric(1.0)
            )
        )

    def apply_gradients(self) -> None:
        pass

    def snapshot_weights(
        self,
    ) -> Dict[str, Tuple[torch.Tensor, torch.dtype, torch.Size]]:
        # TODO: NEEDS fixing: the weights should be remote handle, like RDMA Buffer handle
        return {}
