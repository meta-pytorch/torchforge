# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

import torch
from forge.data_models.distributed_metric import Fraction
from forge.data_models.minibatch import Minibatch


@dataclass
class LossInput:
    minibatch: Minibatch
    trainer_logits: torch.Tensor


@dataclass
class LossOutput:
    loss: Fraction
