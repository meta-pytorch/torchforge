# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass
class DistributedMetric(ABC):
    """Metrics that are calculated in distributed fashion.

    Metrics computed in each rank are going to be wrapped in DistributedMetric
    according to how they are going to be aggregated. For example, average log prob
    can be wrapped as `Fraction(Sum((logp * mask).sum()), Sum(mask.sum()))` where
    `mask` indicates which token is valid.
    """

    # We need to pass a context argument for distribution setup in the future.
    @abstractmethod
    def reduce(self, group: dist.ProcessGroup | None = None) -> torch.Tensor:
        pass

    @abstractmethod
    def local(self) -> torch.Tensor:
        pass


@dataclass
class SumDistributedMetric(DistributedMetric):
    def __init__(self, tensor: torch.Tensor) -> None:
        self.tensor = tensor

    def reduce(self, group: dist.ProcessGroup | None = None) -> torch.Tensor:
        return _try_clone_and_reduce(self.tensor, op=dist.ReduceOp.SUM, group=group)

    def local(self) -> torch.Tensor:
        return self.tensor


@dataclass
class Fraction:
    numerator: DistributedMetric
    denominator: DistributedMetric

    def reduce(self, group: dist.ProcessGroup | None = None) -> torch.Tensor:
        return self.numerator.reduce(group) / self.denominator.reduce(group)

    def local(self) -> torch.Tensor:
        return self.numerator.local() / self.denominator.local()


def _try_clone_and_reduce(
    tensor: torch.Tensor, op: dist.ReduceOp, group: dist.ProcessGroup | None
) -> torch.Tensor:
    cloned = tensor.detach().clone()
    if dist.is_initialized():
        dist.all_reduce(cloned, op=op, group=group)
    return cloned
