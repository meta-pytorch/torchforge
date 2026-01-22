# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Literal

import torch
from forge.observability.metrics import Metric
from pydantic import BaseModel, ConfigDict

AggType = Literal["token_mean", "fixed_horizon", "sequence_mean"]
RatioType = Literal["token", "sequence"]
KLType = Literal["k1", "k2", "k3"]


@dataclass
class LossOutput:
    """Output from all loss functions.

    Attributes:
        loss (torch.Tensor): Scalar loss tensor for backpropagation.
        metrics (list[Metric]): List of Metric objects for distributed logging.
    """

    loss: torch.Tensor
    metrics: list[Metric]


class BaseLossConfig(BaseModel):
    """Base configuration for all policy gradient losses."""

    # Pydantic v2 configuration
    model_config = ConfigDict(
        extra="forbid",  # Raises error on unknown fields (catches typos)
        arbitrary_types_allowed=True,  # Allows torch.Tensor and other non-JSON types
    )
