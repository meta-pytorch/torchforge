# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class WeightsHandleType(Enum):
    SIMPLE = 1  # passing state_dict directly
    TORCH_STORE = 2  # using torchstore


@dataclass
class WeightsHandle:
    """Handle for weights to be transferred between policy and trainer."""

    handle_type: WeightsHandleType = field(default=WeightsHandleType.SIMPLE)
    version: int = field(default=0)
    data: Any = field(default_factory=dict)
