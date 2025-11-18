# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import warnings

from .titan import TitanTrainer

__all__ = ["TitanTrainer", "RLTrainer"]


def __getattr__(name):
    if name == "RLTrainer":
        warnings.warn(
            "RLTrainer is deprecated and will be removed in a future version. "
            "Please use TitanTrainer instead.",
            FutureWarning,
            stacklevel=2,
        )
        return TitanTrainer
    raise AttributeError(f"module {__name__} has no attribute {name}")
