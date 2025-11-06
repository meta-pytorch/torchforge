# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import warnings

__all__ = [
    "Generator",
    "TitanTrainer",
    "RLTrainer",  # Deprecated, use TitanTrainer
    "ReplayBuffer",
    "ReferenceModel",
    "SandboxedPythonCoder",
]


def __getattr__(name):
    if name == "Generator":
        from .generator import Generator

        return Generator
    elif name == "TitanTrainer":
        from .trainer import TitanTrainer

        return TitanTrainer
    elif name == "RLTrainer":
        warnings.warn(
            "RLTrainer is deprecated and will be removed in a future version. "
            "Please use TitanTrainer instead.",
            FutureWarning,
            stacklevel=2,
        )
        from .trainer import RLTrainer

        return RLTrainer
    elif name == "ReplayBuffer":
        from .replay_buffer import ReplayBuffer

        return ReplayBuffer
    elif name == "ReferenceModel":
        from .reference_model import ReferenceModel

        return ReferenceModel
    elif name == "SandboxedPythonCoder":
        from .coder import SandboxedPythonCoder

        return SandboxedPythonCoder
    else:
        raise AttributeError(f"module {__name__} has no attribute {name}")
