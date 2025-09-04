# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

__all__ = ["Policy", "PolicyRouter", "RLTrainer", "ReplayBuffer", "TitanRefModel"]


def __getattr__(name):
    if name == "Policy":
        from .policy import Policy

        return Policy
    elif name == "PolicyRouter":
        from .policy import PolicyRouter

        return PolicyRouter
    elif name == "RLTrainer":
        from .trainer import RLTrainer

        return RLTrainer
    elif name == "ReplayBuffer":
        from .replay_buffer import ReplayBuffer

        return ReplayBuffer
    elif name == "TitanRefModel":
        from .reference_service import TitanRefModel

        return TitanRefModel
    else:
        raise AttributeError(f"module {__name__} has no attribute {name}")
