# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# import monarch
import torch

from forge.rl.entities import ForgeEnvInfo, ForgeRequest


# Stand-in contract for environments
class ForgeEnvironment:
    def __init__(self, data_loader: torch.utils.data.DataLoader, rewarder: Rewarder):
        self.data_loader = data_loader
        self.rewarder = rewarder

    # obs, info
    def reset(self) -> tuple[ForgeRequest, ForgeEnvInfo]:
        pass

    # obs, rew, term, truncated, info
    def step(self, action) -> tuple[ForgeRequest]:
        pass


class ToyEnvironment(ForgeEnvironment):
    pass


class DeepResearchEnvironment(ForgeEnvironment):
    pass
