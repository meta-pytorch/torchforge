# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from dataclasses import dataclass

import torch
from monarch.actor_mesh import Actor, endpoint


@dataclass
class ForgeRequest:
    pass


@dataclass
class ForgeTrajectory:
    pass


@dataclass
class ForgeEnvInfo:
    pass


# ==== Generic RL entities ====
class Trainer(Actor):
    def step(self):
        pass


class Policy(Actor):
    @endpoint
    async def generate(self, request: ForgeRequest) -> ForgeRequest:
        # e.g. run vLLM
        pass

    @endpoint
    async def update_weights(self):
        """Updates the weights of the policy."""
        pass


class EnvironmentTransformActor(Actor):
    @endpoint
    async def reward(self, request: ForgeRequest) -> float:
        # e.g. run reward model
        pass


class ReplayBuffer(Actor):
    def __init__(self):
        pass

    @endpoint
    async def extend(self, sample: ForgeTrajectory):
        pass

    @endpoint
    async def sample(self):
        pass

    @endpoint
    async def len(self) -> int:
        pass

    @endpoint
    async def is_empty(self) -> bool:
        pass


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
