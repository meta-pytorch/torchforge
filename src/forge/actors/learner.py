# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os

import torch

from forge.data_models.minibatch import Minibatch
from forge.trainers.huggingface_trainer import HuggingFaceTrainer
from monarch._src.actor.actor_mesh import Actor, current_rank
from monarch._src.actor.endpoint import endpoint


class Learner(Actor):
    def __init__(self, model_path: str):
        super().__init__()
        rank = current_rank()
        self.rank = rank.rank
        self.local_rank = rank["gpus"]
        self.world_size = rank.extent.nelements
        self._set_env_vars()
        self.trainer = HuggingFaceTrainer(model_path)

    @endpoint
    async def accummulate_gradients(self, minibatch: Minibatch):
        return self.trainer.accummulate_gradients(minibatch)

    @endpoint
    async def apply_gradients(self) -> None:
        return self.trainer.apply_gradients()

    @endpoint
    async def snapshot_weights(
        self,
    ) -> dict[str, tuple[torch.Tensor, torch.dtype, torch.Size]]:
        return self.trainer.snapshot_weights()

    def _set_env_vars(self):
        os.environ["WORLD_SIZE"] = str(self.world_size)
        os.environ["RANK"] = str(self.rank)
        os.environ["LOCAL_RANK"] = str(self.local_rank)
