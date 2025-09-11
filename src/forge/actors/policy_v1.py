# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os

import torch

# from monarch._src.tensor_engine.rdma import RDMABuffer
from forge.data_models.completion import Completion
from forge.data_models.prompt import Prompt
from forge.generators.vllm_generator import VLLMGenerator
from monarch._src.actor.actor_mesh import Actor, current_rank
from monarch._src.actor.endpoint import endpoint


class Policy(Actor):
    def __init__(self, model_path: str):
        super().__init__()
        rank = current_rank()
        self.rank = rank.rank
        self.local_rank = rank["gpus"]
        self.world_size = rank.extent.nelements
        self._set_env_vars()
        self.sampler = VLLMGenerator(model_path)

    @endpoint
    async def generate(self, prompt: Prompt) -> list[Completion]:
        return self.sampler.generate(prompt)

    @endpoint
    async def update_weights(
        self, weights_buffer: dict[str, tuple[torch.Tensor, torch.dtype, torch.Size]]
    ):
        return self.sampler.update_weights(weights_buffer)

    def _set_env_vars(self):
        os.environ["WORLD_SIZE"] = str(self.world_size)
        os.environ["RANK"] = str(self.rank)
        os.environ["LOCAL_RANK"] = str(self.local_rank)
