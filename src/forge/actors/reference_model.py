# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import math
import os

from collections.abc import Mapping
from dataclasses import dataclass, field, fields

import torch
from monarch.actor import current_rank, current_size, endpoint
from torch.distributed.tensor import DTensor

from torchtitan.config.job_config import Checkpoint, Compile, Model, Parallelism
from torchtitan.experiments.forge.engine import ForgeEngine
from torchtitan.experiments.forge.job_config import ForgeJobConfig

from forge.controller import ForgeActor
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


@dataclass
class ReferenceModel(ForgeActor):
    # Refer to titan JobConfig for enabling more ForgeEngine configuration
    model: Model = field(default_factory=Model)
    parallelism: Parallelism = field(default_factory=Parallelism)
    checkpoint: Checkpoint = field(default_factory=Checkpoint)
    compile: Compile = field(default_factory=Compile)

    # Populated in setup
    # TODO: Commented out since engine_config parsing extracts from class members
    # engine: ForgeEngine | None = None

    def __post_init__(self):
        """Initializes config types and env variables."""
        # Instantiate dict fields
        super().__init__()
        for f in fields(self):
            attr = getattr(self, f.name)
            if isinstance(attr, Mapping):
                setattr(self, f.name, f.type(**attr))
            elif not isinstance(attr, f.type):
                raise TypeError(
                    f"{f.name} should be a {f.type} type or a dict like object"
                )

        """
        torchrun normally hands env variables, but we need to do it ourselves
        in monarch for now.
        """
        self.rank = current_rank().rank
        self.size = math.prod(current_size().values())
        print(f"Setting rank to {self.rank}, size to {self.size}")

        env = {
            "RANK": str(self.rank),
            "LOCAL_RANK": str(self.rank),
            "LOCAL_WORLD_SIZE": str(self.size),
            "GROUP_RANK": str(self.size),
            "GROUP_WORLD_SIZE": str(self.size),
            "ROLE_RANK": str(self.rank),
            "ROLE_WORLD_SIZE": str(self.size),
            "ROLE_NAME": "rank",
            "WORLD_SIZE": str(self.size),
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
        os.environ.update(env)

    @endpoint
    async def setup(self):
        logger.info("Setting up reference model...")
        print("Setting up reference model...")
        engine_config = {f.name: getattr(self, f.name) for f in fields(self)}
        self.engine = ForgeEngine(ForgeJobConfig(**engine_config))
        logger.info("Done setting up reference model")
        print("Done setting up reference model")

    @endpoint
    async def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        model_parts = self.engine.model_parts
        parallel_dims = self.engine.parallel_dims
        with torch.no_grad():
            input_ids = input_ids.to("cuda")
            # optional_context_parallel_ctx = (
            #     dist_utils.create_context_parallel_ctx(
            #         cp_mesh=parallel_dims.world_mesh["cp"],
            #         cp_buffers=[inputs, labels] + [m.freqs_cis for m in model_parts],
            #         cp_seq_dims=[1, 1] + [0 for _ in model_parts],
            #         cp_no_restore_buffers={inputs, labels},
            #         cp_rotate_method=self.job_config.parallelism.context_parallel_rotate_method,
            #     )
            #     if parallel_dims.cp_enabled
            #     else None
            # )
            optional_context_parallel_ctx = None
            if parallel_dims.pp_enabled:
                raise NotImplementedError("PP not implemented yet")
            else:
                # (jackkhuu) Not sure if either context are needed for inference here
                with self.engine.train_context(optional_context_parallel_ctx):
                    with self.engine.maybe_enable_amp:
                        logits = model_parts[0](input_ids)
            if isinstance(logits, DTensor):
                logits = logits.full_tensor()
            return logits
