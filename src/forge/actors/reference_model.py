# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import asyncio
import logging
import math
import os

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field, fields

from typing import Any

import torch

from forge.actors.trainer import compute_logprobs, Episode
from forge.controller import ForgeActor
from monarch.actor import current_rank, current_size, endpoint
from omegaconf import DictConfig, OmegaConf
from torch import nn

from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.config.job_config import Checkpoint, Comm, Compile, Model, Parallelism
from torchtitan.distributed import ParallelDims, utils as dist_utils
from torchtitan.experiments.forge.engine import ForgeEngine
from torchtitan.experiments.forge.job_config import ForgeJobConfig
from transformers import AutoModelForCausalLM


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclass
class ReferenceModel(ForgeActor):
    """
    Represents a reference actor leveraging a torchtitan model for execution

    Intended for generating reference_logprobs - for example in KL Divergence
    """

    # Refer to titan JobConfig for enabling more ForgeEngine configuration
    model: Model = field(default_factory=Model)
    checkpoint: Checkpoint = field(default_factory=Checkpoint)
    parallelism: Parallelism = field(default_factory=Parallelism)
    compile: Compile = field(default_factory=Compile)

    # Populated in setup
    # TODO: Commented out since engine_config parsing extracts from class members
    # engine: ForgeEngine | None = None

    def __post_init__(self):
        """Initializes config types and env variables."""
        # Instantiate dict fields
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
        engine_config = {f.name: getattr(self, f.name) for f in fields(self)}
        self.engine = ForgeEngine(ForgeJobConfig(**engine_config))

    @endpoint
    async def forward(self, request: list[int], response: list[int]) -> torch.Tensor:
        """
        Given a request and response tokens, return the log_probability of the
        token_ids, shape (completion_len, )

        """
        model_parts = self.engine.model_parts
        parallel_dims = self.engine.parallel_dims

        # Use provided token_ids directly
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_ids = torch.tensor(
            request + response, dtype=torch.long, device=device
        ).unsqueeze(0)

        optional_context_parallel_ctx = (
            dist_utils.create_context_parallel_ctx(
                cp_mesh=parallel_dims.world_mesh["cp"],
                cp_buffers=[inputs, labels] + [m.freqs_cis for m in model_parts],
                cp_seq_dims=[1, 1] + [0 for _ in model_parts],
                cp_no_restore_buffers={inputs, labels},
                cp_rotate_method=self.job_config.parallelism.context_parallel_rotate_method,
            )
            if parallel_dims.cp_enabled
            else None
        )

        if parallel_dims.pp_enabled:
            raise NotImplementedError("PP not implemented yet")
        else:
            # (jackkhuu) Not sure if either context are needed for inference here
            with self.engine.train_context(optional_context_parallel_ctx):
                assert len(model_parts) == 1
                with self.engine.maybe_enable_amp:
                    logits = model_parts[0](input_ids)

                    # Compute logprobs
                    input_ids = input_ids[:, len(request) :]
                    # (bsz=1, completion_len)
                    logprobs = compute_logprobs(logits, input_ids)
                    # (completion_len, )
                    return logprobs.squeeze(0)

        return pred


class HFReferenceModel(ForgeActor):
    def __init__(self, model_name, device: torch.device | None = None):
        super().__init__()
        self.model_name = model_name

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            trust_remote_code=True,
        ).to(self.device)
        self.model.eval()

        self.logger.info(f"Model initialized on {self.device}")

    @endpoint
    async def forward(self, episode: Episode) -> torch.Tensor:
        req, res = episode.request_tensor, episode.response_tensor
        input_ids = torch.cat([req, res]).to(self.device).unsqueeze(0)
        mask = input_ids != episode.pad_id

        with torch.inference_mode():
            logits = self.model(input_ids=input_ids, attention_mask=mask).logits

        input_ids = input_ids[:, len(req) :]
        return compute_logprobs(logits, input_ids)
