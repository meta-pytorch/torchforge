# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import logging
import math
import os

from collections.abc import Mapping
from dataclasses import dataclass, field, fields

import torch
from monarch.actor import current_rank, current_size, endpoint

from torchtitan.config.job_config import Compile, Checkpoint, Model, Parallelism
from torchtitan.distributed import utils as dist_utils
from torchtitan.experiments.forge.engine import ForgeEngine
from torchtitan.experiments.forge.job_config import ForgeJobConfig

from forge.controller import ForgeActor


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
    parallelism: Parallelism = field(default_factory=Parallelism)
    checkpoint: Checkpoint = field(default_factory=Checkpoint)
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
    async def forward(self, episode: 'Episode') -> torch.Tensor:
        """
        Given an episode, return the log_probability of the
        token_ids, shape (completion_len, )

        """
        req, res = episode.request_tensor, episode.response_tensor
        model_parts = self.engine.model_parts
        parallel_dims = self.engine.parallel_dims

        # Use provided token_ids directly
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # input_ids = torch.tensor(
        #     request + response, dtype=torch.long, device=device
        # ).unsqueeze(0)
        input_ids = torch.cat([req, res]).to(device).unsqueeze(0)

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
                    # Titan Tranformer
                    logits = model_parts[0](input_ids)

                    # Compute logprobs
                    input_ids = input_ids[:, len(req) :]
                    # (bsz=1, completion_len)
                    logprobs = compute_logprobs(logits, input_ids)
                    # (completion_len, )
                    return logprobs.squeeze(0)

        return pred


# Based on torchtune's grpo
def compute_logprobs(
    logits: torch.Tensor, input_ids: torch.Tensor, temperature: float = 1.0
) -> torch.Tensor:
    """
    Compute log probs of the completion input_ids given the logits of the whole sequence.
    Warning: only works if all prompts in the batch have the same length. TODO: support variable length prompts.

    Args:
        logits (torch.Tensor): (batch_size, seq_len, vocab_size), the logits output from the model.
        input_ids (torch.Tensor): (batch_size, completion_len), the token ids for the completion.

    Returns:
        torch.Tensor: (batch_size, completion_len), the log probabilities of the completion tokens.

    Raises:
        ValueError: If the inferred context length is less than or equal to 0.
    """
    context_len = logits.shape[1] - input_ids.shape[1]
    completion_len = input_ids.shape[1]
    if context_len <= 0:
        raise ValueError(
            "Context length must be greater than 0. Otherwise the probability of the first token is undefined."
        )

    # (bsz, completion_len, vocab_size)
    logits = logits[:, context_len - 1 : -1, :]
    assert logits.shape == (
        input_ids.shape[0],
        completion_len,
        logits.shape[-1],
    ), f"logits shape incorrect, {logits.shape=}, {input_ids.shape=}, {logits.shape[-1]=}"
    token_logprobs = torch.log_softmax(logits / temperature, dim=-1)
    # (bsz, completion_len, 1)
    logprobs = torch.gather(token_logprobs, 2, input_ids.unsqueeze(-1))
    # (bsz, completion_len)
    logprobs = logprobs.squeeze(-1)

    return logprobs
