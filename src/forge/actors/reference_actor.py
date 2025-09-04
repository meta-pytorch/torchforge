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

from forge.controller import ForgeActor
from monarch.actor import current_rank, current_size, endpoint
from omegaconf import DictConfig, OmegaConf
from torch import nn

from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.config.job_config import Comm, Model, Parallelism
from torchtitan.distributed import ParallelDims, utils as dist_utils
from torchtitan.experiments.forge.engine import ForgeEngine
from torchtitan.experiments.forge.job_config import ForgeJobConfig
from transformers import AutoModelForCausalLM


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclass
class TitanRefModel(ForgeActor):
    """
    Represents a reference actor leveraging a torchtitan model for execution

    Intended for generating reference_logprobs - for example in KL Divergence
    """

    # Refer to titan JobConfig for enabling more ForgeEngine configuration
    model: Model = field(default_factory=Model)
    parallelism: Parallelism = field(default_factory=Parallelism)

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
        token_ids

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
                    # Titan Tranformer
                    logits = model_parts[0](input_ids)

                    # Compute logprobs
                    input_ids = input_ids[:, len(request) :]
                    logprobs = compute_logprobs(logits, input_ids)

                    return logprobs

        return pred


# Based on torchtune's grpo
def compute_logprobs(
    logits: torch.Tensor, input_ids: torch.Tensor, temperature: float = 1.0
) -> torch.Tensor:
    context_length = logits.shape[1] - input_ids.shape[1]

    # Truncate request logits and drop last
    logits = logits[:, context_length - 1 : -1]

    # Compute logprobs
    logprobs = torch.log_softmax(logits / temperature, dim=-1)
    logprobs = torch.gather(logprobs, 2, input_ids.unsqueeze(-1)).squeeze(-1)

    return logprobs


# Maintained to keep Old GRPO app prior to full migration off of HF
class HuggingFaceRefModel(ForgeActor):
    """
    Represents a reference actor leveraging HuggingFace for execution
    """

    def __init__(self, model_name, device: torch.device | None = None):
        super().__init__()
        self.model_name = model_name

        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        # Initialize model and tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).to(self.device)

        # Set model to eval mode for reference computations
        self.model.eval()

        self.logger.info(f"Model initialized on {self.device}")

    @endpoint
    async def forward(self, token_ids: list[int]) -> torch.Tensor:
        # Use provided token_ids directly
        input_ids = (
            torch.tensor(token_ids, dtype=torch.long).unsqueeze(0).to(self.device)
        )
        # Create attention mask of all 1s since we have actual tokens (no padding)
        attention_mask = torch.ones_like(input_ids).to(self.device)

        # Compute log probabilities using shared utility function
        sequence_log_probs = compute_sequence_logprobs(
            self.model, input_ids, attention_mask, requires_grad=False
        )

        return (
            sequence_log_probs.squeeze()
        )  # Remove batch dimension for single response


def compute_sequence_logprobs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    requires_grad: bool = True,
) -> torch.Tensor:
    context_manager = torch.enable_grad() if requires_grad else torch.no_grad()

    with context_manager:
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        # Apply log softmax to get log probabilities
        log_probs = torch.log_softmax(logits, dim=-1)

        # Extract log probabilities for the actual tokens (excluding the first token for next-token prediction)
        shifted_input_ids = input_ids[:, 1:]  # Remove first token
        shifted_log_probs = log_probs[:, :-1, :]  # Remove last logit

        # Gather log probabilities for actual tokens
        token_log_probs = torch.gather(
            shifted_log_probs, dim=-1, index=shifted_input_ids.unsqueeze(-1)
        ).squeeze(-1)

        # Sum log probabilities across sequence (masked by attention)
        shifted_attention_mask = attention_mask[:, 1:]
        sequence_log_probs = (token_log_probs * shifted_attention_mask).sum(dim=-1)

        return sequence_log_probs


"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Experimental: DO NOT USE (YET)

ReferenceActor: Coordinate requests to reference models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""


@dataclass
class ReferenceActor(ForgeActor):
    """
    DO NOT USE (YET)

    Not updated/used; Original plan was to use this for coordination, but
    it might be overkil if we can rely on the Service Replicas to handle
    the queue.
    We MAY need to still do this for DP and batching support

    For now if you think you need this: directly spin up services of the
    reference models
    """

    model: Model = field(default_factory=Model)
    # parallelism: Parallelism = field(default_factory=Parallelism)
    # comm: Comm = field(default_factory=Comm)

    # For RefModel
    ref_model: ForgeActor | None = None
    device: torch.device | None = None

    # For processing
    running: bool = False
    queue: deque | None = None

    def __post_init__(self):
        """Initializes config types and env variables.

        torchrun normally hands env variables, but we need to do it ourselves
        in monarch for now.

        """
        # Instantiate dict fields
        for f in fields(self):
            attr = getattr(self, f.name)
            if isinstance(attr, Mapping):
                setattr(self, f.name, f.type(**attr))
            elif not isinstance(attr, f.type):
                raise TypeError(
                    f"{f.name} should be a {f.type} type or a dict like object"
                )

        # This might need to be changed to a distributed friendly container
        # We also don't have a traditional scheduler?
        self.queue = deque()

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

        # Spawn the RefModel
        self.ref_model = await spawn_service(
            default_service_cfg,
            HuggingFaceRefModel,
            model_name=self.model.name,
            device=self.device,
        )

        # Kick off background processing
        self.start_processing()

    def start_processing(self):
        """Start the replica's processing loop if not already running."""
        if self._run_task is None or self._run_task.done():
            self._run_task = asyncio.create_task(self.run())

    @endpoint
    async def forward(self, token_ids: list[int]) -> torch.Tensor:
        """
        Enque the tokens and await response
        """
        fut = asyncio.Future()
        self.queue.append((token_ids, fut))
        return await fut

    async def run(self):
        """
        Simple loop to pass things along to the ref model
        """

        # TODO: Consider creating a unified base class for this pattern
        self.running = True

        while self.running:
            request, fut = self.queue.popleft()
            model_output = await self.ref_model.forward(request)
            fut.set_result(model_output)

    @endpoint
    async def stop(self) -> None:
        self.running = False
