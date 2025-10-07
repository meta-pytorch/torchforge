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
from torch.distributed.tensor import DTensor

from torchtitan.config.job_config import (
    Checkpoint,
    Comm,
    Compile,
    Model,
    Parallelism,
    Training,
)
from torchtitan.experiments.forge.engine import ForgeEngine
from torchtitan.experiments.forge.job_config import ForgeJobConfig

from forge.controller import ForgeActor
from forge.observability.metrics import record_metric, Reduce
from forge.observability.perf_tracker import Tracer
from forge.util.ops import compute_logprobs

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclass
class ReferenceModel(ForgeActor):
    # Refer to titan JobConfig for enabling more ForgeEngine configuration
    model: Model = field(default_factory=Model)
    parallelism: Parallelism = field(default_factory=Parallelism)
    checkpoint: Checkpoint = field(default_factory=Checkpoint)
    compile: Compile = field(default_factory=Compile)
    comm: Comm = field(default_factory=Comm)
    training: Training = field(
        default_factory=Training
    )  # Needed in order to set attrs like dtype, garbage collection freq, etc.

    # Populated in setup
    # TODO: Commented out since engine_config parsing extracts from class members
    # engine: ForgeEngine | None = None

    def __post_init__(self):
        """Initializes config types and env variables."""
        super().__init__()

        # Instantiate dict fields
        for f in fields(self):
            attr = getattr(self, f.name)
            if isinstance(attr, Mapping):
                setattr(self, f.name, f.type(**attr))
            elif not isinstance(attr, f.type):
                raise TypeError(
                    f"{f.name} should be a {f.type} type or a dict like object"
                )

        self.step = 0
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
        self.engine.checkpointer.load()
        self.model = self.engine.model_parts[0]  # No pipeline parallelism yet
        self.model.eval()

    @endpoint
    async def forward(
        self, input_ids: torch.Tensor, max_req_tokens: int, return_logprobs: bool
    ) -> torch.Tensor:
        """
        Args:
            input_ids (torch.Tensor): input token ids with shape [group_size, req + res length].
            max_req_tokens (int): maximum request length.
            return_logprobs (bool): whether to return og probabilities instead of raw logits.

            return_logprobs flag significantly impacts the amount of data transferred to the caller:
            - When False: Returns logits with shape [group_size, req + res_length, vocab_size].
              This includes the full vocabulary distribution for each token position.

            - When True: Returns log probabilities with shape [group_size, req_length].
              This only includes probabilities for the request tokens, significantly reducing memory
              usage and transfer overhead.
        """
        # Record reference model metrics
        record_metric("reference_perf/forward/count_forward_passes", 1, Reduce.SUM)
        record_metric(
            "reference_perf/forward/avg_sequence_length",
            input_ids.shape[1],
            Reduce.MEAN,
        )

        t = Tracer("reference_perf/forward", timer="gpu", track_memory=True)
        t.start()
        self.engine.gc_handler.run(self.step)
        t.step("garbage_collection")

        model_parts = self.engine.model_parts
        parallel_dims = self.engine.parallel_dims
        input_ids = input_ids.to("cuda")
        t.step("to_device")
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
        if self.engine.parallel_dims.pp_enabled:
            raise NotImplementedError("PP not implemented yet")
        else:
            # (jackkhuu) Not sure if either context are needed for inference here
            with self.engine.train_context(optional_context_parallel_ctx):
                with self.engine.maybe_enable_amp:
                    with torch.inference_mode():
                        logits = self.model(input_ids)
        self.step += 1
        if isinstance(logits, DTensor):
            logits = logits.full_tensor()
        t.step("forward")

        if not return_logprobs:
            t.stop()
            return logits
        else:
            logprobs = compute_logprobs(logits, input_ids[:, max_req_tokens:])
            t.step("compute_logprobs")
            t.stop()
            return logprobs
