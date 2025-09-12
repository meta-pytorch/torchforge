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
from typing import Callable

from monarch.actor import current_rank, current_size, endpoint
from torch import Tensor
from torchtitan.config.job_config import (
    ActivationCheckpoint,
    Checkpoint,
    Comm,
    Compile,
    Float8,
    LRScheduler,
    Model,
    Optimizer,
    Parallelism,
    Training,
)

from torchtitan.distributed import utils as dist_utils
from torchtitan.experiments.forge.engine import ForgeEngine
from torchtitan.experiments.forge.job_config import ForgeJobConfig

from forge.controller import ForgeActor
from forge.data.utils import batch_to_device

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclass
class RLTrainer(ForgeActor):
    model: Model = field(default_factory=Model)
    optimizer: Optimizer = field(default_factory=Optimizer)
    lr_scheduler: LRScheduler = field(default_factory=LRScheduler)
    training: Training = field(default_factory=Training)
    parallelism: Parallelism = field(default_factory=Parallelism)
    checkpoint: Checkpoint = field(default_factory=Checkpoint)
    activation_checkpoint: ActivationCheckpoint = field(
        default_factory=ActivationCheckpoint
    )
    compile: Compile = field(default_factory=Compile)
    float8: Float8 = field(default_factory=Float8)
    comm: Comm = field(default_factory=Comm)
    loss: Callable = lambda logits, **targets: logits

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

        self.current_step = 0
        self.num_training_steps = self.training.steps
        self.gradient_accumulation_steps = 1
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
        # TODO: update ForgeEngine to not use ForgeJobConfig
        engine_config = {f.name: getattr(self, f.name) for f in fields(self)}
        engine_config.pop("loss")  # Not part of job config
        self.engine = ForgeEngine(ForgeJobConfig(**engine_config))
        self.engine.checkpointer.load(step=self.current_step)
        self.engine.optimizers.zero_grad()

    def forward_backward(self, inputs: dict[Tensor], targets: dict[Tensor]) -> Tensor:
        model_parts = self.engine.model_parts
        parallel_dims = self.engine.parallel_dims

        # apply context parallelism if cp is enabled
        # ensure CP handles the separate freqs_cis buffer for each pp stage
        if getattr(self.engine.model_args, "use_flex_attn", False):
            cp_mesh = (
                parallel_dims.world_mesh["cp"] if parallel_dims.cp_enabled else None
            )
            init_attention_mask(
                inputs, self.engine.tokenizer.base_tokenizer.eos_id, cp_mesh
            )

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
            # TODO implement PP
            # # Pipeline Parallel forward / backward inside step() call
            # with self.train_context(optional_context_parallel_ctx):
            #     targets, losses = (
            #         (labels, []) if self.pp_has_last_stage else (None, None)
            #     )
            #     if self.pp_has_first_stage:
            #         self.pp_schedule.step(
            #             inputs, target=targets, losses=losses, input_batch=inputs
            #         )
            #     else:
            #         self.pp_schedule.step(
            #             target=targets, losses=losses, input_batch=inputs
            #         )
            #
            # # accumulate losses across pipeline microbatches
            # # TODO: PP+FSDP unexpectedly puts the loss back to the CPU
            # loss = (
            #     torch.mean(torch.stack(losses)).to(self.device)
            #     if self.pp_has_last_stage
            #     else torch.tensor([-1.0], device=self.device)
            # )
        else:
            # Non-PP forward / backward
            with self.engine.train_context(optional_context_parallel_ctx):
                assert len(model_parts) == 1
                with self.engine.maybe_enable_amp:
                    logits = model_parts[0](**inputs)
                    loss = self.loss(logits, **targets)
                # need to free to before bwd to avoid peaking memory
                del logits
                loss.backward()

        return loss

    @endpoint
    def train_step(self, inputs: dict[Tensor], targets: dict[Tensor]) -> None:
        batch_to_device(inputs, self.engine.device)
        batch_to_device(targets, self.engine.device)

        # compute policy logprobs
        # TODO implement gradient accumulation
        # with GradientAccumulation(
        #     self.gradient_accumulation_steps,
        #     self.model,
        #     self.data_parallel_size,
        # ) as grad_acc:
        loss = self.forward_backward(inputs, targets)

        # # Gradient clipping (optional but recommended for stability)
        # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        self.engine.optimizers.step()
        self.engine.optimizers.zero_grad()
        self.engine.lr_schedulers.step()

        self.current_step += 1
        self.engine.checkpointer.save(
            curr_step=self.current_step,
            last_step=self.current_step == self.num_training_steps,
        )

        return {"loss": loss.item()}

    @endpoint
    def push_weights(self) -> None:
        pass

    @endpoint
    async def cleanup(self) -> None:
        if self.engine.checkpointer:
            self.engine.checkpointer.close()
