# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import asyncio
import logging
import math
import os
from typing import Any

import torch
import torchtitan.experiments.forge.train_spec as forge_train_spec

# from tqdm import tqdm

from forge.controller import ForgeActor
from monarch.actor import current_rank, current_size, endpoint
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torchstore import MultiProcessStore
from torchstore._state_dict_utils import push_state_dict
from torchtitan.components.loss import LossFunction

# from torchdata.stateful_dataloader import StatefulDataLoader
# from torchtitan.components.checkpoint import ModelWrapper
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.distributed import ParallelDims, utils as dist_utils
from torchtitan.experiments.forge.engine import ForgeEngine
from torchtitan.experiments.forge.job_config import ForgeJobConfig

# from forge.interfaces import RLLoss

# stubs for now
Checkpointer = Any
Dataloader = Any
MetricLogger = Any
Profiler = Any
Tokenizer = Any

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class RLTrainer(ForgeActor, ForgeEngine):
    job_config: ForgeJobConfig
    train_spec: forge_train_spec.ForgeTrainSpec
    parallel_dims: ParallelDims
    model: list[nn.Module]
    loss_fn: LossFunction
    optimizer: OptimizersContainer
    lr_scheduler: LRSchedulersContainer
    checkpointer: Checkpointer
    tokenizer: Tokenizer
    train_dataloader: Dataloader
    # val_dataloader: Dataloader
    profiler: Profiler
    device: torch.device
    step: int

    def __init__(self, config: DictConfig):
        job_config = ForgeJobConfig().to_dict()
        # Hack to deal with literal types from titan
        job_config = OmegaConf.merge(job_config, config)

        self.current_step = 0
        self.num_training_steps = job_config.training.steps
        self.gradient_accumulation_steps = 1  # Example value, adjust as needed
        self._rank = current_rank().rank
        self._size = math.prod(current_size().values())

        # init torchstore
        self._tstore = asyncio.run(MultiProcessStore.create_store())

        self._init_dist()
        super().__init__(job_config)

    def _init_dist(self):
        """Initializes torch distributed.

        torchrun normally hands this, but we need to do it ourselves
        in monarch for now.

        We should consider putting this into ForgeActor, but having this
        be explicit for now.

        """
        env = {
            "RANK": str(self._rank),
            "LOCAL_RANK": str(self._rank),
            "LOCAL_WORLD_SIZE": str(self._size),
            "GROUP_RANK": str(self._size),
            "GROUP_WORLD_SIZE": str(self._size),
            "ROLE_RANK": str(self._rank),
            "ROLE_WORLD_SIZE": str(self._size),
            "ROLE_NAME": "rank",
            "WORLD_SIZE": str(self._size),
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
        os.environ.update(env)
        logger.info("env: {}".format(env))

    @endpoint
    async def setup(self):
        self.checkpointer.load(step=self.current_step)
        # self.profiler = self.setup_profiler(self.train_config.profiler_config)
        # self.logger = self.setup_logger(self.train_config.logger_config)
        self.optimizers.zero_grad()

        # self.pbar = tqdm(
        #     initial=0,
        #     total=self.num_training_steps,
        #     desc=f"{self.current_step}",
        # )
        #

    def forward_backward(
        self, input_dict: dict[str, torch.Tensor], labels: torch.Tensor
    ) -> torch.Tensor:
        model_parts = self.model_parts
        parallel_dims = self.parallel_dims

        # apply context parallelism if cp is enabled
        # ensure CP handles the separate freqs_cis buffer for each pp stage
        inputs = input_dict["tokens"]

        if getattr(self.model_args, "use_flex_attn", False):
            cp_mesh = (
                parallel_dims.world_mesh["cp"] if parallel_dims.cp_enabled else None
            )
            init_attention_mask(inputs, self.tokenizer.base_tokenizer.eos_id, cp_mesh)

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
            with self.train_context(optional_context_parallel_ctx):
                assert len(model_parts) == 1
                with self.maybe_enable_amp:
                    pred = model_parts[0](inputs)
                    loss = self.loss_fn(pred, labels)
                # need to free to before bwd to avoid peaking memory
                del pred
                loss.backward()

        return loss

    @endpoint
    def train_step(self, batch) -> None:
        # Move tensors to the appropriate device
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to("cuda")  # TODO: hardcoded for now

        # TODO implement gradient accumulation
        # with GradientAccumulation(
        #     self.gradient_accumulation_steps,
        #     self.model,
        #     self.data_parallel_size,
        # ) as grad_acc:
        # TODO: convert to GRPO Loss
        labels = batch.pop("labels")
        loss = self.forward_backward(batch, labels)
        # self.pbar.update(1)
        # self.pbar.set_description(f"{self.current_step}|Loss: {loss}")

        self.optimizers.step()
        self.optimizers.zero_grad()
        self.lr_schedulers.step()

        # self.profiler.step()
        self.current_step += 1

        # save to torchstore. Hacking in to the Checkpointer's prepped state-dict for now.
        # TODOs:
        # 1. Figure out if there is a value in calling state_dict_adatpr.to_hf()
        # 2. Checkpoint invokes state-dict flattening during dcp_save for [MODEL].
        #    May need to replicate the same in this code path.
        # 3. Integrate zero-overhead version of push_state_dict.
        # 4. Figure out a way to notify the generator app that weights are ready. This beyond the initial integration success.
        # 5. Unify CheckpointManager and TorchStore weights save control path.
        push_state_dict(self._tstore, self.checkpointer.states, f"v{self.current_step}")
        # if self.current_step % self.train_config.val_every_n_steps == 0:
        #     self.validate()
        self.checkpointer.save(
            curr_step=self.current_step,
            last_step=self.current_step == self.num_training_steps,
        )

    @endpoint
    def push_weights(self) -> None:
        pass

    @endpoint
    async def cleanup(self) -> None:
        # self.pbar.close()
        if self.checkpointer:
            self.checkpointer.close()

    def __repr__(self) -> str:
        return "Trainer"
