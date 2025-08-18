# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import logging
import math
import os

# from functools import partial
from typing import Any

import torch

import torchtitan.experiments.forge.train_spec as forge_train_spec

from monarch.actor import current_rank, current_size, endpoint
from torch import nn

# from torchdata.stateful_dataloader import StatefulDataLoader
# from torchtitan.components.checkpoint import ModelWrapper
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.distributed import ParallelDims, utils as dist_utils
from torchtitan.experiments.forge.engine import ForgeEngine
from torchtitan.experiments.forge.job_config import ForgeJobConfig

# from tqdm import tqdm

from forge.controller import ForgeActor
from forge.data.replay_buffer import ReplayBuffer
from forge.interfaces import RLLoss

# from forge.data.collate import collate_packed
# from forge.data.datasets.packed import PackedDataset, TextPacker
# from forge.data.datasets.sft_dataset import AlpacaToMessages, sft_iterable_dataset
# from forge.data.tokenizer import HuggingFaceModelTokenizer

# stubs for now
Checkpointer = Any
Dataloader = Any
MetricLogger = Any
Profiler = Any
Tokenizer = Any

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# TODO
# 1. test trainer step
# 2. loss types
# 3. compare to cabernet trainer/losses
# 4. cleanup/remove postprocessing and other actor updates


class RLTrainer(ForgeActor, ForgeEngine):
    job_config: ForgeJobConfig
    train_spec: forge_train_spec.ForgeTrainSpec
    parallel_dims: ParallelDims
    model: list[nn.Module]
    loss_fn: RLLoss
    optimizer: OptimizersContainer
    lr_scheduler: LRSchedulersContainer
    checkpointer: Checkpointer
    tokenizer: Tokenizer
    train_dataloader: Dataloader
    # val_dataloader: Dataloader
    metric_logger: MetricLogger
    profiler: Profiler
    device: torch.device
    step: int

    def __init__(self, job_config: ForgeJobConfig):
        self.current_step = 0
        self.num_training_steps = job_config.training.steps
        self.metric_logger = None
        self.gradient_accumulation_steps = 1  # Example value, adjust as needed
        self._rank = current_rank().rank
        self._size = math.prod(current_size().values())
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
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": "12345",
            "RANK": str(self._rank),
            "LOCAL_RANK": str(self._rank),
            "LOCAL_WORLD_SIZE": str(self._size),
            "GROUP_RANK": str(self._size),
            "GROUP_WORLD_SIZE": str(self._size),
            "ROLE_RANK": str(self._rank),
            "ROLE_WORLD_SIZE": str(self._size),
            "ROLE_NAME": "rank",
            "WORLD_SIZE": str(self._size),
            "CUDA_VISIBLE_DEVICES": str(self._rank),
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
        os.environ.update(env)
        logger.info("env: {}".format(env))

    @endpoint
    async def setup(self, replay_buffer: ReplayBuffer):
        self.replay_buffer = replay_buffer
        self.checkpointer.load(step=self.current_step)
        # self.profiler = self.setup_profiler(self.train_config.profiler_config)
        # self.logger = self.setup_logger(self.train_config.logger_config)

    # def setup_data(self):
    #     tokenizer = HuggingFaceModelTokenizer(
    #         tokenizer_json_path=os.path.join(
    #             self.job_config.model.tokenizer_path, "tokenizer.json"
    #         ),
    #         tokenizer_config_json_path=os.path.join(
    #             self.job_config.model.tokenizer_path, "tokenizer_config.json"
    #         ),
    #         generation_config_path=os.path.join(
    #             self.job_config.model.tokenizer_path, "generation_config.json"
    #         ),
    #     )
    #
    #     dataset = sft_iterable_dataset(
    #         model_transform=tokenizer,
    #         message_transform=AlpacaToMessages(),
    #         path="yahma/alpaca-cleaned",
    #         split="train",
    #     )
    #     packer = TextPacker(padding_idx=0)
    #     dataset = PackedDataset(
    #         dataset=dataset,
    #         packer=packer,
    #         target_tokens_per_pack=self.job_config.training.seq_len,  # TODO: get this from model
    #     )
    #     dataloader = StatefulDataLoader(
    #         dataset=dataset,
    #         batch_size=self.job_config.training.local_batch_size,
    #         collate_fn=partial(
    #             collate_packed, mask_fn=packer.create_block_mask, device=self.device
    #         ),
    #     )
    #
    #     # Ultimately we probably want something like this
    #     # packer = build_packing_strategy(packing_config)
    #     # dataset = build_dataset(dataset_config)
    #     # dataloader = build_dataloader(dataloader_config, dataset, packer)
    #     return dataloader

    def forward_backward(
        self, input_dict: dict[str, torch.Tensor], labels: torch.Tensor
    ) -> torch.Tensor:
        model_parts = self.model_parts
        parallel_dims = self.parallel_dims

        # apply context parallelism if cp is enabled
        # ensure CP handles the separate freqs_cis buffer for each pp stage
        inputs = input_dict["tokens"]
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

    def train_step(self, batch) -> None:
        # TODO
        # with GradientAccumulation(
        #     self.gradient_accumulation_steps,
        #     self.model,
        #     self.data_parallel_size,
        # ) as grad_acc:
        labels = batch.pop("labels")
        loss = self.forward_backward(batch, labels)
        # self.pbar.update(1)
        # self.pbar.set_description(f"{self.current_step}|Loss: {loss}")

        self.optimizers.step()
        self.lr_schedulers.step()

    def push_weights(self) -> None:
        pass

    @endpoint
    async def train(self) -> None:
        # dataloader = iter(self.train_dataloader)
        self.optimizers.zero_grad()

        # self.pbar = tqdm(
        #     initial=0,
        #     total=self.num_training_steps,
        #     desc=f"{self.current_step}",
        # )

        while self.current_step < self.num_training_steps:
            logger.info(f"step: {self.current_step}/{self.num_training_steps}")
            batch = self.replay_buffer.sample()

            # Move tensors to the appropriate device
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to("cuda")  # TODO: hardcoded for now
            self.train_step(batch)
            # self.profiler.step()
            self.current_step += 1

            # if self.current_step % self.train_config.val_every_n_steps == 0:
            #     self.validate()
            self.checkpointer.save(
                curr_step=self.current_step,
                last_step=self.current_step == self.num_training_steps,
            )

        # self.pbar.close()

    @endpoint
    async def cleanup(self) -> None:
        if self.checkpointer:
            self.checkpointer.close()
        if self.metric_logger:
            self.metric_logger.close()

    def __repr__(self) -> str:
        return "Trainer"


async def _test(config, guided_decoding=False):
    # TODO: Create proper test
    trainer_mesh = await proc_mesh(
        gpus=config["resources"],
        env={
            "MASTER_ADDR": str(get_loopback_ip()),
            "MASTER_PORT": str(get_open_port()),
        },
    )

    policy_actor = await policy_mesh.spawn("policy", Policy, **config)

    await policy_actor.setup.call()
    await router.setup.call()
    print("Model setup")

    router.run.call()
    print("Model running")

    prompt = "What is 3+5?" if guided_decoding else "Tell me a joke"
    response = await router.generate.call_one(prompt)
    print(f"User: {prompt}\nAssistant: {response.outputs[0].text}")

    await router.shutdown.call()


if __name__ == "__main__":
    config = {
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "tensor_parallel_size": 2,
        "pipeline_parallel_size": 1,
        "enforce_eager": True,
        "resources": 2,
    }
    asyncio.run(_test(config))
