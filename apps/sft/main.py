# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""To run:

python -m apps.sft.main --config apps/sft/llama3_8b.yaml

"""

import asyncio
import os
import sys
import time
from dataclasses import asdict
from functools import partial
from typing import Any

import torch

import torchtitan.experiments.forge.train_spec as forge_train_spec
from forge.cli.config import parse
from forge.data.collate import collate_packed
from forge.data.datasets.packed import PackedDataset, TextPacker
from forge.data.datasets.sft_dataset import AlpacaToMessages, sft_iterable_dataset
from forge.data.tokenizer import HuggingFaceModelTokenizer

from monarch.actor import Actor, current_rank, current_size, endpoint, proc_mesh

from omegaconf import DictConfig, OmegaConf
from torch import nn
from torchdata.stateful_dataloader import StatefulDataLoader
from torchtitan.components.checkpoint import ModelWrapper
from torchtitan.components.loss import LossFunction
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.distributed import ParallelDims, utils as dist_utils
from torchtitan.experiments.forge.engine import ForgeEngine
from torchtitan.experiments.forge.job_config import ForgeJobConfig
from tqdm import tqdm

# stubs for now
Checkpointer = Any
Dataloader = Any
MetricLogger = Any
Profiler = Any
Tokenizer = Any


class ForgeSFTRecipe(Actor, ForgeEngine):
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
    metric_logger: MetricLogger
    profiler: Profiler
    device: torch.device
    step: int

    def __init__(self, job_config: ForgeJobConfig):
        self.current_step = 0
        self.num_training_steps = 1000  # Example value, adjust as needed
        self.gradient_accumulation_steps = 1  # Example value, adjust as needed
        self.rank = current_rank()["gpus"]
        self.world_size = current_size()["gpus"]
        self._init_dist()
        super().__init__(job_config)

    def _init_dist(self):
        """Initializes torch distributed.

        torchrun would normally do this. We'll need to do something
        similar for all Torch-based actors - probably enough
        reason to introduce a `TorchActor` abstraction, or something
        similar.

        """
        env = {
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": "12345",
            "RANK": str(self.rank),
            "LOCAL_RANK": str(self.rank),
            "LOCAL_WORLD_SIZE": str(self.world_size),
            "GROUP_RANK": str(self.world_size),
            "GROUP_WORLD_SIZE": str(self.world_size),
            "ROLE_RANK": str(self.rank),
            "ROLE_WORLD_SIZE": str(self.world_size),
            "ROLE_NAME": "rank",
            "WORLD_SIZE": str(self.world_size),
            "CUDA_VISIBLE_DEVICES": str(self.rank + 4),
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
        os.environ.update(env)
        self.rlog("env: {}".format(env))
        self.clog("full env: {}".format(os.environ))

    @endpoint
    async def setup(self):
        self.train_dataloader = self.setup_data()
        # self.train_dataloader = self.setup_data(
        #     self.train_config.train_dataset_config,
        #     self.train_config.train_dataloader_config,
        #     self.train_config.packing_config,
        # )
        # self.val_dataloader = self.setup_data(
        #     self.train_config.val_dataset_config,
        #     self.train_config.val_dataloader_config,
        #     self.train_config.packing_config,
        # )

        # TODO: confirm that this is working properly
        # Should also use load, not dcp_load
        self.checkpointer.dcp_load(
            state_dict=ModelWrapper(self.model_parts).state_dict(),
            checkpoint_id=self.job_config.checkpoint.folder,
            from_hf=True,
        )
        # self.profiler = self.setup_profiler(self.train_config.profiler_config)
        # self.logger = self.setup_logger(self.train_config.logger_config)

    def setup_data(self):
        tokenizer = HuggingFaceModelTokenizer(
            tokenizer_json_path=os.path.join(
                self.job_config.model.tokenizer_path, "tokenizer.json"
            ),
            tokenizer_config_json_path=os.path.join(
                self.job_config.model.tokenizer_path, "tokenizer_config.json"
            ),
            generation_config_path=os.path.join(
                self.job_config.model.tokenizer_path, "generation_config.json"
            ),
        )

        dataset = sft_iterable_dataset(
            model_transform=tokenizer,
            message_transform=AlpacaToMessages(),
            path="yahma/alpaca-cleaned",
            split="train",
        )
        packer = TextPacker(padding_idx=0)
        dataset = PackedDataset(
            dataset=dataset,
            packer=packer,
            target_tokens_per_pack=self.job_config.training.seq_len,  # TODO: get this from model
        )
        dataloader = StatefulDataLoader(
            dataset=dataset,
            batch_size=self.job_config.training.local_batch_size,
            collate_fn=partial(
                collate_packed, mask_fn=packer.create_block_mask, device=self.device
            ),
        )

        # Ultimately we probably want something like this
        # packer = build_packing_strategy(packing_config)
        # dataset = build_dataset(dataset_config)
        # dataloader = build_dataloader(dataloader_config, dataset, packer)
        return dataloader

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
            # Pipeline Parallel forward / backward inside step() call
            with self.train_context(optional_context_parallel_ctx):
                targets, losses = (
                    (labels, []) if self.pp_has_last_stage else (None, None)
                )
                if self.pp_has_first_stage:
                    self.pp_schedule.step(
                        inputs, target=targets, losses=losses, input_batch=inputs
                    )
                else:
                    self.pp_schedule.step(
                        target=targets, losses=losses, input_batch=inputs
                    )

            # accumulate losses across pipeline microbatches
            # TODO: PP+FSDP unexpectedly puts the loss back to the CPU
            loss = (
                torch.mean(torch.stack(losses)).to(self.device)
                if self.pp_has_last_stage
                else torch.tensor([-1.0], device=self.device)
            )
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
        self.pbar.update(1)
        self.pbar.set_description(f"{self.current_step}|Loss: {loss}")

        self.optimizers.step()
        self.lr_schedulers.step()

    @endpoint
    async def train(self) -> None:
        dataloader = iter(self.train_dataloader)
        self.optimizers.zero_grad()

        self.pbar = tqdm(
            initial=0,
            total=self.num_training_steps,
            desc=f"{self.current_step}",
        )

        while self.current_step < self.num_training_steps:
            batch = next(dataloader)
            # Move tensors to the appropriate device
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to("cuda")  # TODO: hardcoded for now
            self.train_step(batch)
            # self.profiler.step()
            self.current_step += 1

            # if self.current_step % self.train_config.val_every_n_steps == 0:
            #     self.validate()
            # TODO: uncomment
            # if (
            #     self.current_step
            #     % self.train_config.checkpoint_config.save_every_n_steps
            #     == 0
            # ):
            #     self.checkpointer.save()

    @endpoint
    async def cleanup(self) -> None:
        if self.checkpointer:
            self.checkpointer.close()
        if self.metric_logger:
            self.metric_logger.close()

    def __repr__(self) -> str:
        return "Trainer"

    def rlog(self, msg: str):
        """Log for all replicas."""
        timestamp = time.strftime("%m-%d %H:%M:%S")
        print(
            "{} [{}-{}/{}] {}".format(timestamp, self, self.rank, self.world_size, msg)
        )

    def clog(self, msg: str):
        """Log only for worker 0."""
        timestamp = time.strftime("%m-%d %H:%M:%S")
        print("{} [Trainer] {}".format(timestamp, msg))


def calculate_gpu_count_from_parallelism(parallelism_cfg):
    """
    Calculate the total number of GPUs needed based on parallelism configuration.

    Args:
        parallelism_cfg: Parallelism configuration object with degrees

    Returns:
        int: Total number of GPUs required
    """
    # Extract parallelism degrees
    data_parallel_replicate = getattr(
        parallelism_cfg, "data_parallel_replicate_degree", 1
    )
    data_parallel_shard = getattr(parallelism_cfg, "data_parallel_shard_degree", 1)
    tensor_parallel = getattr(parallelism_cfg, "tensor_parallel_degree", 1)
    pipeline_parallel = getattr(parallelism_cfg, "pipeline_parallel_degree", 1)
    context_parallel = getattr(parallelism_cfg, "context_parallel_degree", 1)
    expert_parallel = getattr(parallelism_cfg, "expert_parallel_degree", 1)

    # Handle special case where -1 means use all available or auto-determine
    # For now, treat -1 as 1, but this might need adjustment based on your specific use case
    if data_parallel_shard == -1:
        data_parallel_shard = 1

    # Calculate total GPU count as product of all parallelism degrees
    total_gpus = (
        data_parallel_replicate
        * data_parallel_shard
        * tensor_parallel
        * pipeline_parallel
        * context_parallel
        * expert_parallel
    )

    return total_gpus


async def run(cfg: DictConfig) -> None:
    print("parallelism cfg: ", cfg.parallelism)

    # Parallelism to proc setup
    required_gpus = calculate_gpu_count_from_parallelism(cfg.parallelism)
    print(f"Required GPUs based on parallelism config: {required_gpus}")

    if required_gpus > 8:
        raise NotImplementedError("Only supports up to 8 GPUs (single host) for now")

    p = await proc_mesh(gpus=required_gpus, env={"CUDA_VISIBLE_DEVICES": "6,7"})
    # p = await proc_mesh(gpus=required_gpus)
    print("Created proc mesh: ", p)

    recipe = await p.spawn("sft", ForgeSFTRecipe, cfg)
    print("Created recipe: ", recipe)
    await recipe.setup.call()

    print("Recipe has been setup. Training now.")
    await recipe.train.call()
    print("Done training. Clean up")
    await recipe.cleanup.call()


@parse
def recipe_main(cfg: DictConfig) -> None:
    # TODO: this is a hack to get the defaults from ForgeJobConfig
    default_cfg = ForgeJobConfig()
    # Hack to deal with literal types from titan
    default_cfg = asdict(default_cfg)
    cfg = OmegaConf.merge(default_cfg, cfg)
    print("Cfg is: ", cfg)
    asyncio.run(run(cfg))


if __name__ == "__main__":
    sys.exit(recipe_main())
