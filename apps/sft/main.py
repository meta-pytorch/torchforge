# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
from dataclasses import asdict
from functools import partial
from typing import Any

import torch

import torchtitan.experiments.forge.train_spec as forge_train_spec
from forge.config.parse import parse
from forge.data.collate import collate_packed
from forge.data.transforms.tokenizers import HuggingFaceModelTokenizer
from forge.data.transforms.transforms import AlpacaToMessages
from forge.datasets.packed import PackedDataset, TextPacker
from forge.datasets.sft import sft_iterable_dataset

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


class ForgeSFTRecipe(ForgeEngine):
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
        super().__init__(job_config)

    def setup(self):
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
            target_tokens_per_pack=1024,  # TODO: get this from model
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

    def train(self) -> None:
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

    def cleanup(self) -> None:
        if self.checkpointer:
            self.checkpointer.close()
        if self.metric_logger:
            self.metric_logger.close()


@parse
def recipe_main(cfg: DictConfig) -> None:
    # TODO: this is a hack to get the defaults from ForgeJobConfig
    default_cfg = ForgeJobConfig()
    # Hack to deal with literal types from titan
    default_cfg = asdict(default_cfg)
    cfg = OmegaConf.merge(default_cfg, cfg)
    recipe = ForgeSFTRecipe(cfg)
    recipe.setup()
    recipe.train()
    recipe.cleanup()


if __name__ == "__main__":
    sys.exit(recipe_main())
