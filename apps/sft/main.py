# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""To run:

python -m apps.sft.main --config apps/sft/llama3_8b.yaml
"""

import asyncio

import logging
import math
import os
import sys
from functools import partial
from typing import Any

import torch

import torchtitan.experiments.forge.train_spec as forge_train_spec
from forge.controller import ForgeActor
from forge.controller.provisioner import init_provisioner, shutdown
from forge.data.collate import collate_packed
from forge.data.datasets.packed import PackedDataset, TextPacker
from forge.data.datasets.sft_dataset import AlpacaToMessages, sft_iterable_dataset
from forge.data.tokenizer import HuggingFaceModelTokenizer
from forge.observability import get_or_create_metric_logger, record_metric, Reduce
from forge.types import LauncherConfig, ProvisionerConfig
from forge.util.config import parse

from monarch.actor import current_rank, current_size, endpoint
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torchdata.stateful_dataloader import StatefulDataLoader
from torchtitan.components.loss import LossFunction
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.distributed import ParallelDims, utils as dist_utils
from torchtitan.experiments.forge.engine import ForgeEngine
from torchtitan.experiments.forge.job_config import ForgeJobConfig

# stubs for now
Checkpointer = Any
Dataloader = Any
MetricLogger = Any
Profiler = Any
Tokenizer = Any

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ForgeSFTRecipe(ForgeActor, ForgeEngine):
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

    def __init__(self, config: DictConfig):
        job_config = ForgeJobConfig().to_dict()
        # Hack to deal with literal types from titan
        job_config = OmegaConf.merge(job_config, config)

        self.current_step = 0
        self.num_training_steps = job_config.training.steps
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
        # Calculate local rank - rank within the node
        # For multi-node setups, LOCAL_RANK should be rank % gpus_per_node
        size_info = current_size()

        # size_info = {'hosts': 8, 'procs': 4} for 8 nodes with 4 GPUs each
        local_world_size = (
            size_info.get("procs", self._size) if size_info else self._size
        )
        local_rank = self._rank % local_world_size

        env = {
            "RANK": str(self._rank),
            "LOCAL_RANK": str(local_rank),
            "LOCAL_WORLD_SIZE": str(self._size),
            "GROUP_RANK": str(self._size),
            "GROUP_WORLD_SIZE": str(self._size),
            "ROLE_RANK": str(self._rank),
            "ROLE_WORLD_SIZE": str(self._size),
            "ROLE_NAME": "rank",
            "WORLD_SIZE": str(self._size),
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            # Add other environment variables as needed - NCCL related variables, etc
        }
        os.environ.update(env)
        logger.info("env: {}".format(env))

    async def setup_metric_logger(self):
        """Retrieve the already-initialized metric logger from main process"""
        # Don't create new logger - it was already initialized in main process
        # Just retrieve the existing one
        mlogger = await get_or_create_metric_logger()
        return mlogger

    def record_batch_metrics(self, data_metrics: list):
        """Since the dataloader creates new processes, we dont call `record_metric` in the dataset.
        Instead, pop the metrics from the batch and record them here."""
        for metric in data_metrics:
            record_metric(metric.key, metric.value, metric.reduction)

    @endpoint
    async def setup(self):
        self.train_dataloader = self.setup_data()

        self.mlogger = await self.setup_metric_logger()
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

        # Setup training data (first 90% of train split)

        # Load checkpoint if resuming
        self.checkpointer.load(step=self.current_step)
        # self.profiler = self.setup_profiler(self.train_config.profiler_config)
        # self.logger = self.setup_logger(self.train_config.logger_config)

    def setup_data(self):
        """Setup data with configurable dataset path and split."""
        print(os.path.join(self.job_config.model.hf_assets_path, "tokenizer.json"))
        tokenizer = HuggingFaceModelTokenizer(
            tokenizer_json_path=os.path.join(
                self.job_config.model.hf_assets_path, "tokenizer.json"
            ),
            tokenizer_config_json_path=os.path.join(
                self.job_config.model.hf_assets_path, "tokenizer_config.json"
            ),
            generation_config_path=os.path.join(
                self.job_config.model.hf_assets_path, "generation_config.json"
            ),
            chat_template_path=(
                path
                if os.path.exists(
                    path := os.path.join(
                        self.job_config.model.hf_assets_path, "chat_template.jinja"
                    )
                )
                else None
            ),
        )

        # Ultimately we probably want something like this
        # packer = build_packing_strategy(packing_config)
        # dataset = build_dataset(dataset_config)
        # dataloader = build_dataloader(dataloader_config, dataset, packer)

        # Get data config from YAML (num_shards_per_rank, num_dataloader_workers)
        data_config = getattr(self.job_config, "data", None)
        num_shards_per_rank = (
            getattr(data_config, "num_shards_per_rank", 64) if data_config else 64
        )
        num_dataloader_workers = (
            getattr(data_config, "num_dataloader_workers", 0) if data_config else 0
        )

        dataset = sft_iterable_dataset(
            model_transform=tokenizer,
            message_transform=AlpacaToMessages(),
            path="yahma/alpaca-cleaned",
            split="train",
            num_shards_per_rank=num_shards_per_rank,
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
            num_workers=num_dataloader_workers,
            collate_fn=partial(
                collate_packed, mask_fn=packer.create_block_mask, device=self.device
            ),
        )

        # Ultimately we probably want something like this
        # packer = build_packing_strategy(packing_config)

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
        loss = loss.item()

        # Record loss metric
        record_metric("ForgeSFTRecipe/train_step/loss", loss, Reduce.MEAN)
        logger.info(f"{self.current_step} / {self.num_training_steps}|Loss: {loss}")
        # self.pbar.set_description(f"{self.current_step}|Loss: {loss}")
        # self.pbar.update(1)

        self.optimizers.step()
        self.lr_schedulers.step()

    @endpoint
    async def train(self) -> None:
        dataloader = iter(self.train_dataloader)
        self.optimizers.zero_grad()

        # TODO: tqdm is broken in Monarch actors
        # self.pbar = tqdm(initial=self.current_step, total=self.num_training_steps)

        while self.current_step < self.num_training_steps:
            batch = next(dataloader)

            # Pop and record metrics from batch before moving to device
            self.record_batch_metrics(batch.pop("metrics", []))
            record_metric("ForgeSFTRecipe/train/step", self.current_step, Reduce.MEAN)

            # Move tensors to the appropriate device
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(self.device)  # TODO: hardcoded for now

            self.train_step(batch)
            # self.profiler.step()
            self.current_step += 1

            # Flush metrics
            if self._rank == 0 and self.mlogger is not None:
                logger.debug(f"Flushing metrics at step {self.current_step}")
                await self.mlogger.flush.call_one(global_step=self.current_step)

            # Save checkpoints
            self.checkpointer.save(
                curr_step=self.current_step,
                last_step=self.current_step == self.num_training_steps,
            )
        # self.pbar.close()

    @endpoint
    async def cleanup(self) -> None:
        if self.checkpointer:
            self.checkpointer.close()
        if getattr(self, "mlogger", None):
            await self.mlogger.shutdown.call_one()

    def __repr__(self) -> str:
        return "Trainer"


async def run(cfg: DictConfig) -> None:
    """Main SFT training loop with provisioner support for multi-node training."""
    # ---- Global setups ---- #
    provisioner = None
    if cfg.get("provisioner", None) is not None:
        logging.info("Initializing provisioner with launcher configuration...")
        provisioner = await init_provisioner(
            ProvisionerConfig(launcher_config=LauncherConfig(**cfg.provisioner))
        )
    else:
        logging.info("Initializing default provisioner...")
        provisioner = await init_provisioner()

    # ---- Initialize metric logger in main process ---- #
    metric_logging_cfg = cfg.get("metric_logging", {})
    mlogger = await get_or_create_metric_logger(process_name="Controller")
    await mlogger.init_backends.call_one(metric_logging_cfg)

    # ---- Setup SFT Recipe Actor ---- #
    logging.info("Spawning recipe...")
    actor_cfg = cfg.pop("actors", None)

    if actor_cfg is None:
        # Fallback to old "processes" config for backward compatibility
        actor_cfg = cfg.pop("processes", {"procs": 8, "with_gpus": True})
        logging.warning(
            "Using legacy 'processes' config. Consider migrating to 'actors' config."
        )

    recipe_options = actor_cfg.get("trainer", actor_cfg)
    recipe = await ForgeSFTRecipe.options(**recipe_options).as_actor(cfg)

    logging.info("Created recipe, running setup.")
    await recipe.setup.call()

    logging.info("Recipe has been setup. Training now.")

    try:
        await recipe.train.call()
    except KeyboardInterrupt:
        logging.info("Training interrupted by user")
    finally:
        logging.info("Done training. Clean up")
        await recipe.cleanup.call()
        await ForgeSFTRecipe.shutdown(recipe)

        # Shutdown provisioner
        await shutdown()
        logging.info("All done!")


@parse
def recipe_main(cfg: DictConfig) -> None:
    asyncio.run(run(cfg))


if __name__ == "__main__":
    sys.exit(recipe_main())
