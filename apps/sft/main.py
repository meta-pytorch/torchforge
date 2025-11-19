# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""To run:

python -m apps.sft.main --config apps/sft/llama3_8b.yaml

"""

import asyncio
import contextlib
import logging
import math
import os
import sys
from functools import partial
from typing import Any

import torch

import torchtitan.experiments.forge.train_spec as forge_train_spec
from forge.controller import ForgeActor
from forge.data.collate import collate_packed
from forge.data.datasets.packed import PackedDataset, TextPacker
from forge.data.datasets.sft_dataset import AlpacaToMessages, sft_iterable_dataset
from forge.data.tokenizer import HuggingFaceModelTokenizer
from forge.data.utils import StopAfterOneEpoch
from forge.observability import get_or_create_metric_logger, record_metric, Reduce
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

# from tqdm import tqdm

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
        super().__init__(job_config)

    async def setup_metric_logger(self):
        """Initialization happens in the main process. Here we just retrieve it"""
        mlogger = await get_or_create_metric_logger()
        return mlogger

    def record_batch_metrics(self, data_metrics: list):
        """Since the dataloader creates new processes, we dont call `record_metric` in the dataset.
        Instead, pop the metrics from the batch and record them here."""
        for metric in data_metrics:
            record_metric(metric.key, metric.value, metric.reduction)

    @endpoint
    async def setup(self):

        # all ranks should record loss, except when PP=True. Then, only the last stage should record loss.
        self.rank_should_record_loss = True
        if hasattr(self, "pp_has_last_stage") and not self.pp_has_last_stage:
            self.rank_should_record_loss = False

        # metric logger
        self.mlogger = await self.setup_metric_logger()

        # Load training datasets
        logger.info("Setting training datasets")
        train_datasets_config = self.job_config.training.datasets
        self.train_dataloader = self.setup_data(train_datasets_config)

        # Load eval datasets
        eval_config = self.job_config["eval"]
        self.val_dataloaders = {}
        self.eval_every_n_steps = eval_config["eval_every_n_steps"]
        max_eval_steps = eval_config["max_eval_steps"]
        self.max_eval_steps = (
            max_eval_steps if max_eval_steps and max_eval_steps > 0 else None
        )
        self.validation_enabled = (
            self.eval_every_n_steps is not None and self.eval_every_n_steps > 0
        )
        if self.validation_enabled:
            logger.info("Setting eval datasets")
            self.eval_datasets_config = eval_config.datasets

            for i, dataset_config in enumerate(self.eval_datasets_config):
                ds_name = dataset_config.get("dataset_name", i)

                # TODO: Support separate eval batch size from config (eval.local_batch_size)
                dataloader = self.setup_data([dataset_config])
                self.val_dataloaders[ds_name] = dataloader

        # TODO: confirm that this is working properly
        # Should also use load, not dcp_load
        self.checkpointer.load(step=self.current_step)

        # self.profiler = self.setup_profiler(self.train_config.profiler_config)
        # self.logger = self.setup_logger(self.train_config.logger_config)

    def setup_data(self, dataset_configs: list[dict]) -> StatefulDataLoader:
        """Instantiates datasets and returns a StatefulDataLoader.

        Args:
            dataset_configs (list[dict]): List of dataset config dicts used as `sft_iterable_dataset(**dataset_configs[i])`.

        Returns:
            StatefulDataLoader

        Raises:
            ValueError: If multiple datasets provided (not yet supported)
        """
        # TODO felipemello: Currently only support single dataset
        if len(dataset_configs) > 1:
            raise ValueError(
                f"Multiple training datasets not supported yet. "
                f"Got {len(dataset_configs)} datasets. "
            )

        dataset_config = dataset_configs[0]

        # TODO: Evaluate if tokenizers should be created once and shared for every dataset
        # Load tokenizer
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

        # Get DP mesh for data sharding
        dp_mesh = None
        if self.parallel_dims is not None and self.parallel_dims.dp_enabled:
            dp_mesh = self.parallel_dims.world_mesh.get_group("dp")

        # Pass config directly to dataset constructor
        dataset = sft_iterable_dataset(
            model_transform=tokenizer,
            message_transform=AlpacaToMessages(),
            dp_mesh=dp_mesh,
            **dataset_config,
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
        self,
        input_dict: dict[str, torch.Tensor],
        labels: torch.Tensor,
        skip_backward: bool = False,
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
                    self.pp_schedule.step(inputs, target=targets, losses=losses)
                else:
                    self.pp_schedule.step(target=targets, losses=losses)

            # accumulate losses across pipeline microbatches
            # TODO: PP+FSDP unexpectedly puts the loss back to the CPU
            loss = (
                torch.sum(torch.stack(losses)).to(self.device)
                if self.pp_has_last_stage
                else torch.tensor(-1.0, device=self.device)
            )

            # TODO: PP requires gradients enabled and cant deactive with no_grad
            if skip_backward:
                loss = loss.detach()

        else:
            # Non-PP forward / backward
            with self.train_context(optional_context_parallel_ctx):
                assert len(model_parts) == 1
                with self.maybe_enable_amp:
                    pred = model_parts[0](inputs)
                    loss = self.loss_fn(pred, labels)
                # need to free to before bwd to avoid peaking memory
                del pred

                # Only run backward if requested. Useful for eval.
                if not skip_backward:
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

        if self.rank_should_record_loss:
            loss_val = loss.item()
            record_metric("ForgeSFTRecipe/train_step/loss", loss_val, Reduce.MEAN)
            logger.info(
                f"step {self.current_step} / {self.num_training_steps} | Loss: {loss_val}"
            )

        # self.pbar.set_description(f"{self.current_step}|Loss: {loss}")
        # self.pbar.update(1)
        self.optimizers.step()
        self.lr_schedulers.step()

    async def evaluate(self) -> None:
        """Run evaluation on multiple datasets, one at a time.

        1. Set models to eval mode
        2. For each eval dataset:
            - Create fresh iterator (starts from epoch 0)
            - Use StopAfterOneEpoch to iterate until epoch boundary. This utility
                is necessary for infinite iterable dataset, since epoch boundaries are not known.
            - Respect max_eval_steps cap if configured
            - Record loss and step metrics (on dp rank only)
        3. Restore models to train mode
        """

        # Set models to eval mode
        for model_part in self.model_parts:
            model_part.eval()

        # Get DP process group for epoch synchronization
        dp_mesh = None
        if self.parallel_dims is not None and self.parallel_dims.dp_enabled:
            dp_mesh = self.parallel_dims.world_mesh.get_group("dp")

        # For non-PP: disable gradients to save memory
        # TODO: For PP, if disabling gradients, throws error
        maybe_no_grad = (
            contextlib.nullcontext()
            if self.parallel_dims.pp_enabled
            else torch.no_grad()
        )

        # Evaluate each dataset sequentially
        all_dataset_losses = []
        all_dataset_steps = []
        for dataset_name, val_dataloader in self.val_dataloaders.items():
            logger.info(f"=====Evaluating dataset: {dataset_name}=====")

            # Evaluation loop for this dataset
            total_loss = torch.tensor(0.0, device=self.device)
            num_steps = 0

            # NOTE: Assumes batch contains field "metrics" containing "num_epochs"
            batch_iter = StopAfterOneEpoch(
                iter=iter(val_dataloader),  # Fresh iterator from epoch 0,
                device=self.device,
                dp_mesh=dp_mesh,
            )

            with maybe_no_grad:
                for batch in batch_iter:
                    # if max_eval_steps>len(dataset), it will be stopped earlier by StopAfterOneEpoch.
                    if (
                        self.max_eval_steps is not None
                        and num_steps >= self.max_eval_steps
                    ):
                        logger.info(
                            f"[{dataset_name}] Reached max_eval_steps cap of {self.max_eval_steps}"
                        )
                        break

                    # Move tensors to device
                    for key, value in batch.items():
                        if isinstance(value, torch.Tensor):
                            batch[key] = value.to(self.device)

                    # Process batch
                    labels = batch.pop("labels")
                    loss = self.forward_backward(batch, labels, skip_backward=True)
                    total_loss += loss
                    num_steps += 1

                    # Log progress
                    if self.rank_should_record_loss:
                        loss_val = loss.item()
                        logger.info(
                            f"[dataset {dataset_name}] Step {num_steps} | Loss: {loss_val:.4f}"
                        )

            # log loss
            avg_loss = (total_loss / max(num_steps, 1)).item()
            all_dataset_losses.append(avg_loss)
            all_dataset_steps.append(num_steps)
            logger.info(
                f"[dataset {dataset_name}] Final Step {num_steps} | Avg Loss: {avg_loss:.4f}"
            )
            if self.rank_should_record_loss:
                record_metric(
                    f"evaluate/dataset_{dataset_name}_avg_loss",
                    avg_loss,
                    Reduce.MEAN,
                )

        # Record macro and micro average losses across datasets (only if multiple datasets)
        if self.rank_should_record_loss and len(all_dataset_losses) > 1:
            # Macro: same weight for all datasets
            macro_avg_loss = sum(all_dataset_losses) / len(all_dataset_losses)
            record_metric("evaluate/macro_avg_loss", macro_avg_loss, Reduce.MEAN)

            # Micro: weighted mean by dataset size
            total_steps = sum(all_dataset_steps)
            micro_avg_loss = (
                sum(
                    loss * steps
                    for loss, steps in zip(all_dataset_losses, all_dataset_steps)
                )
                / total_steps
            )
            record_metric("evaluate/micro_avg_loss", micro_avg_loss, Reduce.MEAN)

            logger.info(
                f"Macro avg loss (unweighted): {macro_avg_loss:.4f}, "
                f"Micro avg loss (weighted): {micro_avg_loss:.4f}"
            )

        # Restore train mode
        for model_part in self.model_parts:
            model_part.train()

        logger.info("==Evaluation complete==")

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
                    batch[k] = v.to("cuda")  # TODO: hardcoded for now

            self.train_step(batch)
            # self.profiler.step()
            self.current_step += 1

            # Run evaluation periodically if enabled
            if (
                self.validation_enabled
                and self.current_step % self.eval_every_n_steps == 0
            ):
                await self.evaluate()

            self.checkpointer.save(
                curr_step=self.current_step,
                last_step=self.current_step == self.num_training_steps,
            )

            # Flush metrics
            if self._rank == 0:
                await self.mlogger.flush.call_one(global_step=self.current_step)

        # self.pbar.close()

        if self.validation_enabled:
            logger.info("Running final evaluation at end of training...")
            await self.evaluate()

    @endpoint
    async def cleanup(self) -> None:
        if self.checkpointer:
            self.checkpointer.close()
        if getattr(self, "mlogger", None):
            await self.mlogger.shutdown.call_one()

    def __repr__(self) -> str:
        return "Trainer"


async def run(cfg: DictConfig) -> None:
    logging.info("Spawning recipe...")
    process_cfg = cfg.pop("processes")

    # Initialize metric logger in main process
    metric_logging_cfg = cfg.get("metric_logging", {})
    mlogger = await get_or_create_metric_logger(process_name="Controller")
    await mlogger.init_backends.call_one(metric_logging_cfg)

    recipe = await ForgeSFTRecipe.options(**process_cfg).as_actor(cfg)

    logging.info("Created recipe, running setup.")
    await recipe.setup.call()

    logging.info("Recipe has been setup. Training now.")
    await recipe.train.call()

    logging.info("Done training. Clean up")
    await recipe.cleanup.call()

    await recipe.mesh.stop()
    logging.info("All done!")


@parse
def recipe_main(cfg: DictConfig) -> None:
    asyncio.run(run(cfg))


if __name__ == "__main__":
    sys.exit(recipe_main())
