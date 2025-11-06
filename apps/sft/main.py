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
from forge.data.collate import collate_packed
from forge.data.datasets.packed import PackedDataset, TextPacker
from forge.data.datasets.sft_dataset import AlpacaToMessages, sft_iterable_dataset
from forge.data.tokenizer import HuggingFaceModelTokenizer
from forge.data.utils import StopAfterOneEpoch
from forge.observability import get_or_create_metric_logger, record_metric, Reduce
from forge.util.config import parse
from forge.util.logging import log_rank_zero

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
        # Load training datasets
        logger.info("Setting training datasets")
        train_datasets_config = self.job_config.training.datasets
        self.train_dataloader = self.setup_data(train_datasets_config)

        # Load eval datasets
        eval_config = self.job_config.get("eval", {})
        self.val_dataloaders = {}
        self.eval_every_n_steps = eval_config.get("eval_every_n_steps", None)
        max_eval_steps = eval_config.get("max_eval_steps", None)
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

                dataloader = self.setup_data([dataset_config])
                self.val_dataloaders[ds_name] = dataloader

        # Load checkpoint if resuming
        self.checkpointer.load(step=self.current_step)

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

        # Store tokenizer for later use (e.g., decoding in debug logs)
        self.tokenizer = tokenizer

        # Get DP mesh for data sharding
        dp_mesh = None
        if self.parallel_dims is not None and self.parallel_dims.dp_enabled:
            dp_mesh = self.parallel_dims.world_mesh.get_group("dp")

        # Pass config directly to dataset constructor
        dataset = sft_iterable_dataset(
            model_transform=tokenizer,
            message_transform=AlpacaToMessages(),
            dp_mesh=dp_mesh,
            **dataset_config,  # Unpack config (path, split, etc.)
        )

        packer = TextPacker(padding_idx=0)
        dataset = PackedDataset(
            dataset=dataset,
            packer=packer,
            target_tokens_per_pack=self.job_config.training.seq_len,
        )

        return StatefulDataLoader(
            dataset=dataset,
            batch_size=self.job_config.training.local_batch_size,
            collate_fn=partial(
                collate_packed, mask_fn=packer.create_block_mask, device=self.device
            ),
        )

    def forward(
        self,
        input_dict: dict[str, torch.Tensor],
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass only (no backward)"""
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
            # Note: backward only happens if not in torch.no_grad() context
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
            # Non-PP forward
            with self.train_context(optional_context_parallel_ctx):
                assert len(model_parts) == 1
                with self.maybe_enable_amp:
                    pred = model_parts[0](inputs)
                    loss = self.loss_fn(pred, labels)
                # need to free to before bwd to avoid peaking memory
                del pred

        return loss

    def forward_backward(
        self,
        input_dict: dict[str, torch.Tensor],
        labels: torch.Tensor,
    ) -> torch.Tensor:
        loss = self.forward(input_dict, labels)

        # For non-PP, explicitly call backward (PP does it inside step())
        if not self.parallel_dims.pp_enabled:
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

        record_metric("ForgeSFTRecipe/train_step/loss", loss, Reduce.MEAN)
        logger.info(f"{self.current_step} / {self.num_training_steps}|Loss: {loss}")
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
        logger.debug("==STARTING EVALUATION==")

        # Set models to eval mode
        for model_part in self.model_parts:
            model_part.eval()

        # Get DP process group for epoch synchronization
        dp_mesh = None
        if self.parallel_dims is not None and self.parallel_dims.dp_enabled:
            dp_mesh = self.parallel_dims.world_mesh.get_group("dp")

        # Evaluate each dataset sequentially
        for dataset_name, val_dataloader in self.val_dataloaders.items():
            logger.debug(f"=====Evaluating dataset: {dataset_name}=====")

            # Evaluation loop for this dataset
            total_loss = torch.tensor(0.0, device=self.device)
            num_steps = 0

            # NOTE: Assumes batch contains samples with Metric("num_epochs", ...) field
            batch_iter = StopAfterOneEpoch(
                dataloader_iter=iter(val_dataloader),  # Fresh iterator from epoch 0,
                dp_mesh=dp_mesh,
            )

            with torch.no_grad():
                for batch in batch_iter:
                    # Check max_eval_steps limit
                    if (
                        self.max_eval_steps is not None
                        and num_steps >= self.max_eval_steps
                    ):
                        log_rank_zero(
                            logger,
                            f"[{dataset_name}] Reached max_eval_steps cap of {self.max_eval_steps}",
                        )
                        break

                    # Move tensors to device
                    for key, value in batch.items():
                        if isinstance(value, torch.Tensor):
                            batch[key] = value.to(self.device)

                    # Process batch
                    labels = batch.pop("labels")
                    loss = self.forward(batch, labels)
                    total_loss += loss
                    num_steps += 1

                    # Log progress (rank 0 only)
                    if num_steps % 100 == 0:
                        loss_val = loss.item()
                        log_rank_zero(
                            logger,
                            f"  [{dataset_name}] Step {num_steps} | Loss: {loss_val:.4f}",
                        )

            # Compute average loss
            avg_loss = (total_loss / max(num_steps, 1)).item()
            log_rank_zero(logger, f"  [{dataset_name}] avg_loss: {avg_loss:.4f}")

            # Record metrics only on DP rank 0 to avoid double counting
            # record_metric aggregates across all processes via monarch
            should_record = True
            if dp_mesh is not None:
                dp_rank = torch.distributed.get_rank(group=dp_mesh)
                should_record = dp_rank == 0

            if should_record:
                record_metric(
                    f"ForgeSFTRecipe/evaluate/{dataset_name}_loss",
                    avg_loss,
                    Reduce.MEAN,
                )
                record_metric(
                    f"ForgeSFTRecipe/evaluate/{dataset_name}_steps",
                    num_steps,
                    Reduce.MEAN,
                )

        # Restore train mode
        for model_part in self.model_parts:
            model_part.train()

        # Summary
        logger.debug("==EVALUATION COMPLETE==")

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

        # self.pbar.close()

        # Run final evaluation at end of training
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
