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
from apps.sft.eval_utils import get_dp_process_group, run_evaluation
from forge.controller import ForgeActor
from forge.data.collate import collate_packed
from forge.data.datasets.packed import PackedDataset, TextPacker
from forge.data.datasets.sft_dataset import AlpacaToMessages, sft_iterable_dataset
from forge.data.tokenizer import HuggingFaceModelTokenizer
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

        # Evaluation settings - no defaults, must be explicit in config
        validation_config = job_config.get("validation")
        if validation_config is not None:
            self.validation_enabled = validation_config.get("enabled")
            self.eval_interval = validation_config.get("eval_interval")
            self.eval_steps = validation_config.get("eval_steps")
        else:
            self.validation_enabled = False
            self.eval_interval = None
            self.eval_steps = None

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
        # Always expect dataset_val.datasets configuration
        dataset_val_config = self.job_config.get("dataset_val")

        datasets = dataset_val_config["datasets"]

        # Setup all datasets
        self.val_dataloaders = {}
        self.train_dataloader = None

        for i, dataset_spec in enumerate(datasets):
            dataset_name = dataset_spec.get("name")
            dataset_path = dataset_spec.get("path")
            dataset_split = dataset_spec.get("split")

            if not dataset_name or not dataset_path or not dataset_split:
                raise ValueError(
                    f"Each dataset must have 'name', 'path', and 'split'. "
                    f"Got dataset[{i}]: {dataset_spec}"
                )

            dataloader = self.setup_data(
                dataset_path=dataset_path,
                dataset_split=dataset_split,
            )

            # First dataset with split starting with 'train' is used for training
            if i == 0 and dataset_split.startswith("train"):
                self.train_dataloader = dataloader
                logger.info(
                    f"Setup training dataset: {dataset_name} (split={dataset_split})"
                )

            # All datasets can be used for validation
            self.val_dataloaders[dataset_name] = dataloader
            logger.info(f"Setup dataset: {dataset_name} (split={dataset_split})")

        # If validation disabled, clear validation dataloaders (but keep training)
        if not self.validation_enabled:
            self.val_dataloaders = None
            logger.info("Validation disabled - only training dataloader will be used")

        # Load checkpoint if resuming
        self.checkpointer.load(step=self.current_step)

    def setup_data(
        self, dataset_path: str = "yahma/alpaca-cleaned", dataset_split: str = "train"
    ):
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
        )

        dataset = sft_iterable_dataset(
            model_transform=tokenizer,
            message_transform=AlpacaToMessages(),
            path=dataset_path,
            split=dataset_split,
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
            drop_last=True,  # Ensure consistent batch sizes across DP ranks
        )

        return dataloader

    def forward(
        self,
        input_dict: dict[str, torch.Tensor],
        labels: torch.Tensor,
        compute_gradients: bool = True,
    ) -> torch.Tensor:
        """Forward pass with optional gradient computation.

        Args:
            input_dict: Input dictionary containing tokens
            labels: Target labels
            compute_gradients: If True, compute gradients (training mode).
                             If False, skip backward pass (evaluation mode).

        Returns:
            Loss tensor
        """
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
            # Pipeline Parallel forward (with optional backward)
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
            # Non-PP forward (with optional backward)
            with self.train_context(optional_context_parallel_ctx):
                assert len(model_parts) == 1
                with self.maybe_enable_amp:
                    pred = model_parts[0](inputs)
                    loss = self.loss_fn(pred, labels)
                # need to free to before bwd to avoid peaking memory
                del pred
                if compute_gradients:
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
        loss = self.forward(batch, labels, compute_gradients=True)
        loss = loss.item()

        record_metric("ForgeSFTRecipe/train_step/loss", loss, Reduce.MEAN)
        logger.info(f"{self.current_step} / {self.num_training_steps}|Loss: {loss}")
        # self.pbar.set_description(f"{self.current_step}|Loss: {loss}")
        # self.pbar.update(1)
        self.optimizers.step()
        self.lr_schedulers.step()

    async def evaluate(self) -> dict[str, dict[str, float]]:
        """Run evaluation with async all_reduce for cross-rank epoch synchronization.

        Evaluates on all configured validation datasets and returns per-dataset metrics.

        Returns:
            Dict mapping dataset name to metrics dict, e.g.:
            {
                "val_in_domain": {"val_loss": 2.5, "val_batches": 100},
                "val_out_domain": {"val_loss": 3.1, "val_batches": 100}
            }
        """

        # Create a wrapper that calls forward with compute_gradients=False
        def forward_eval(input_dict, labels):
            return self.forward(input_dict, labels, compute_gradients=False)

        return await run_evaluation(
            val_dataloaders=self.val_dataloaders,
            model_parts=self.model_parts,
            forward_fn=forward_eval,
            device=self.device,
            eval_steps=self.eval_steps,
            dp_process_group=get_dp_process_group(self.parallel_dims),
        )

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
            if self.validation_enabled and self.current_step % self.eval_interval == 0:
                eval_metrics = await self.evaluate()
                logger.info(f"Step {self.current_step} | Eval metrics: {eval_metrics}")

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
