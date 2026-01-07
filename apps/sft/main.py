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
from typing import Any

import torch

from forge.actors.trainer import TitanTrainer
from forge.data.collate import collate_padded
from forge.data.datasets.sft_dataset import AlpacaToMessages, sft_iterable_dataset
from forge.data.tokenizer import HuggingFaceModelTokenizer
from forge.data.utils import StopAfterOneEpoch
from forge.observability import get_or_create_metric_logger, record_metric, Reduce
from forge.util.config import parse

from monarch.actor import current_rank, current_size, endpoint
from omegaconf import DictConfig, OmegaConf
from torchdata.stateful_dataloader import StatefulDataLoader
from torchtitan.experiments.forge.job_config import ForgeJobConfig

# stubs for now
Checkpointer = Any
Dataloader = Any
MetricLogger = Any
Profiler = Any
Tokenizer = Any

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ForgeSFTRecipe(TitanTrainer):
    """SFT Recipe built on TitanTrainer.

    Inherits from TitanTrainer which provides:
    - rank_should_record_loss: Only last PP stage records loss
    - record_batch_metrics(): Generic batch metric recording
    - setup_metric_logger(): Metric logger initialization
    - forward_backward(): Forward/backward with PP+CP support
    - train_step_sft(): SFT-style training step

    Uses composition pattern: access engine via self.engine.X

    This class adds SFT-specific functionality:
    - Data loading (tokenizer, datasets)
    - Evaluation loop
    - Training loop with eval triggers
    """

    def __init__(self, config: DictConfig):
        # Get valid ForgeJobConfig fields
        forge_job_config_defaults = ForgeJobConfig().to_dict()

        # Store the full config for SFT-specific fields
        full_config = OmegaConf.to_container(config, resolve=True)

        # Only merge config keys that are valid ForgeJobConfig fields
        # Filter out non-ForgeJobConfig fields like model_name, processes, metric_logging, eval, etc.
        filtered_config = {
            k: v for k, v in full_config.items() if k in forge_job_config_defaults
        }

        # Store datasets separately before filtering them out (they're SFT-specific, not in Training dataclass)
        training_datasets = None
        if "training" in filtered_config and "datasets" in filtered_config["training"]:
            training_datasets = filtered_config["training"].pop("datasets")

        # Store eval config separately (SFT-specific, not in ForgeJobConfig)
        self.eval_config = full_config.get("eval", {})

        job_config = OmegaConf.merge(forge_job_config_defaults, filtered_config)

        self.job_config = ForgeJobConfig(**job_config)
        # Restore datasets to job_config for SFT use
        if training_datasets is not None:
            self.job_config.training.datasets = training_datasets

        self.current_step = 0
        self.num_training_steps = self.job_config.training.steps
        self.gradient_accumulation_steps = 1
        self._rank = current_rank().rank
        self._size = math.prod(current_size().values())

        # Convert job_config to dict for TitanTrainer, ensuring datasets is removed
        titan_config = OmegaConf.to_container(job_config, resolve=True)
        if "training" in titan_config and "datasets" in titan_config["training"]:
            del titan_config["training"]["datasets"]

        # Initialize TitanTrainer with job config fields (without datasets)
        super().__init__(**titan_config)

    # setup_metric_logger() inherited from TitanTrainer
    # record_batch_metrics() inherited from TitanTrainer

    @endpoint
    async def setup(self):
        # Call parent's setup helper to initialize engine and rank_should_record_loss
        # self._setup_engine()

        # Validate that compile is only used with flex attention
        if self.job_config.compile.enable:
            raise ValueError(
                "compile.enable=True is not currently supported. "
                "Compile is only supported with flex attention enabled, which requires PyTorch nightly. "
                "Please set compile.enable=false in your config."
            )

        # metric logger (inherited from TitanTrainer)
        self.mlogger = await self.setup_metric_logger()

        # Load training datasets
        logger.info("Setting training datasets")
        train_datasets_config = self.job_config.training.datasets
        self.train_dataloader = self.setup_data(train_datasets_config)

        # Load eval datasets (using self.eval_config stored in __init__)
        self.val_dataloaders = {}
        self.eval_every_n_steps = self.eval_config.get("eval_every_n_steps")
        max_eval_steps = self.eval_config.get("max_eval_steps")
        self.max_eval_steps = (
            max_eval_steps if max_eval_steps and max_eval_steps > 0 else None
        )
        self.validation_enabled = (
            self.eval_every_n_steps is not None and self.eval_every_n_steps > 0
        )
        if self.validation_enabled:
            logger.info("Setting eval datasets")
            self.eval_datasets_config = self.eval_config.get("datasets", [])

            for i, dataset_config in enumerate(self.eval_datasets_config):
                ds_name = dataset_config.get("dataset_name", i)

                # TODO: Support separate eval batch size from config (eval.local_batch_size)
                dataloader = self.setup_data([dataset_config])
                self.val_dataloaders[ds_name] = dataloader

        # TODO: confirm that this is working properly
        # Should also use load, not dcp_load
        self.engine.checkpointer.load(step=self.current_step)

    def setup_data(self, dataset_configs: list[dict]) -> StatefulDataLoader:
        """Instantiates datasets and returns a StatefulDataLoader.

        Args:
            dataset_configs (list[dict]): List of dataset config dicts.

        Returns:
            StatefulDataLoader
        """

        # TODO felipemello: Currently only support single dataset
        if len(dataset_configs) > 1:
            raise ValueError(
                f"Multiple training datasets not supported yet. "
                f"Got {len(dataset_configs)} datasets. "
            )

        dataset_config = dataset_configs[0]

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

        # Get DP mesh for data sharding (use self.engine for composition)
        dp_mesh = None
        if (
            self.engine.parallel_dims is not None
            and self.engine.parallel_dims.dp_enabled
        ):
            dp_mesh = self.engine.parallel_dims.world_mesh.get_group("dp")

        # Pass config directly to dataset constructor
        dataset = sft_iterable_dataset(
            model_transform=tokenizer,
            message_transform=AlpacaToMessages(),
            dp_mesh=dp_mesh,
            **dataset_config,
        )

        dataloader = StatefulDataLoader(
            dataset=dataset,
            batch_size=self.job_config.training.local_batch_size,
            collate_fn=collate_padded,
        )

        return dataloader

    # forward_backward() inherited from TitanTrainer
    # train_step_sft() inherited from TitanTrainer

    async def evaluate(self) -> None:
        """Run evaluation on multiple datasets, one at a time."""

        # Set models to eval mode (use self.engine for composition)
        for model_part in self.engine.model_parts:
            model_part.eval()

        # Get DP process group for epoch synchronization
        dp_mesh = None
        if (
            self.engine.parallel_dims is not None
            and self.engine.parallel_dims.dp_enabled
        ):
            dp_mesh = self.engine.parallel_dims.world_mesh.get_group("dp")

        # For non-PP: disable gradients to save memory
        maybe_no_grad = (
            contextlib.nullcontext()
            if self.engine.parallel_dims.pp_enabled
            else torch.no_grad()
        )

        # Evaluate each dataset sequentially
        all_dataset_losses = []
        all_dataset_steps = []
        for dataset_name, val_dataloader in self.val_dataloaders.items():
            logger.info(f"=====Evaluating dataset: {dataset_name}=====")

            # Evaluation loop for this dataset
            total_loss = torch.tensor(0.0, device=self.engine.device)
            num_steps = 0

            batch_iter = StopAfterOneEpoch(
                iter=iter(val_dataloader),
                device=self.engine.device,
                dp_mesh=dp_mesh,
            )

            with maybe_no_grad:
                for batch in batch_iter:
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
                            batch[key] = value.to(self.engine.device)

                    # Process batch - use inherited forward_backward
                    labels = batch.pop("labels")
                    loss = self.forward_backward(batch, labels, skip_backward=True)
                    total_loss += loss
                    num_steps += 1

                    if self.rank_should_record_loss:
                        loss_val = loss.item()
                        logger.info(
                            f"[dataset {dataset_name}] Step {num_steps} | Loss: {loss_val:.4f}"
                        )

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

        # Record macro and micro average losses across datasets
        if self.rank_should_record_loss and len(all_dataset_losses) > 1:
            macro_avg_loss = sum(all_dataset_losses) / len(all_dataset_losses)
            record_metric("evaluate/macro_avg_loss", macro_avg_loss, Reduce.MEAN)

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
        for model_part in self.engine.model_parts:
            model_part.train()

        logger.info("==Evaluation complete==")

    @endpoint
    async def train(self) -> None:
        dataloader = iter(self.train_dataloader)
        self.engine.optimizers.zero_grad()

        while self.current_step < self.num_training_steps:
            batch = next(dataloader)

            # Pop and record metrics from batch (inherited from TitanTrainer)
            self.record_batch_metrics(batch.pop("metrics", []))
            record_metric("ForgeSFTRecipe/train/step", self.current_step, Reduce.MEAN)

            # Move tensors to the appropriate device
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(self.engine.device)

            # Use inherited train_step_sft
            self.train_step_sft(batch)
            self.current_step += 1

            # Run evaluation periodically if enabled
            if (
                self.validation_enabled
                and self.current_step % self.eval_every_n_steps == 0
            ):
                await self.evaluate()

            self.engine.checkpointer.save(
                curr_step=self.current_step,
                last_step=self.current_step == self.num_training_steps,
            )

            # Flush metrics
            if self._rank == 0:
                await self.mlogger.flush.call_one(global_step=self.current_step)

        if self.validation_enabled:
            logger.info("Running final evaluation at end of training...")
            await self.evaluate()

    @endpoint
    async def cleanup(self) -> None:
        if self.engine.checkpointer:
            self.engine.checkpointer.close()
        if getattr(self, "mlogger", None):
            await self.mlogger.shutdown.call_one()

    def __repr__(self) -> str:
        return "ForgeSFTRecipe"


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
