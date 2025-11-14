# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Trainer actor implementation for SFT training.

This is a concrete implementation of BaseForgeActor for supervised fine-tuning.
"""

import logging

import torch
import torchtitan.experiments.forge.train_spec as forge_train_spec
from apps.sft.actor import BaseForgeActor
from apps.sft.utils import (
    create_context_parallel_context,
    log_training_step,
    move_batch_to_device,
    setup_sft_dataloader,
    setup_tokenizer,
)
from monarch.actor import endpoint
from omegaconf import DictConfig

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class TrainerActor(BaseForgeActor):
    """
    Concrete trainer actor for supervised fine-tuning.

    Handles training loop, forward/backward passes, and checkpoint management.
    """

    train_spec: forge_train_spec.ForgeTrainSpec
    train_dataloader: any
    num_training_steps: int

    def __init__(self, config: DictConfig):
        """
        Initialize the trainer actor.

        Args:
            config: Configuration dictionary containing training settings
        """
        super().__init__(config)
        self.num_training_steps = self.job_config.training.steps

    @endpoint
    async def setup(self):
        """
        Setup the trainer (load data, checkpoint, etc.).
        """
        logger.info("Setting up trainer actor...")

        self.tokenizer = setup_tokenizer(
            hf_assets_path=self.job_config.model.hf_assets_path
        )

        self.train_dataloader = setup_sft_dataloader(
            tokenizer=self.tokenizer,
            dataset_path="yahma/alpaca-cleaned",
            dataset_split="train",
            target_tokens_per_pack=self.job_config.training.seq_len,
            batch_size=self.job_config.training.local_batch_size,
            device=self.device,
        )

        if self.checkpointer:
            logger.info("Loading checkpoint...")
            self.checkpointer.load(step=self.current_step)

        logger.info("Trainer setup complete.")

    def forward_backward(
        self, input_dict: dict[str, torch.Tensor], labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Perform forward and backward pass.

        Args:
            input_dict: Dictionary containing input tokens
            labels: Ground truth labels

        Returns:
            Computed loss value
        """
        model_parts = self.model_parts
        parallel_dims = self.parallel_dims
        inputs = input_dict["tokens"]

        optional_context_parallel_ctx = create_context_parallel_context(
            parallel_dims=parallel_dims,
            inputs=inputs,
            labels=labels,
            model_parts=model_parts,
            rotate_method=self.job_config.parallelism.context_parallel_rotate_method,
        )

        if parallel_dims.pp_enabled:
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

            loss = (
                torch.mean(torch.stack(losses)).to(self.device)
                if self.pp_has_last_stage
                else torch.tensor([-1.0], device=self.device)
            )
        else:
            with self.train_context(optional_context_parallel_ctx):
                assert len(model_parts) == 1
                with self.maybe_enable_amp:
                    pred = model_parts[0](inputs)
                    loss = self.loss_fn(pred, labels)
                del pred
                loss.backward()

        return loss

    def train_step(self, batch: dict[str, torch.Tensor]) -> None:
        """
        Execute a single training step.

        Args:
            batch: Dictionary containing batch data (tokens, labels, etc.)
        """
        labels = batch.pop("labels")
        loss = self.forward_backward(batch, labels)

        log_training_step(self.current_step, self.num_training_steps, loss, logger)

        self.optimizers.step()
        self.lr_schedulers.step()

    @endpoint
    async def run(self) -> None:
        """
        Main training loop.
        """
        logger.info("Starting training loop...")

        dataloader = iter(self.train_dataloader)
        self.optimizers.zero_grad()

        while self.current_step < self.num_training_steps:
            batch = next(dataloader)
            batch = move_batch_to_device(batch, self.device)

            self.train_step(batch)
            self.current_step += 1

            if self.checkpointer:
                self.checkpointer.save(
                    curr_step=self.current_step,
                    last_step=self.current_step == self.num_training_steps,
                )

        logger.info("Training complete!")

    @endpoint
    async def cleanup(self) -> None:
        """
        Cleanup resources (close checkpointer, logger, etc.).
        """
        logger.info("Cleaning up trainer actor...")

        if self.checkpointer:
            self.checkpointer.close()
        if self.metric_logger:
            self.metric_logger.close()

        logger.info("Cleanup complete.")

    def __repr__(self) -> str:
        return "TrainerActor"
