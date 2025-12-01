# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Trainer actor implementation for SFT training.

This is a concrete implementation of BaseForgeActor for supervised fine-tuning.
"""

import contextlib
import logging

import torch
import torchtitan.experiments.forge.train_spec as forge_train_spec
from apps.sft.actor import BaseForgeActor
from apps.sft.utils import (
    create_context_parallel_context,
    log_training_step,
    move_batch_to_device,
    setup_eval_dataloaders,
    setup_sft_dataloader,
    setup_tokenizer,
)
from forge.data.utils import StopAfterOneEpoch
from monarch.actor import endpoint
from omegaconf import DictConfig

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class TrainerActor(BaseForgeActor):
    """
    Concrete trainer actor for supervised fine-tuning.

    Handles training loop, forward/backward passes, and checkpoint management.

    Args:
        config: Configuration dictionary containing training settings
    """

    train_spec: forge_train_spec.ForgeTrainSpec
    train_dataloader: any
    num_training_steps: int

    def __init__(self, config: DictConfig):
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

        # Setup training dataloader
        self.train_dataloader = setup_sft_dataloader(
            tokenizer=self.tokenizer,
            dataset_path="yahma/alpaca-cleaned",
            dataset_split="train",
            target_tokens_per_pack=self.job_config.training.seq_len,
            batch_size=self.job_config.training.local_batch_size,
            device=self.device,
        )

        # Setup evaluation dataloaders if configured
        eval_config = self.job_config.get("eval", {})
        self.val_dataloaders = {}
        self.eval_every_n_steps = eval_config.get("eval_every_n_steps")
        max_eval_steps = eval_config.get("max_eval_steps")
        self.max_eval_steps = (
            max_eval_steps if max_eval_steps and max_eval_steps > 0 else None
        )
        self.validation_enabled = (
            self.eval_every_n_steps is not None and self.eval_every_n_steps > 0
        )

        if self.validation_enabled:
            logger.info("Setting up eval datasets...")
            eval_datasets_config = eval_config.get("datasets", [])
            self.val_dataloaders = setup_eval_dataloaders(
                tokenizer=self.tokenizer,
                eval_datasets_config=eval_datasets_config,
                target_tokens_per_pack=self.job_config.training.seq_len,
                batch_size=self.job_config.training.local_batch_size,
                device=self.device,
            )
            logger.info(f"Loaded {len(self.val_dataloaders)} eval datasets")

        # Load checkpoint if exists
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

            # Run evaluation periodically if enabled
            if (
                self.validation_enabled
                and self.current_step % self.eval_every_n_steps == 0
            ):
                await self.evaluate()

            if self.checkpointer:
                self.checkpointer.save(
                    curr_step=self.current_step,
                    last_step=self.current_step == self.num_training_steps,
                )

        # Final evaluation
        if self.validation_enabled:
            logger.info("Running final evaluation at end of training...")
            await self.evaluate()

        logger.info("Training complete!")

    async def evaluate(self) -> None:
        """
        Run evaluation on multiple datasets, one at a time.

        1. Set models to eval mode
        2. For each eval dataset:
            - Create fresh iterator (starts from epoch 0)
            - Use StopAfterOneEpoch to iterate until epoch boundary
            - Respect max_eval_steps cap if configured
            - Record loss and step metrics
        3. Restore models to train mode
        """
        logger.info("==Starting evaluation==")

        # Set models to eval mode
        for model_part in self.model_parts:
            model_part.eval()

        # Get DP mesh for epoch synchronization
        dp_mesh = None
        if self.parallel_dims is not None and self.parallel_dims.dp_enabled:
            dp_mesh = self.parallel_dims.world_mesh.get_group("dp")

        # For non-PP: disable gradients to save memory
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

            total_loss = torch.tensor(0.0, device=self.device)
            num_steps = 0

            # NOTE: Assumes batch contains field "metrics" containing "num_epochs"
            batch_iter = StopAfterOneEpoch(
                iter=iter(val_dataloader),  # Fresh iterator from epoch 0
                device=self.device,
                dp_mesh=dp_mesh,
            )

            with maybe_no_grad:
                for batch in batch_iter:
                    # If max_eval_steps>len(dataset), it will be stopped earlier
                    if (
                        self.max_eval_steps is not None
                        and num_steps >= self.max_eval_steps
                    ):
                        logger.info(
                            f"[{dataset_name}] Reached max_eval_steps cap of {self.max_eval_steps}"
                        )
                        break

                    # Move batch to device
                    batch = move_batch_to_device(batch, self.device)

                    # Forward pass only (no backward)
                    labels = batch.pop("labels")
                    loss = self.forward_backward_eval(batch, labels)
                    total_loss += loss
                    num_steps += 1

                    logger.info(
                        f"[dataset {dataset_name}] Step {num_steps} | Loss: {loss.item():.4f}"
                    )

            # Log average loss for this dataset
            avg_loss = (total_loss / max(num_steps, 1)).item()
            all_dataset_losses.append(avg_loss)
            all_dataset_steps.append(num_steps)
            logger.info(
                f"[dataset {dataset_name}] Final Step {num_steps} | Avg Loss: {avg_loss:.4f}"
            )

        # Record macro and micro average losses across datasets
        if len(all_dataset_losses) > 1:
            # Macro: same weight for all datasets
            macro_avg_loss = sum(all_dataset_losses) / len(all_dataset_losses)
            logger.info(f"Macro avg loss (unweighted): {macro_avg_loss:.4f}")

            # Micro: weighted mean by dataset size
            total_steps = sum(all_dataset_steps)
            micro_avg_loss = (
                sum(
                    loss * steps
                    for loss, steps in zip(all_dataset_losses, all_dataset_steps)
                )
                / total_steps
            )
            logger.info(f"Micro avg loss (weighted): {micro_avg_loss:.4f}")

        # Restore train mode
        for model_part in self.model_parts:
            model_part.train()

        logger.info("==Evaluation complete==")

    def forward_backward_eval(
        self, input_dict: dict[str, torch.Tensor], labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Perform forward pass only (for evaluation).

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
                    self.pp_schedule.step(inputs, target=targets, losses=losses)
                else:
                    self.pp_schedule.step(target=targets, losses=losses)

            loss = (
                torch.sum(torch.stack(losses)).to(self.device)
                if self.pp_has_last_stage
                else torch.tensor(-1.0, device=self.device)
            )
        else:
            with self.train_context(optional_context_parallel_ctx):
                assert len(model_parts) == 1
                with self.maybe_enable_amp:
                    pred = model_parts[0](inputs)
                    loss = self.loss_fn(pred, labels)
                del pred

        return loss

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
