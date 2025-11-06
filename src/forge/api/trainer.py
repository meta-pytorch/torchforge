# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Trainer protocol for Forge.

This module defines the unified training interface that all trainer implementations
must conform to.

"""

from typing import Any, Protocol, runtime_checkable

import torch

from forge.api.types import (
    ForwardBackwardResult,
    ForwardResult,
    OptimStepResult,
    TextTrainBatch,
    TrainerInfo,
    TrainerStatus,
)


@runtime_checkable
class Trainer(Protocol):
    """Protocol defining the standard interface for all Forge trainers."""

    async def forward_backward(self, batch: TextTrainBatch) -> ForwardBackwardResult:
        """Execute forward pass and backward pass for one batch of data.

        Basic usage - single batch per optimizer step:
            >>> batch = TextTrainBatch(
            >>>     input_ids=torch.tensor([[1, 2, 3, 4, 5]]),
            >>>     target_ids=torch.tensor([[2, 3, 4, 5, 6]]),
            >>> )
            >>> result = await trainer.forward_backward(batch)
            >>> await trainer.optim_step()  # Apply gradients

        To accumulate gradients over multiple batches before optimizer step:
            >>> await trainer.forward_backward(batch1)  # Accumulates
            >>> await trainer.forward_backward(batch2)  # Accumulates another batch
            >>> await trainer.optim_step()  # Apply all accumulated gradients

        Args:
            batch: TextTrainBatch containing input_ids, target_ids, and optional
                target_mask/target_weights. See forge.api.types.TextTrainBatch for details.

        Returns:
            ForwardBackwardResult containing loss and metrics

        Note:
            The loss function is configured at trainer creation time via the
            `loss` parameter, not passed to this method.
        """
        ...

    async def optim_step(self, params: dict[str, Any] | None = None) -> OptimStepResult:
        """Apply optimizer step using accumulated gradients, then clear gradients.

        This method:
        1. Applies accumulated gradients via the optimizer
        2. Steps the learning rate scheduler
        3. Clears all gradients (zero_grad)
        4. Increments the training step counter
        5. May trigger automatic checkpointing (implementation-dependent)

        Gradients must have been accumulated via forward_backward() calls before
        calling this method.

        Args:
            params: Optional optimizer parameters. Currently reserved for future use.
                Most implementations ignore this and use the optimizer config from
                trainer initialization.

        Returns:
            OptimStepResult containing step number, learning rate, and accumulated batch count

        Example:
            >>> # Accumulate over 4 batches
            >>> for batch in batches[:4]:
            >>>     await trainer.forward_backward(batch)
            >>> result = await trainer.optim_step()
            >>> print(f"Step {result.step}, LR {result.learning_rate:.2e}")
            >>> print(f"Accumulated {result.accumulated_microbatches} batches")
        """
        ...

    async def clear_gradients(self) -> None:
        """Clear accumulated gradients without applying them.

        Use this when you need to discard accumulated gradients without performing
        an optimizer step. Common scenarios:
        - Exception during gradient accumulation
        - Skipping a training step due to some condition
        - Recovering from OOM or other errors

        This is equivalent to calling optimizer.zero_grad() and resetting internal
        accumulation counters.

        Example - Error recovery:
            >>> try:
            >>>     for batch in batches:
            >>>         await trainer.forward_backward(batch)
            >>>     await trainer.optim_step()
            >>> except torch.cuda.OutOfMemoryError:
            >>>     await trainer.clear_gradients()  # Discard partial gradients
            >>>     # Retry with smaller batches

        Example - Conditional skip:
            >>> await trainer.forward_backward(batch)
            >>> if should_skip_step():
            >>>     await trainer.clear_gradients()  # Don't apply these gradients
            >>> else:
            >>>     await trainer.optim_step()
        """
        ...

    async def forward(self, inputs: dict[str, torch.Tensor]) -> ForwardResult:
        """Run forward pass only, without backward pass (for evaluation/inference).

        This method executes the model's forward pass without computing gradients.
        Useful for:
        - Evaluation on validation/test data
        - Getting model predictions/logits
        - Debugging model outputs

        Args:
            inputs: Dictionary containing model inputs. Typically includes:
                - input_ids: torch.Tensor [batch_size, seq_len]
                Other keys depend on the model architecture.

        Returns:
            ForwardResult containing model logits

        Note:
            This runs in torch.no_grad() context - no gradients are computed.

        Example:
            >>> eval_batch = {"input_ids": torch.tensor([[1, 2, 3, 4]])}
            >>> output = await trainer.forward(eval_batch)
            >>> logits = output.logits  # [1, 4, vocab_size]
            >>> predictions = logits.argmax(dim=-1)  # [1, 4]
        """
        ...

    async def save_state(
        self, name: str | None = None, path: str | None = None
    ) -> dict[str, Any]:
        """Save a checkpoint of the current trainer state.

        Saves the complete training state including model weights, optimizer state,
        learning rate scheduler state, and current step counter. This checkpoint
        can be loaded later to resume training from this exact point.

        Args:
            name: Optional checkpoint name/identifier. If None, uses the current
                step number (e.g., "step-1000").
            path: Optional base directory or URI where checkpoint should be saved.
                If None, uses the default checkpoint directory configured at trainer
                creation. Supports different backends via URI schemes:
                - `/local/path` - local filesystem
                - `ts://key` - TorchStore
                - `s3://bucket/key` - S3

        Location resolution:
            - Both provided: path/name (e.g., "/checkpoints" + "best" = "/checkpoints/best")
            - Only path: use path directly
            - Only name: default_dir/name
            - Neither: default_dir/step-{current_step}

        Returns:
            dict containing:
                - path: str - Full path where checkpoint was saved
                - step: int - Training step at which checkpoint was saved

        Example:
            >>> # Save to default location with step number
            >>> result = await trainer.save_state()  # => /default/step-1000
            >>>
            >>> # Save with custom name to default location
            >>> result = await trainer.save_state("best-model")  # => /default/best-model
            >>>
            >>> # Save to custom base directory
            >>> result = await trainer.save_state("final", "/custom/checkpoints")
            >>> # => /custom/checkpoints/final
        """
        ...

    async def load_state(self, path: str | None = None) -> dict[str, Any]:
        """Load a previously saved checkpoint.

        Restores the complete training state from a checkpoint, including model
        weights, optimizer state, learning rate scheduler state, and step counter.

        Args:
            path: Optional path or URI to the checkpoint to load. If None, loads
                the most recent checkpoint from the default directory. Can be:
                - `/local/path/checkpoint` - local filesystem
                - `ts://key` - TorchStore
                - `s3://bucket/key` - S3

        Returns:
            dict containing:
                - step: int - Training step from the loaded checkpoint
                - learning_rate: float - Learning rate from the loaded checkpoint

        Example:
            >>> # Load latest checkpoint from default location
            >>> result = await trainer.load_state()
            >>> print(f"Resumed from step {result['step']}")
            >>>
            >>> # Load specific checkpoint by path
            >>> result = await trainer.load_state("/checkpoints/step-5000")
            >>>
            >>> # Load from TorchStore
            >>> result = await trainer.load_state("ts://checkpoint-key")
        """
        ...

    async def save_weights(
        self, name: str | None = None, path: str | None = None
    ) -> dict[str, Any]:
        """Save model weights only (without optimizer/scheduler state).

        Saves only the model weights in a format suitable for inference/sampling.
        This is lighter weight than save_state() since it excludes training state
        like optimizer and scheduler.

        Args:
            name: Optional checkpoint name/identifier. If None, uses the current
                step number (e.g., "weights-step-1000").
            path: Optional base directory or URI where weights should be saved.
                If None, uses the default location configured at trainer creation.
                Supports different backends via URI schemes:
                - `/local/path` - local filesystem
                - `ts://key` - TorchStore
                - `s3://bucket/key` - S3

        Location resolution:
            - Both provided: path/name
            - Only path: use path directly
            - Only name: default_dir/name
            - Neither: default_dir/step-{current_step}

        Returns:
            dict containing:
                - path: str - Full URI where weights were saved
                - version: str | int - The name/version that was saved

        Example:
            >>> # Save to default location with step number
            >>> result = await trainer.save_weights()
            >>>
            >>> # Save to TorchStore for inference server
            >>> result = await trainer.save_weights("policy-v1", "ts://policy-weights")
            >>> # → ts://policy-weights/policy-v1
            >>>
            >>> # Save to S3
            >>> result = await trainer.save_weights(path="s3://bucket/models/final")
        """
        ...

    async def get_info(self) -> TrainerInfo:
        """Get static trainer and model metadata.

        Returns information about the trainer configuration and model architecture
        that doesn't change during training.

        Returns:
            TrainerInfo containing model name, step, config, and parallelism settings

        Example:
            >>> info = await trainer.get_info()
            >>> print(f"Training {info.model_name} at step {info.step}")
            >>> print(f"Vocab size: {info.config['vocab_size']}")
            >>> print(f"Data parallel degree: {info.parallelism['dp_degree']}")
        """
        ...

    async def get_status(self) -> TrainerStatus:
        """Get current runtime status of the trainer.

        Returns dynamic information about the trainer's current state that changes
        during training.

        Returns:
            TrainerStatus containing current step and accumulated batch count

        Example:
            >>> status = await trainer.get_status()
            >>> print(f"Current step: {status.step}")
            >>> if status.accumulated_microbatches > 0:
            >>>     print(f"Warning: {status.accumulated_microbatches} "
            >>>           f"batches accumulated without optimizer step")
        """
        ...

    def get_tokenizer(self):
        """Get the tokenizer associated with this model.

        Returns the tokenizer used for encoding/decoding text with this model.
        Useful for preprocessing inputs or decoding model outputs.

        Returns:
            PreTrainedTokenizer: The HuggingFace tokenizer for this model

        Note:
            This is a synchronous method (not async) since tokenizer access is
            typically fast and doesn't require remote calls.

        Example:
            >>> tokenizer = trainer.get_tokenizer()
            >>> tokens = tokenizer.encode("Hello world")
            >>> text = tokenizer.decode([1, 2, 3, 4])
        """
        ...
