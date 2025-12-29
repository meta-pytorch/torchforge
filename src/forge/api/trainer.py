# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Type definitions and trainer protocol for the Forge API."""

from dataclasses import dataclass
from typing import Any, Callable, Protocol, runtime_checkable, TypeAlias

import torch


# Loss function signature: takes model outputs (as dict) and batch, returns scalar loss
# The dict will typically contain logits, but may include other keys depending on use case.
LossFn: TypeAlias = Callable[[dict[str, Any], "TextTrainBatch"], torch.Tensor]


@dataclass
class TextTrainBatch:
    """A batch of text training data for forward_backward.

    This dataclass defines the standard format for text training batches across all
    Forge text trainers.

    Attributes:
        input_ids: Input token IDs. Shape: [batch_size, seq_len]
        target_ids: Target token IDs for loss computation. Shape: [batch_size, seq_len]
        target_mask: Mask indicating which tokens to compute loss on.
            Shape: [batch_size, seq_len]. Values are 0 (ignore) or 1 (compute loss).
            If None, computes loss on all tokens.
        target_weights: Per-token weights for loss computation.
            Shape: [batch_size, seq_len]. Used for importance weighting, such as
            advantages in RL (GRPO, PPO) or custom loss weighting schemes.
            If None, all tokens have weight 1.0.

    Example:
        >>> batch = TextTrainBatch(
        >>>     input_ids=torch.tensor([[1, 2, 3, 4, 5]]),
        >>>     target_ids=torch.tensor([[2, 3, 4, 5, 6]]),
        >>>     target_mask=torch.tensor([[0, 0, 1, 1, 1]]),  # Only predict last 3 tokens
        >>>     target_weights=torch.tensor([[0, 0, 1.0, 0.8, 1.2]]),  # Weight by advantage
        >>> )
        >>> result = await trainer.forward_backward(batch)
    """

    input_ids: torch.Tensor
    target_ids: torch.Tensor
    target_mask: torch.Tensor | None = None
    target_weights: torch.Tensor | None = None


@dataclass
class ForwardBackwardResult:
    """Result from a forward_backward pass.

    Attributes:
        loss: Loss value computed for the batch
        metrics: Additional metrics computed during training (e.g., perplexity,
            accuracy, KL divergence). May be empty if no additional metrics are tracked.
            Values can be scalars, tensors, or other structured data depending on the loss.

    Example:
        >>> result = await trainer.forward_backward(batch)
        >>> result.loss
        0.3542
        >>> result.metrics
        {"perplexity": 1.42, "kl_divergence": 0.05}
    """

    loss: float
    metrics: dict[str, Any]


@dataclass
class OptimStepResult:
    """Result from an optimizer step.

    Attributes:
        step: Training step number after this optimizer step
        learning_rate: Current learning rate used for this step
        accumulated_microbatches: Number of forward_backward calls that were
            accumulated before this optimizer step. Useful for tracking gradient
            accumulation behavior.

    Example:
        >>> result = await trainer.optim_step()
        >>> result.step
        1000
        >>> result.learning_rate
        0.0001
        >>> result.accumulated_microbatches
        4
    """

    step: int
    learning_rate: float
    accumulated_microbatches: int


@dataclass
class ParallelismConfig:
    """Parallelism configuration for distributed training.

    Attributes:
        dp_degree: Data parallel degree (number of data parallel replicas)
        tp_degree: Tensor parallel degree (model sharding across devices)
        pp_degree: Pipeline parallel degree (model sharding across pipeline stages)
        cp_degree: Context parallel degree (sequence parallelism for long contexts)
        ep_degree: Expert parallel degree (for MoE models)
        world_size: Total number of processes in the distributed training job
        dp_rank: Current data parallel rank (0 to dp_degree-1)
        tp_rank: Current tensor parallel rank (0 to tp_degree-1)
        device: Device identifier (e.g., "cuda:0", "cuda:1")

    Example:
        >>> config = await trainer.get_config()
        >>> config.parallelism.dp_degree
        4
        >>> config.parallelism.tp_degree
        2
        >>> config.parallelism.pp_degree
        1
        >>> config.parallelism.cp_degree
        1
        >>> config.parallelism.ep_degree
        1
        >>> config.parallelism.device
        "cuda:0"
    """

    dp_degree: int
    tp_degree: int
    pp_degree: int
    cp_degree: int
    ep_degree: int
    world_size: int
    dp_rank: int
    tp_rank: int
    device: str


@dataclass
class TrainerConfig:
    """Static trainer and model configuration.

    This contains configuration information that doesn't change during training.

    Attributes:
        model_name: Name or path of the model being trained
        model_config: Model architecture configuration. Common keys include:
            - vocab_size: int - Size of the vocabulary
            - hidden_size: int - Hidden dimension size
            - num_layers: int - Number of transformer layers
            - num_attention_heads: int - Number of attention heads
            - max_seq_len: int - Maximum sequence length
        parallelism: Parallelism configuration for distributed training

    Example:
        >>> config = await trainer.get_config()
        >>> config.model_name
        "Qwen/Qwen2.5-7B"
        >>> config.model_config["vocab_size"]
        151936
        >>> config.parallelism.dp_degree
        4
    """

    model_name: str
    model_config: dict[str, Any]
    parallelism: ParallelismConfig


@dataclass
class TrainerStatus:
    """Runtime status of the trainer.

    This contains dynamic information about the trainer's current state that
    changes during training.

    Attributes:
        step: Current training step
        accumulated_microbatches: Number of batches accumulated since the last
            optim_step. Will be 0 if gradients were just applied/cleared.

    Example:
        >>> status = await trainer.get_status()
        >>> status.step
        1000
        >>> status.accumulated_microbatches
        2
    """

    step: int
    accumulated_microbatches: int


@runtime_checkable
class Trainer(Protocol):
    """Protocol defining the standard interface for all Forge trainers.

    Trainer implementations are expected to accept a default loss function at
    initialization time. This loss function is used when loss_fn is not
    provided to forward_backward(). The default loss should follow the
    LossFn signature.
    """

    async def forward_backward(
        self, batch: TextTrainBatch, loss_fn: LossFn | None = None
    ) -> ForwardBackwardResult:
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

        Custom loss function for specific batches:
            >>> def custom_loss(outputs: dict[str, Any], batch: TextTrainBatch) -> torch.Tensor:
            >>>     # Custom loss computation (e.g., PPO clip, DPO, cut cross entropy, etc.)
            >>>     logits = outputs["logits"]
            >>>     # ... compute loss from logits, or use other outputs like hidden_states
            >>>     return loss
            >>>
            >>> result = await trainer.forward_backward(batch, loss_fn=custom_loss)

        Args:
            batch: TextTrainBatch containing input_ids, target_ids, and optional
                target_mask/target_weights. See forge.api.trainer.TextTrainBatch for details.
            loss_fn: Optional custom loss function. If None, uses the loss function
                configured at trainer creation. Signature: (outputs, batch) -> loss
                where outputs is a dict with at least "logits" key.
                Useful for mixed training objectives or experimentation.

        Returns:
            ForwardBackwardResult containing loss and metrics

        Note:
            The default loss function is configured at trainer creation time via the
            `loss` parameter. The `loss_fn` parameter here allows per-batch override.
            All loss functions should accept (outputs: dict[str, Any], batch: TextTrainBatch)
            where outputs contains at minimum a "logits" key.
        """
        ...

    async def optim_step(self) -> OptimStepResult:
        """Apply optimizer step using accumulated gradients, then clear gradients.

        This method:
        1. Applies accumulated gradients via the optimizer
        2. Steps the learning rate scheduler
        3. Clears all gradients (zero_grad)
        4. Increments the training step counter
        5. May trigger automatic checkpointing (implementation-dependent)

        Gradients must have been accumulated via forward_backward() calls before
        calling this method.

        Returns:
            OptimStepResult containing step number, learning rate, and accumulated batch count

        Example:
            >>> # Accumulate over 4 batches
            >>> for batch in batches[:4]:
            >>>     await trainer.forward_backward(batch)
            >>> result = await trainer.optim_step()
            >>> result.step
            1000
            >>> result.learning_rate
            0.0001
            >>> result.accumulated_microbatches
            4
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

    async def forward(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
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
            Model output logits. Shape: [batch_size, seq_len, vocab_size]

        Note:
            This runs in torch.no_grad() context - no gradients are computed.

        Example:
            >>> eval_batch = {"input_ids": torch.tensor([[1, 2, 3, 4]])}
            >>> logits = await trainer.forward(eval_batch)  # [1, 4, vocab_size]
            >>> predictions = logits.argmax(dim=-1)  # [1, 4]
        """
        ...

    async def save(
        self,
        name: str | None = None,
        path: str | None = None,
        weights_only: bool = False,
    ) -> str:
        """Save trainer state or weights to persistent storage.

        By default, saves complete training state (model weights, optimizer state,
        learning rate scheduler state, and step counter). Set weights_only=True to
        save only model weights for inference/deployment.

        Args:
            name: Optional checkpoint name/identifier. If None, uses the current
                step number (e.g., "step-1000" or "weights-step-1000").
            path: Optional base directory or URI where checkpoint should be saved.
                If None, uses the default checkpoint directory configured at trainer
                creation. Supports different backends via URI schemes:
                - `/local/path` - local filesystem
                - `ts://key` - TorchStore
                - `s3://bucket/key` - S3
            weights_only: If True, saves only model weights (lighter, for inference).
                If False (default), saves full training state including optimizer.


        Returns:
            Full path/URI where checkpoint was saved

        Example:
            >>> # Save full training state (default)
            >>> path = await trainer.save(name="checkpoint-1000")
            >>> path
            "/default/checkpoint-1000"
            >>>
            >>> # Save weights only for inference
            >>> path = await trainer.save(name="policy-v1", weights_only=True)
            >>> path
            "/default/policy-v1"
            >>>
            >>> # Save to TorchStore
            >>> path = await trainer.save(name="best", path="ts://checkpoints")
            >>> path
            "ts://checkpoints/best"
        """
        ...

    async def load(self, path: str | None = None) -> str:
        """Load a previously saved checkpoint.

        Restores training state from a checkpoint. Automatically handles both
        full checkpoints and weights-only checkpoints.

        Args:
            path: Optional path or URI to the checkpoint to load. If None, loads
                the most recent checkpoint from the default directory. Can be:
                - `/local/path/checkpoint` - local filesystem
                - `ts://key` - TorchStore
                - `s3://bucket/key` - S3

        Returns:
            Path/URI that was loaded

        Example:
            >>> # Load latest checkpoint from default location
            >>> path = await trainer.load()
            >>> path
            "/default/step-5000"
            >>>
            >>> # Load specific checkpoint by path
            >>> path = await trainer.load("/checkpoints/step-5000")
            >>> path
            "/checkpoints/step-5000"
            >>>
            >>> # Load from TorchStore
            >>> path = await trainer.load("ts://checkpoint-key")
            >>> path
            "ts://checkpoint-key"
        """
        ...

    async def get_config(self) -> TrainerConfig:
        """Get static trainer and model configuration.

        Returns configuration information that doesn't change during training.
        For runtime state like current step, use get_status() instead.

        Returns:
            TrainerConfig containing model name, model_config, and parallelism settings

        Example:
            >>> config = await trainer.get_config()
            >>> config.model_name
            "Qwen/Qwen2.5-7B"
            >>> config.model_config["vocab_size"]
            151936
            >>> config.parallelism.dp_degree
            4
            >>> config.parallelism.device
            "cuda:0"
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
            >>> status.step
            1000
            >>> status.accumulated_microbatches
            2
        """
        ...

    async def get_tokenizer(self):
        """Get the tokenizer associated with this model.

        Returns the tokenizer used for encoding/decoding text with this model.
        Useful for preprocessing inputs or decoding model outputs.

        Returns:
            PreTrainedTokenizer: The HuggingFace tokenizer for this model

        Example:
            >>> tokenizer = await trainer.get_tokenizer()
            >>> tokens = tokenizer.encode("Hello world")
            >>> text = tokenizer.decode([1, 2, 3, 4])
        """
        ...
