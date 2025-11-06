# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Type definitions for the Forge API."""

from dataclasses import dataclass
from typing import Any, Callable, TypeAlias

import torch


# Loss function signature: takes logits and batch, returns scalar loss
LossFn: TypeAlias = Callable[[torch.Tensor, "TextTrainBatch"], torch.Tensor]


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
            accuracy). May be empty if no additional metrics are tracked.

    Example:
        >>> result = await trainer.forward_backward(batch)
        >>> print(f"Loss: {result.loss:.4f}")
        >>> if result.metrics:
        >>>     print(f"Metrics: {result.metrics}")
    """

    loss: float
    metrics: dict[str, float]


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
        >>> print(f"Step {result.step}, LR={result.learning_rate:.2e}")
        >>> print(f"Accumulated {result.accumulated_microbatches} batches")
    """

    step: int
    learning_rate: float
    accumulated_microbatches: int


@dataclass
class ForwardResult:
    """Result from a forward pass (evaluation/inference).

    Attributes:
        logits: Model output logits (pre-softmax). Shape: [batch_size, seq_len, vocab_size]

    Example:
        >>> result = await trainer.forward(eval_batch)
        >>> predictions = result.logits.argmax(dim=-1)  # [batch_size, seq_len]
    """

    logits: torch.Tensor


@dataclass
class TrainerInfo:
    """Static trainer and model metadata.

    This contains information about the trainer configuration and model architecture
    that doesn't change during training.

    Note:
        The exact format of `config` and `parallelism` dicts depends on the underlying
        trainer implementation (TorchTitan, HuggingFace, etc.). The fields below
        document common keys, but implementations may include additional fields.

    Attributes:
        model_name: Name or path of the model being trained
        step: Current training step
        config: Model configuration. Common keys include:
            - vocab_size: int - Size of the vocabulary
            - hidden_size: int - Hidden dimension size
            - num_layers: int - Number of transformer layers
            - num_attention_heads: int - Number of attention heads
            - max_seq_len: int - Maximum sequence length
        parallelism: Parallelism configuration. Common keys include:
            - dp_degree: int - Data parallel degree
            - tp_degree: int - Tensor parallel degree
            - pp_degree: int - Pipeline parallel degree
            - dp_rank: int - Current data parallel rank
            - tp_rank: int - Current tensor parallel rank
            - device: str - Device identifier (e.g., "cuda:0")
            - gradient_accumulation_steps: int - Number of microbatches per step

    Example:
        >>> info = await trainer.get_info()
        >>> print(f"Training {info.model_name} at step {info.step}")
        >>> print(f"Vocab size: {info.config['vocab_size']}")
        >>> print(f"DP={info.parallelism['dp_degree']}, "
        >>>       f"TP={info.parallelism['tp_degree']}")
        >>> print(f"Device: {info.parallelism['device']}")
    """

    model_name: str
    step: int
    config: dict[str, Any]
    parallelism: dict[str, Any]


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
        >>> print(f"Current step: {status.step}")
        >>> if status.accumulated_microbatches > 0:
        >>>     print(f"Warning: {status.accumulated_microbatches} batches "
        >>>           f"accumulated without optimizer step")
    """

    step: int
    accumulated_microbatches: int
