# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Type definitions for the Forge API."""

from dataclasses import dataclass
from typing import Any, Callable, TypeAlias

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
