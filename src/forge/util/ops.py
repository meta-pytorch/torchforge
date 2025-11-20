# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F

from forge.data.common import CROSS_ENTROPY_IGNORE_IDX


def compute_logprobs(
    logits: torch.Tensor,
    targets: torch.Tensor,
    temperature: float = 1.0,
    ignore_index: int = CROSS_ENTROPY_IGNORE_IDX,
) -> torch.Tensor:
    """
    Computes the log probabilities of target tokens given the model logits.

    Args:
        logits: Model logits [batch, seq_len, vocab]
        targets: Target token IDs [batch, seq_len]
        temperature: Temperature for scaling
        ignore_index: Positions with this value in targets are masked (get 0.0 logprob)

    Returns:
        logprobs: [batch, seq_len] - Positions with ignore_index automatically get 0.0
    """
    scaled_logits = logits / temperature
    scaled_logits_fp32 = scaled_logits.float()

    batch_size, seq_len, vocab_size = scaled_logits_fp32.shape
    logprobs = -F.cross_entropy(
        scaled_logits_fp32.reshape(-1, vocab_size),
        targets.reshape(-1).long(),
        reduction="none",
        ignore_index=ignore_index,
    )

    return logprobs.reshape(batch_size, seq_len)


def create_shifted_targets(
    input_ids: torch.Tensor,
    loss_mask: torch.Tensor | None = None,
    ignore_index: int = CROSS_ENTROPY_IGNORE_IDX,
) -> torch.Tensor:
    """
    Create next-token prediction targets using torch.roll.
    Maintains same shape as input_ids.

    Args:
        input_ids: [batch, seq_len] or [seq_len] - Input token IDs
        loss_mask: [batch, seq_len] or [seq_len] - Trainable positions (bool or float)
                   If None, all positions are trainable
        ignore_index: Value for masked positions (default: -100)

    Returns:
        targets: Same shape as input_ids
                 targets[i] = input_ids[i+1] where trainable, else ignore_index
    """
    if input_ids.dim() == 1:
        # 1D case
        targets = torch.roll(input_ids, shifts=-1, dims=0)
        targets[-1] = ignore_index  # Last position wraps, mask it
    else:
        # 2D case (batched)
        targets = torch.roll(input_ids, shifts=-1, dims=-1)
        targets[:, -1] = ignore_index  # Last position wraps, mask it

    if loss_mask is not None:
        loss_mask = loss_mask.to(input_ids.device)
        targets = torch.where(
            loss_mask.bool(), targets, torch.full_like(targets, ignore_index)
        )

    return targets
