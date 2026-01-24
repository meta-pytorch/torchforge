# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Annotated

import torch
from forge.rl.loss.ops import (
    aggregate,
    compute_entropy,
    compute_logprobs,
    compute_ratio,
    pg_ppo_clip,
)
from forge.rl.loss.types import AggType, BaseLossConfig, LossOutput
from pydantic import Field


class GSPOLoss(BaseLossConfig):
    """GSPO: Group Sequence Policy Optimization.

    Reference: Zheng et al., "Group Sequence Policy Optimization" (2025).
    https://arxiv.org/abs/2507.18071

    Per-token: L_t = max(-s*A, -clip(s, max=1+ε)*A)
    Aggregated: L = mean_i(sum_t(L_t * mask) / sum_t(mask))

    where:
        s = exp(mean_t(log π_θ - log π_old))    — sequence-level ratio
        s_t = sg(s) * π_θ(y_t) / sg(π_θ(y_t))   — reparameterized for token gradients
        A = (R - mean(R)) / std(R)
        sg(·) = stop gradient (detach)

    Note: s_t has same VALUE as s in forward pass, but gradient flows through π_θ(y_t).

    GSPO computes one importance ratio per sequence instead of per token. This
    matches how rewards are actually assigned (per-response, not per-token),
    which reduces variance, especially for long sequences and MoE models.

    Differences from GRPO:
        1. Sequence-level ratio: Computes one ratio per sequence (geometric mean of
           token ratios) instead of per-token. Reduces variance for long sequences.

    Args:
        clip_low (float): Lower clip bound offset (default 0.2).
        clip_high (float): Upper clip bound offset (default 0.2).
        agg_type (AggType): Aggregation method (default "sequence_mean").
    """

    clip_low: Annotated[float, Field(ge=0, le=1)] = 0.2
    clip_high: Annotated[float, Field(ge=0, le=1)] = 0.2
    agg_type: AggType = "sequence_mean"

    def __call__(
        self,
        logits: torch.Tensor,  # (B, S, V)
        target_ids: torch.Tensor,  # (B, S)
        advantages: torch.Tensor,  # (B, S)
        generator_logprobs: torch.Tensor,  # (B, S)
        loss_mask: torch.Tensor,  # (B, S)
        loss_scale: torch.Tensor | None = None,
    ) -> LossOutput:
        logprobs, lp_m = compute_logprobs(logits, target_ids)
        entropy, ent_m = compute_entropy(logits, loss_mask)
        ratio, log_ratio, ratio_m = compute_ratio(
            logprobs, generator_logprobs, loss_mask, ratio_type="sequence"
        )
        pg_loss, clip_m = pg_ppo_clip(
            ratio, advantages, loss_mask, self.clip_low, self.clip_high
        )

        loss, agg_m = aggregate(pg_loss, loss_mask, self.agg_type, loss_scale)

        return LossOutput(loss, lp_m + ent_m + ratio_m + clip_m + agg_m)
