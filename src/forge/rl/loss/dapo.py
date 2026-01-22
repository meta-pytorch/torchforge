# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Annotated

import torch
from forge.observability.metrics import Metric, Reduce
from forge.rl.loss.ops import (
    aggregate,
    compute_entropy,
    compute_logprobs,
    compute_ratio,
    masked_mean,
    pg_ppo_clip,
)
from forge.rl.loss.types import AggType, BaseLossConfig, LossOutput
from pydantic import Field


def pg_dual_clip(
    pg_loss: torch.Tensor,
    advantages: torch.Tensor,
    mask: torch.Tensor,
    c: float = 3.0,
) -> tuple[torch.Tensor, list[Metric]]:
    """DAPO's dual-clip for negative advantages.

    Reference: Yu et al., "DAPO: An Open-Source LLM Reinforcement Learning System at Scale" (2025).
    https://arxiv.org/abs/2503.14476

    Formula: L = min(L_PPO, -c*A) when A < 0

    Standard PPO clipping can over-penalize bad actions, especially in reasoning
    tasks where some "wrong" tokens are actually productive exploration. Dual-clip
    adds a ceiling: penalties on negative-advantage tokens cannot exceed c times
    the advantage magnitude.

    Args:
        pg_loss (torch.Tensor): Per-token PPO loss from pg_ppo_clip (B, S).
        advantages (torch.Tensor): Advantage estimates (B, S).
        mask (torch.Tensor): Valid token mask (B, S).
        c (float): Dual-clip constant (default 3.0).

    Returns:
        tuple[torch.Tensor, list[Metric]]: Dual-clipped loss (B, S).
    """
    dual_clip_bound = -c * advantages
    loss = torch.where(
        advantages < 0,
        torch.minimum(pg_loss, dual_clip_bound),
        pg_loss,
    )

    with torch.no_grad():
        neg_mask = (advantages < 0) & mask.bool()
        was_dual_clipped = (pg_loss > dual_clip_bound) & neg_mask
        metrics = [
            Metric(
                key="loss/dual_clip/clip_fraction",
                value=masked_mean(was_dual_clipped.float(), mask),
                reduction=Reduce.MEAN,
            ),
        ]

    return loss, metrics


class DAPOLoss(BaseLossConfig):
    """DAPO: Decoupled clip + Dynamic sAmpling Policy Optimization.

    Reference: Yu et al., "DAPO: An Open-Source LLM Reinforcement Learning System at Scale" (2025).
    https://arxiv.org/abs/2503.14476

    Per-token:
        L_clip = max(-r*A, -clip(r, 1-ε_low, 1+ε_high)*A)
        L_t = min(L_clip, -c*A) when A < 0, else L_clip
    Aggregated: L = sum(L_t * mask) / sum(mask)

    where:
        r = π_θ/π_old                            — importance ratio
        A = (R - mean(R)) / std(R)
        ε_high > ε_low                           — asymmetric clip (more exploration)
        c = dual-clip cap penalty

    Differences from GRPO:
    - Clip-higher: ε_high > ε_low allows more exploration for low-probability tokens.
    - Dual-clip: Caps penalty on negative advantages to prevent over-penalization.
    - Token-level aggregation: Divides by total trainable tokens across all sequences.

    NOTE: DAPO paper also introduces preprocessing techniques not in this loss:
    - Dynamic Sampling: Filters groups where all responses have same reward.
    - Overlong Reward Shaping: Filters truncated sequences + soft length penalty.

    Args:
        clip_low (float): Lower clip bound (default 0.2).
        clip_high (float): Upper clip bound (default 0.28).
        dual_clip_c (float): Dual-clip constant (default 3.0).
        agg_type (AggType): Aggregation method (default "token_mean").
    """

    clip_low: Annotated[float, Field(ge=0, le=1)] = 0.2
    clip_high: Annotated[float, Field(ge=0, le=1)] = 0.28
    dual_clip_c: Annotated[float, Field(ge=1)] = 3.0
    agg_type: AggType = "token_mean"

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
            logprobs, generator_logprobs, loss_mask, ratio_type="token"
        )
        pg_loss, clip_m = pg_ppo_clip(
            ratio, advantages, loss_mask, self.clip_low, self.clip_high
        )
        pg_loss, dual_m = pg_dual_clip(pg_loss, advantages, loss_mask, self.dual_clip_c)
        loss, agg_m = aggregate(pg_loss, loss_mask, self.agg_type, loss_scale)

        return LossOutput(loss, lp_m + ent_m + ratio_m + clip_m + dual_m + agg_m)
