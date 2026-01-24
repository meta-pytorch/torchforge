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
)
from forge.rl.loss.types import AggType, BaseLossConfig, LossOutput
from pydantic import Field


def pg_cispo(
    ratio: torch.Tensor,
    logprobs: torch.Tensor,
    advantages: torch.Tensor,
    mask: torch.Tensor,
    clip_low: float = 1.0,
    clip_high: float = 5.0,
) -> tuple[torch.Tensor, list[Metric]]:
    """CISPO: Clipped Importance Sampling Policy Optimization.

    Reference: Chen et al., "MiniMax-M1: Scaling Test-Time Compute Efficiently with Lightning Attention" (2025).
    https://arxiv.org/abs/2506.13585

    Formula: L = -clip(r, 1-ε_low, 1+ε_high).detach() * A * logprobs

    Unlike PPO which uses the ratio directly in the surrogate objective, CISPO
    uses REINFORCE-style gradients: the ratio is detached and acts as an
    importance weight on -A * log(π). In long reasoning chains, some tokens have
    very high importance ratios because they represent reflective reasoning steps.
    PPO would zero out their gradients entirely, but CISPO preserves them (just
    weighted down by the clipped ratio).

    Paper recommendation: No lower clipping. Use clip_low=1.0 (min=0, no effective
    lower bound).

    Args:
        ratio (torch.Tensor): Importance ratio (B, S).
        logprobs (torch.Tensor): Log probs from current policy (B, S).
        advantages (torch.Tensor): Advantage estimates (B, S).
        mask (torch.Tensor): Valid token mask (B, S).
        clip_low (float): Lower clip bound offset (default 1.0, no effective clipping).
        clip_high (float): Upper clip bound offset (default 5.0).

    Returns:
        tuple[torch.Tensor, list[Metric]]: CISPO loss (B, S).
    """
    clipped_ratio = torch.clamp(ratio, min=1 - clip_low, max=1 + clip_high).detach()
    pg_loss = -clipped_ratio * advantages * logprobs

    with torch.no_grad():
        clipped_high = ratio > (1 + clip_high)
        clipped_low = ratio < (1 - clip_low)
        metrics = [
            Metric(
                key="loss/clip/clipped_ratio/mean",
                value=masked_mean(clipped_ratio, mask),
                reduction=Reduce.MEAN,
            ),
            Metric(
                key="loss/clip/high_fraction_unconditional",
                value=masked_mean(clipped_high.float(), mask),
                reduction=Reduce.MEAN,
            ),
            Metric(
                key="loss/clip/low_fraction_unconditional",
                value=masked_mean(clipped_low.float(), mask),
                reduction=Reduce.MEAN,
            ),
        ]

    return pg_loss, metrics


class CISPOLoss(BaseLossConfig):
    """CISPO: Clipped Importance Sampling Policy Optimization.

    Reference: Chen et al., "MiniMax-M1: Scaling Test-Time Compute Efficiently with Lightning Attention" (2025).
    https://arxiv.org/abs/2506.13585

    Per-token: L_t = -sg(clip(r, 1-ε_low, 1+ε_high)) * A * log π_θ
    Aggregated: L = sum(L_t * mask) / sum(mask)

    where:
        r = π_θ/π_old                            — importance ratio
        A = (R - mean(R)) / std(R)
        clip(r, 1-ε_low, 1+ε_high)               — clipping bounds
        sg(·) = stop gradient (detach)           — ratio is detached

    CISPO uses REINFORCE-style gradients with a clipped, detached importance
    weight. Unlike PPO where the gradient flows through the ratio, here it flows
    through logprobs. This preserves learning signal for high-ratio "reflective"
    tokens that PPO would completely clip away. In long reasoning chains, some
    tokens have very high importance ratios because they represent reflective
    reasoning steps. PPO would zero out their gradients, but CISPO preserves them
    (just weighted down).

    Paper recommendation: No lower clipping. Use clip_low=1.0 (min=0, no effective
    lower bound since ratio=exp()>=0).

    Differences from GRPO:
        1. REINFORCE-style: Ratio is detached; gradient flows through logprobs.
        2. Upper-only clipping (default): No lower bound, like GSPO.
        3. Token-level aggregation: Divides by total trainable tokens across all sequences.

    Args:
        clip_low (float): Lower clip bound offset (default 1.0,  effectively
            no lower clipping).
        clip_high (float): Upper clip bound offset (default 4.0).
        agg_type (AggType): Aggregation method (default "token_mean").
    """

    clip_low: Annotated[float, Field(ge=0)] = 1.0
    clip_high: Annotated[float, Field(ge=0)] = 4.0
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
        pg_loss, cispo_m = pg_cispo(
            ratio, logprobs, advantages, loss_mask, self.clip_low, self.clip_high
        )
        loss, agg_m = aggregate(pg_loss, loss_mask, self.agg_type, loss_scale)

        return LossOutput(loss, lp_m + ent_m + ratio_m + cispo_m + agg_m)
