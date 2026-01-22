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


def pg_soft_gate(
    ratio: torch.Tensor,
    advantages: torch.Tensor,
    mask: torch.Tensor,
    tau_pos: float = 1.0,
    tau_neg: float = 1.05,
) -> tuple[torch.Tensor, list[Metric]]:
    """SAPO's soft sigmoid gating.

    Reference: Gao et al., "Soft Adaptive Policy Optimization" (2025).
    https://arxiv.org/abs/2511.20347

    Formula: gate(r) = (4/τ) * sigmoid(τ * (r - 1))
             L = -gate(r) * A

    Replaces PPO's hard clipping with smooth sigmoid decay. The 4/τ normalization
    ensures the GRADIENT ∂gate/∂r = 1.0 at r=1, matching vanilla policy gradient
    on-policy. As r deviates from 1, the gate decays smoothly toward 0.

    Asymmetric temperature: τ_neg > τ_pos makes the gate decay faster for
    negative advantages. When decreasing a token's probability (negative
    advantage), that probability mass redistributes across the entire vocabulary.
    This one-to-many effect amplifies noise in negative updates. A higher τ_neg
    compensates by applying a tighter trust region for negative advantages.

    Args:
        ratio (torch.Tensor): Importance ratio (B, S).
        advantages (torch.Tensor): Advantage estimates (B, S).
        mask (torch.Tensor): Valid token mask (B, S).
        tau_pos (float): Temperature for positive advantages (default 1.0).
        tau_neg (float): Temperature for negative advantages (default 1.05).

    Returns:
        tuple[torch.Tensor, list[Metric]]: Soft-gated loss (B, S).
    """
    pos_gate = (4.0 / tau_pos) * torch.sigmoid(tau_pos * (ratio - 1))
    neg_gate = (4.0 / tau_neg) * torch.sigmoid(tau_neg * (ratio - 1))
    gate = torch.where(advantages > 0, pos_gate, neg_gate)
    pg_loss = -gate * advantages

    with torch.no_grad():
        metrics = [
            Metric(
                key="loss/soft_gate/gate/mean",
                value=masked_mean(gate, mask),
                reduction=Reduce.MEAN,
            ),
        ]

    return pg_loss, metrics


class SAPOLoss(BaseLossConfig):
    """SAPO: Soft Adaptive Policy Optimization.

    Reference: Gao et al., "Soft Adaptive Policy Optimization" (2025).
    https://arxiv.org/abs/2511.20347

    Per-token: L_t = -gate(r) * A
    Aggregated: L = mean over sequences of (mean over tokens of L_t)

    where:
        gate(r) = (4/τ) * sigmoid(τ * (r - 1))   — soft sigmoid gate
        τ = τ_pos if A > 0, else τ_neg           — asymmetric temperature
        r = π_θ/π_old                            — importance ratio
        A = (R - mean(R)) / std(R)

    SAPO replaces PPO's hard clipping with smooth sigmoid gating. The 4/τ factor
    is chosen so that the effective gradient scaling equals 1.0 at r=1 (on-policy).
    As r deviates from 1, the gate decays smoothly toward 0.

    Asymmetric temperature: τ_neg > τ_pos makes the gate decay faster for
    negative advantages. When decreasing a token's probability (negative
    advantage), that probability mass redistributes across the entire vocabulary.
    This one-to-many effect amplifies noise in negative updates. A higher τ_neg
    compensates by applying a tighter trust region for negative advantages.

    Differences from GRPO:
        1. Soft gating: No discontinuity at clip boundary. Gradients decay
           smoothly rather than dropping to zero.

    Args:
        tau_pos (float): Temperature for positive advantages (default 1.0).
        tau_neg (float): Temperature for negative advantages (default 1.05).
        agg_type (AggType): Aggregation method (default "sequence_mean").
    """

    tau_pos: Annotated[float, Field(gt=0)] = 1.0
    tau_neg: Annotated[float, Field(gt=0)] = 1.05
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
            logprobs, generator_logprobs, loss_mask, ratio_type="token"
        )
        pg_loss, gate_m = pg_soft_gate(
            ratio, advantages, loss_mask, self.tau_pos, self.tau_neg
        )
        loss, agg_m = aggregate(pg_loss, loss_mask, self.agg_type, loss_scale)

        return LossOutput(loss, lp_m + ent_m + ratio_m + gate_m + agg_m)
