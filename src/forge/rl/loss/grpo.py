# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Annotated

import torch
from forge.observability.metrics import Metric
from forge.rl.loss.ops import (
    aggregate,
    compute_entropy,
    compute_kl,
    compute_logprobs,
    compute_ratio,
    pg_ppo_clip,
)
from forge.rl.loss.types import AggType, BaseLossConfig, LossOutput
from pydantic import Field


class GRPOLoss(BaseLossConfig):
    """DR-GRPO: "Done Right" GRPO with unbiased aggregation.

    Reference: Liu et al., "Understanding R1-Zero-Like Training" (2025).
    https://arxiv.org/abs/2503.20783

    Per-token: L_t = max(-r*A, -clip(r, 1-ε, 1+ε)*A) + β*KL
    Aggregated: L = sum(L_t * mask) / (B * MAX_LEN)

    where:
        r = π_θ(y_t|q,y_<t) / π_old(y_t|q,y_<t)  — importance ratio
        A = R - mean(R)                          - No std norm, to avoid difficulty bias
        KL = r_ref - log(r_ref) - 1              — k3 estimator, r_ref = π_ref/π_θ
        B * MAX_LEN = fixed denominator batch_size * max sequence length

    GRPO replaces PPO's learned value function with group-relative advantages.
    Sample multiple responses per prompt, compute advantages by comparing rewards
    within each group. This eliminates the need for a separate critic model at
    the cost of sampling more responses.

    DR-GRPO fixes two biases in vanilla GRPO:
    1. Length bias: GRPO divides by |o_i|, i.e. agg_type='sequence_mean',
       rewarding the model for producing shorter correct and longer incorrect sequences,
       resulting in unnecessarily increased lengths during training.
       DR-GRPO uses agg_type='fixed_horizon' to remove this bias, dividing by a constant
       denominator (sequence dimension size) instead.
    2. Difficulty bias: GRPO normalizes advantages by std, over-weighting easy
       problems with low variance. DR-GRPO uses mean-only advantages. NOTE:
       This should be changed at the **advantage** computation level.

    NOTE: Default sets clip_high>clip_low, as this reportedly better, although not
    explored in the original paper.

    Args:
        clip_low (float): Lower clip bound (default 0.2).
        clip_high (float): Upper clip bound (default 0.28).
        beta (float): KL penalty coefficient (default 0.1).
        agg_type (AggType): Aggregation method (default "fixed_horizon").
    """

    clip_low: Annotated[float, Field(ge=0, le=1)] = 0.2
    clip_high: Annotated[float, Field(ge=0, le=1)] = 0.28
    beta: Annotated[float, Field(ge=0)] = 0.1
    agg_type: AggType = "fixed_horizon"

    def __call__(
        self,
        logits: torch.Tensor,  # (B, S, V)
        target_ids: torch.Tensor,  # (B, S)
        advantages: torch.Tensor,  # (B, S)
        generator_logprobs: torch.Tensor,  # (B, S)
        loss_mask: torch.Tensor,  # (B, S)
        ref_logprobs: torch.Tensor | None = None,  # (B, S) or None
        loss_scale: torch.Tensor | None = None,
    ) -> LossOutput:
        logprobs, lp_m = compute_logprobs(logits, target_ids)
        entropy, ent_m = compute_entropy(logits, loss_mask)  # logging only
        ratio, log_ratio, ratio_m = compute_ratio(
            logprobs, generator_logprobs, loss_mask, ratio_type="token"
        )
        pg_loss, clip_m = pg_ppo_clip(
            ratio, advantages, loss_mask, self.clip_low, self.clip_high
        )

        kl_m: list[Metric] = []
        if self.beta > 0:
            if ref_logprobs is None:
                raise ValueError("ref_logprobs required when beta > 0")
            kl, kl_m = compute_kl(logprobs, ref_logprobs, loss_mask)
            pg_loss = pg_loss + self.beta * kl

        loss, agg_m = aggregate(pg_loss, loss_mask, self.agg_type, loss_scale)

        return LossOutput(loss, lp_m + ent_m + ratio_m + clip_m + kl_m + agg_m)
