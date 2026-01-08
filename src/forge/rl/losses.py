# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Annotated, Literal

import torch
import torch.nn.functional as F

from forge.observability.metrics import Metric, Reduce
from pydantic import BaseModel, ConfigDict, Field


# =============================================================================
# TYPE ALIASES
# =============================================================================

AggType = Literal["token_mean", "fixed_horizon", "sequence_mean"]
RatioType = Literal["token", "sequence"]
KLType = Literal["k1", "k2", "k3"]


# =============================================================================
# HELPERS
# =============================================================================


def masked_mean(
    values: torch.Tensor,
    mask: torch.Tensor,
    loss_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute masked mean: sum(values * mask) / divisor.

    Can be specially useful in distributed settings, where loss_scale is the global
    number of tokens / grad_avg_group_size. This ensures that normalization
    takes into account all tokens in the batch, not just the local ones.

    Args:
        values (torch.Tensor): Per-token values (B, S).
        mask (torch.Tensor): Valid token mask (B, S).
        loss_scale (torch.Tensor | None): If provided, use as divisor instead of mask.sum().

    Returns:
        torch.Tensor: Scalar mean.
    """
    masked_sum = (values * mask).sum()
    if loss_scale is not None:
        divisor = loss_scale.clamp(min=1.0)
    else:
        divisor = mask.sum().clamp(min=1.0)
    return masked_sum / divisor


# =============================================================================
# OUTPUT TYPES
# =============================================================================


@dataclass
class LossOutput:
    """Output from all loss functions.

    Attributes:
        loss (torch.Tensor): Scalar loss tensor for backpropagation.
        metrics (list[Metric]): List of Metric objects for distributed logging.
    """

    loss: torch.Tensor
    metrics: list[Metric]


# =============================================================================
# BASE CLASSES
# =============================================================================


class BaseLossConfig(BaseModel):
    """Base configuration for all policy gradient losses."""

    # extra="forbid": Raises error if user passes unknown fields (catches typos).
    # arbitrary_types_allowed=True: Allows torch.Tensor and other non-JSON types in fields.
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)


class PolicyGradientLoss(ABC):
    """Abstract base class for policy gradient losses."""

    @abstractmethod
    def __call__(self, logits: torch.Tensor, **kwargs) -> LossOutput:
        """Compute the policy gradient loss.

        Args:
            logits (torch.Tensor): Model output logits (B, S, V).
            **kwargs: Additional inputs specific to each loss.

        Returns:
            LossOutput
        """
        pass

    @property
    def name(self) -> str:
        """Return the loss name (class name without 'Loss' suffix)."""
        return self.__class__.__name__.replace("Loss", "")


# =============================================================================
# PRIMITIVES: Core Computation Functions
# =============================================================================


CROSS_ENTROPY_IGNORE_IDX = -100


def create_shifted_targets(
    input_ids: torch.Tensor,
    loss_mask: torch.Tensor | None = None,
    ignore_index: int = CROSS_ENTROPY_IGNORE_IDX,
) -> torch.Tensor:
    """Create next-token prediction targets using torch.roll.

    Maintains same shape as input_ids. For position i, target_ids[i] = input_ids[i+1].
    The last position is set to ignore_index since there's no next token.

    Optionally applies loss_mask: positions where loss_mask is 0 (or False) are set
    to ignore_index, so cross-entropy will ignore them.

    Args:
        input_ids: [batch, seq_len] or [seq_len] - Input token IDs.
        loss_mask: [batch, seq_len] or [seq_len] - Positions to train on (1=train, 0=ignore).
            If None, all positions except last are trainable.
        ignore_index: Value for masked/last positions (default: -100).

    Returns:
        targets: Same shape as input_ids.
            targets[i] = input_ids[i+1] where trainable, else ignore_index.
    """
    targets = torch.roll(input_ids, shifts=-1, dims=-1)
    if input_ids.dim() == 1:
        targets[-1] = ignore_index
    else:
        targets[:, -1] = ignore_index

    if loss_mask is not None:
        targets = torch.where(
            loss_mask.bool(), targets, torch.full_like(targets, ignore_index)
        )

    return targets


def compute_logprobs(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    temperature: float = 1.0,
    ignore_index: int = CROSS_ENTROPY_IGNORE_IDX,
) -> tuple[torch.Tensor, list[Metric]]:
    """Compute log probabilities for sampled tokens via negative cross-entropy, given model logits output.

    Implementation note: Casts to fp32 before temperature division to preserve
    numerical precision when training with bf16/fp16.

    Args:
        logits (torch.Tensor): Model output logits (B, S, V).
        target_ids (torch.Tensor): Target token ids (B, S). Positions with ignore_index
            are returned as 0.
        temperature (float): Softmax temperature (default 1.0).
        ignore_index (int): Target value to ignore (default -100).

    Returns:
        tuple[torch.Tensor, list[Metric]]: logprobs is (B, S), metrics is empty list.
            Positions where target_ids == ignore_index have logprobs = 0.
    """
    # Cast to fp32 BEFORE dividing to preserve precision with bf16/fp16
    logits_fp32 = logits.float() / temperature
    B, S, V = logits_fp32.shape
    logprobs = -F.cross_entropy(
        logits_fp32.view(-1, V),
        target_ids.view(-1).long(),
        ignore_index=ignore_index,
        reduction="none",
    ).view(B, S)

    return logprobs, []


def compute_entropy(
    logits: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, list[Metric]]:
    """Compute per-token entropy.

    Formula: H = logsumexp(logits) - sum(softmax(logits) * logits)
        This is equivalent to -sum(p * log(p)) but numerically stable.

    Args:
        logits (torch.Tensor): Model output logits (B, S, V).
        mask (torch.Tensor): Valid token mask (B, S).

    Returns:
        tuple[torch.Tensor, list[Metric]]: entropy is (B, S).
    """
    logits_fp32 = logits.float()
    probs = F.softmax(logits_fp32, dim=-1)
    entropy = torch.logsumexp(logits_fp32, dim=-1) - (probs * logits_fp32).sum(dim=-1)

    with torch.no_grad():
        metrics = [
            Metric(
                key="loss/entropy/mean",
                value=masked_mean(entropy, mask),
                reduction=Reduce.MEAN,
            ),
        ]

    return entropy, metrics


# -----------------------------------------------------------------------------
# Ratio Computation
# -----------------------------------------------------------------------------


def compute_ratio(
    logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    mask: torch.Tensor,
    ratio_type: RatioType = "token",
) -> tuple[torch.Tensor, torch.Tensor, list[Metric]]:
    """Compute importance sampling ratio for off-policy correction.

    The ratio r = π_θ/π_old measures how much the current policy differs from
    the policy that generated the samples. This enables reusing samples from an
    old policy while adjusting for distribution shift.

    Formula:
        token:    r_t = exp(logprobs_t - old_logprobs_t)
        sequence: R_seq = exp(mean_t[logprobs - old_logprobs])

    Interpretation:
    - ratio = 1.0: on-policy (no distribution change)
    - ratio > 1.0: current policy assigns higher probability
    - ratio < 1.0: current policy assigns lower probability

    Token vs Sequence:
    - token: Per-token ratio. Standard approach, but variance accumulates
      quadratically with sequence length.
    - sequence: One ratio per sequence, broadcast to all tokens. Matches how
      rewards are assigned (per-response). Lower variance for long sequences.
      Uses reparameterization trick to maintain per-token gradient flow.

    Reference (sequence): Zheng et al., "GSPO" (arXiv:2507.18071, 2025).

    Args:
        logprobs (torch.Tensor): Log probs from current policy (B, S).
        old_logprobs (torch.Tensor): Log probs from sampling policy (B, S).
        mask (torch.Tensor): Valid token mask (B, S).
        ratio_type (RatioType): "token" for per-token ratio, "sequence" for sequence-level.

    Returns:
        tuple[torch.Tensor, torch.Tensor, list[Metric]]: (ratio, log_ratio, metrics). Both
            ratio and log_ratio are (B, S). For sequence type, values are broadcast from
            per-sequence computation.
    """
    if ratio_type == "token":
        log_ratio = logprobs - old_logprobs.detach()
        log_ratio = torch.clamp(log_ratio, min=-20.0, max=20.0)
        ratio = torch.exp(log_ratio)

    elif ratio_type == "sequence":
        token_log_ratio = logprobs - old_logprobs.detach()
        seq_lengths = mask.sum(dim=-1).clamp(min=1)
        seq_log_ratio = (token_log_ratio * mask).sum(dim=-1) / seq_lengths

        # Reparameterization: forward uses seq ratio, backward uses token grads
        log_ratio = logprobs - logprobs.detach() + seq_log_ratio.detach().unsqueeze(-1)
        log_ratio = torch.clamp(log_ratio, min=-20.0, max=20.0)
        ratio = torch.exp(log_ratio)

    else:
        raise ValueError(f"Unknown ratio_type: {ratio_type}")

    with torch.no_grad():
        metrics = [
            Metric(
                key="loss/ratio/mean",
                value=masked_mean(ratio, mask),
                reduction=Reduce.MEAN,
            ),
            Metric(
                key="loss/ratio/approx_kl",
                value=masked_mean(-log_ratio, mask),
                reduction=Reduce.MEAN,
            ),
        ]

    return ratio, log_ratio, metrics


# -----------------------------------------------------------------------------
# Policy Gradient Strategies
# -----------------------------------------------------------------------------


def pg_ppo_clip(
    ratio: torch.Tensor,
    advantages: torch.Tensor,
    mask: torch.Tensor,
    clip_low: float = 0.2,
    clip_high: float = 0.2,
) -> tuple[torch.Tensor, list[Metric]]:
    """PPO clipped surrogate objective.

    Reference: Schulman et al., "Proximal Policy Optimization" (2017).
    https://arxiv.org/abs/1707.06347

    Clips the importance ratio to prevent the policy from changing too much in
    a single update. The max() operator creates a "pessimistic" bound: we only
    take credit for improvement up to the clip boundary. This keeps updates
    within a trust region around the old policy.

    Formula: L = max(-r*A, -clip(r, 1-ε_low, 1+ε_high)*A)

    Args:
        ratio (torch.Tensor): Importance ratio π_θ/π_old (B, S).
        advantages (torch.Tensor): Advantage estimates (B, S).
        mask (torch.Tensor): Valid token mask (B, S).
        clip_low (float): Lower bound offset. Ratio is clamped to min of (1 - clip_low).
            E.g., clip_low=0.2 means ratio >= 0.8. Default: 0.2.
        clip_high (float): Upper bound offset. Ratio is clamped to max of (1 + clip_high).
            E.g., clip_high=0.2 means ratio <= 1.2. Default: 0.2.

    Returns:
        tuple[torch.Tensor, list[Metric]]: Per-token loss (B, S).
    """
    clipped_ratio = torch.clamp(ratio, 1 - clip_low, 1 + clip_high)
    unclipped_loss = -ratio * advantages
    clipped_loss = -clipped_ratio * advantages
    pg_loss = torch.maximum(unclipped_loss, clipped_loss)

    with torch.no_grad():
        clipped_high = (ratio > 1 + clip_high) & mask.bool()
        clipped_low = (ratio < 1 - clip_low) & mask.bool()

        # Advantage-conditioned clip metrics (VERL-style, more informative)
        pos_adv = advantages > 0
        neg_adv = advantages < 0
        metrics = [
            Metric(
                key="loss/ppo_clip/high_fraction",
                value=masked_mean((clipped_high & pos_adv).float(), mask),
                reduction=Reduce.MEAN,
            ),
            Metric(
                key="loss/ppo_clip/low_fraction",
                value=masked_mean((clipped_low & neg_adv).float(), mask),
                reduction=Reduce.MEAN,
            ),
        ]

    return pg_loss, metrics


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
                key="loss/dual_clip/fraction",
                value=masked_mean(was_dual_clipped.float(), mask),
                reduction=Reduce.MEAN,
            ),
        ]

    return loss, metrics


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
                key="loss/sapo_gate/mean",
                value=masked_mean(gate, mask),
                reduction=Reduce.MEAN,
            ),
        ]

    return pg_loss, metrics


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
                key="loss/cispo/clip_high_fraction",
                value=masked_mean(clipped_high.float(), mask),
                reduction=Reduce.MEAN,
            ),
            Metric(
                key="loss/cispo/clip_low_fraction",
                value=masked_mean(clipped_low.float(), mask),
                reduction=Reduce.MEAN,
            ),
        ]

    return pg_loss, metrics


# -----------------------------------------------------------------------------
# KL Divergence
# -----------------------------------------------------------------------------


def compute_kl(
    log_policy: torch.Tensor,
    log_ref: torch.Tensor,
    mask: torch.Tensor,
    kl_type: KLType = "k3",
) -> tuple[torch.Tensor, list[Metric]]:
    """Compute per-token KL divergence using Schulman's estimators.

    Reference: Schulman's blog post (http://joschu.net/blog/kl-approx.html).

    KL divergence measures how much the current policy differs from a reference
    policy. In RLHF, this prevents the model from straying too far from its
    pretrained behavior.

    Estimator properties (for KL[policy, ref]):
    - k1: Unbiased KL estimate, but E[grad k1] = 0 (useless for optimization).
    - k2: Biased KL estimate, but E[grad k2] = grad KL (unbiased gradient).
    - k3: Unbiased KL estimate with low variance. E[grad k3] = grad KL[ref, policy].

    k3 is preferred for monitoring KL value. k2 is preferred when using KL as a
    regularizer (gradient flows correctly). k1 is rarely used in practice.

    Args:
        log_policy (torch.Tensor): Log probs from current policy (B, S).
        log_ref (torch.Tensor): Log probs from reference policy (B, S).
        mask (torch.Tensor): Valid token mask (B, S).
        kl_type (KLType): KL estimator type: "k1", "k2", or "k3" (default: "k3").

    Returns:
        tuple[torch.Tensor, list[Metric]]: Per-token KL (B, S) and loss/kl/mean metric.
    """
    log_ratio = log_policy - log_ref.detach()  # log(π_θ / π_ref)

    if kl_type == "k1":
        kl = log_ratio
    elif kl_type == "k2":
        kl = 0.5 * log_ratio.square()
    elif kl_type == "k3":
        log_ratio_clamped = torch.clamp(-log_ratio, min=-20.0, max=20.0)
        ratio = torch.exp(log_ratio_clamped)  # π_ref / π_θ
        kl = ratio - log_ratio_clamped - 1
        kl = torch.clamp(kl, min=0.0, max=10.0)
    else:
        raise ValueError(f"Unknown kl_type: {kl_type}")

    with torch.no_grad():
        metrics = [
            Metric(
                key="loss/kl/mean",
                value=masked_mean(kl, mask),
                reduction=Reduce.MEAN,
            ),
        ]

    return kl, metrics


# -----------------------------------------------------------------------------
# Aggregation
# -----------------------------------------------------------------------------


def aggregate(
    per_token_loss: torch.Tensor,
    mask: torch.Tensor,
    agg_type: AggType = "token_mean",
    loss_scale: torch.Tensor | None = None,
) -> tuple[torch.Tensor, list[Metric]]:
    """Aggregate per-token loss to scalar.

    Different aggregation strategies have different bias properties that affect
    training dynamics:

    token_mean: sum(loss*mask) / loss_scale
        Where loss_scale defaults to sum(mask) if None is given.
        For distributed training, pass loss_scale = global_tokens / grad_avg_group_size
        for proper normalization.

    fixed_horizon: sum(loss*mask) / (B * S)
        Constant denominator (total elements) removes length bias.
        Each token contributes equally regardless of sequence length.

    sequence_mean: sum(loss*mask) / sum(mask, dim=-1) / B
        Mean per sequence then mean across batch.
        NOTE: This introduces a length bias, as discussed in DR-GRPO paper.

    Args:
        per_token_loss (torch.Tensor): Per-token loss (B, S).
        mask (torch.Tensor): Valid token mask (B, S).
        agg_type (AggType): Aggregation strategy.
        loss_scale (torch.Tensor | None): For token_mean only. If provided, use as divisor
            instead of mask.sum().

    Returns:
        tuple[torch.Tensor, list[Metric]]: Aggregated loss.
    """
    if agg_type == "token_mean":
        loss = masked_mean(per_token_loss, mask, loss_scale)

    elif agg_type == "fixed_horizon":
        # divide by (B * S)
        loss = (per_token_loss * mask).sum() / mask.numel()

    elif agg_type == "sequence_mean":
        seq_lengths = mask.sum(dim=-1).clamp(min=1.0)
        seq_means = (per_token_loss * mask).sum(dim=-1) / seq_lengths
        loss = seq_means.mean()

    else:
        raise ValueError(f"Unknown agg_type: {agg_type}")

    with torch.no_grad():
        metrics = [
            Metric(
                key="loss/aggregate/active_fraction",
                value=mask.mean(),
                reduction=Reduce.MEAN,
            ),
        ]

    return loss, metrics


# =============================================================================
# LOSSES
# =============================================================================


class GRPOLoss(PolicyGradientLoss, BaseLossConfig):
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

    Args:
        clip_low (float): Lower clip bound (default 0.2).
        clip_high (float): Upper clip bound (default 0.2).
        beta (float): KL penalty coefficient (default 0.1).
        agg_type (AggType): Aggregation method (default "fixed_horizon").
    """

    clip_low: Annotated[float, Field(ge=0, le=1)] = 0.2
    clip_high: Annotated[float, Field(ge=0, le=1)] = 0.2
    beta: Annotated[float, Field(ge=0)] = 0.1
    agg_type: AggType = "fixed_horizon"

    def __call__(
        self,
        logits: torch.Tensor,  # (B, S, V)
        target_ids: torch.Tensor,  # (B, S)
        advantages: torch.Tensor,  # (B, S)
        old_logprobs: torch.Tensor,  # (B, S)
        loss_mask: torch.Tensor,  # (B, S)
        ref_logprobs: torch.Tensor | None = None,  # (B, S) or None
        loss_scale: torch.Tensor | None = None,
        *args,
        **kwargs,
    ) -> LossOutput:
        logprobs, lp_m = compute_logprobs(logits, target_ids)
        entropy, ent_m = compute_entropy(logits, loss_mask)  # logging only
        ratio, log_ratio, ratio_m = compute_ratio(
            logprobs, old_logprobs, loss_mask, ratio_type="token"
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


class DAPOLoss(PolicyGradientLoss, BaseLossConfig):
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
        old_logprobs: torch.Tensor,  # (B, S)
        loss_mask: torch.Tensor,  # (B, S)
        loss_scale: torch.Tensor | None = None,
        *args,
        **kwargs,
    ) -> LossOutput:
        logprobs, lp_m = compute_logprobs(logits, target_ids)
        entropy, ent_m = compute_entropy(logits, loss_mask)
        ratio, log_ratio, ratio_m = compute_ratio(
            logprobs, old_logprobs, loss_mask, ratio_type="token"
        )
        pg_loss, clip_m = pg_ppo_clip(
            ratio, advantages, loss_mask, self.clip_low, self.clip_high
        )
        pg_loss, dual_m = pg_dual_clip(pg_loss, advantages, loss_mask, self.dual_clip_c)
        loss, agg_m = aggregate(pg_loss, loss_mask, self.agg_type, loss_scale)

        return LossOutput(loss, lp_m + ent_m + ratio_m + clip_m + dual_m + agg_m)


class GSPOLoss(PolicyGradientLoss, BaseLossConfig):
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
        2. Upper-only clipping: Only clips the upper bound (ratio <= 1+ε). The lower
           bound is effectively disabled (clip_low=1.0 → min=0).

    Args:
        clip_low (float): Lower clip bound offset (default 1.0, effectively
            no lower clipping).
        clip_high (float): Upper clip bound offset (default 0.2).
        agg_type (AggType): Aggregation method (default "sequence_mean").
    """

    clip_low: Annotated[float, Field(ge=0, le=1)] = 1.0
    clip_high: Annotated[float, Field(ge=0, le=1)] = 0.2
    agg_type: AggType = "sequence_mean"

    def __call__(
        self,
        logits: torch.Tensor,  # (B, S, V)
        target_ids: torch.Tensor,  # (B, S)
        advantages: torch.Tensor,  # (B, S)
        old_logprobs: torch.Tensor,  # (B, S)
        loss_mask: torch.Tensor,  # (B, S)
        loss_scale: torch.Tensor | None = None,
        *args,
        **kwargs,
    ) -> LossOutput:
        logprobs, lp_m = compute_logprobs(logits, target_ids)
        entropy, ent_m = compute_entropy(logits, loss_mask)
        ratio, log_ratio, ratio_m = compute_ratio(
            logprobs, old_logprobs, loss_mask, ratio_type="sequence"
        )
        pg_loss, clip_m = pg_ppo_clip(
            ratio, advantages, loss_mask, self.clip_low, self.clip_high
        )

        loss, agg_m = aggregate(pg_loss, loss_mask, self.agg_type, loss_scale)

        return LossOutput(loss, lp_m + ent_m + ratio_m + clip_m + agg_m)


class CISPOLoss(PolicyGradientLoss, BaseLossConfig):
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
        old_logprobs: torch.Tensor,  # (B, S)
        loss_mask: torch.Tensor,  # (B, S)
        loss_scale: torch.Tensor | None = None,
        *args,
        **kwargs,
    ) -> LossOutput:
        logprobs, lp_m = compute_logprobs(logits, target_ids)
        entropy, ent_m = compute_entropy(logits, loss_mask)
        ratio, log_ratio, ratio_m = compute_ratio(
            logprobs, old_logprobs, loss_mask, ratio_type="token"
        )
        pg_loss, cispo_m = pg_cispo(
            ratio, logprobs, advantages, loss_mask, self.clip_low, self.clip_high
        )
        loss, agg_m = aggregate(pg_loss, loss_mask, self.agg_type, loss_scale)

        return LossOutput(loss, lp_m + ent_m + ratio_m + cispo_m + agg_m)


class SAPOLoss(PolicyGradientLoss, BaseLossConfig):
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
        old_logprobs: torch.Tensor,  # (B, S)
        loss_mask: torch.Tensor,  # (B, S)
        loss_scale: torch.Tensor | None = None,
        *args,
        **kwargs,
    ) -> LossOutput:
        logprobs, lp_m = compute_logprobs(logits, target_ids)
        entropy, ent_m = compute_entropy(logits, loss_mask)
        ratio, log_ratio, ratio_m = compute_ratio(
            logprobs, old_logprobs, loss_mask, ratio_type="token"
        )
        pg_loss, gate_m = pg_soft_gate(
            ratio, advantages, loss_mask, self.tau_pos, self.tau_neg
        )
        loss, agg_m = aggregate(pg_loss, loss_mask, self.agg_type, loss_scale)

        return LossOutput(loss, lp_m + ent_m + ratio_m + gate_m + agg_m)
