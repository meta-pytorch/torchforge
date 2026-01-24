# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F
from forge.observability.metrics import Metric, Reduce
from forge.rl.loss.types import AggType, KLType, RatioType

CROSS_ENTROPY_IGNORE_IDX = -100


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

    # we return empty metrics to preserve the pattern used in all primitives
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


def compute_ratio(
    logprobs: torch.Tensor,
    generator_logprobs: torch.Tensor,
    mask: torch.Tensor,
    ratio_type: RatioType = "token",
) -> tuple[torch.Tensor, torch.Tensor, list[Metric]]:
    """Compute importance sampling ratio for off-policy correction.

    The ratio r = π_θ/π_old measures how much the current policy differs from
    the policy that generated the samples. This enables reusing samples from an
    old policy while adjusting for distribution shift.

    Formula:
        token:    r_t = exp(logprobs_t - generator_logprobs_t)
        sequence: R_seq = exp(mean_t[logprobs - generator_logprobs])

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
        generator_logprobs (torch.Tensor): Log probs from sampling policy (B, S).
        mask (torch.Tensor): Valid token mask (B, S).
        ratio_type (RatioType): "token" for per-token ratio, "sequence" for sequence-level.
            default: "token".

    Returns:
        tuple[torch.Tensor, torch.Tensor, list[Metric]]: (ratio, log_ratio, metrics). Both
            ratio and log_ratio are (B, S). For sequence type, values are broadcast from
            per-sequence computation.
    """
    if ratio_type == "token":
        log_ratio = logprobs - generator_logprobs.detach()
        ratio = torch.exp(log_ratio)

    elif ratio_type == "sequence":
        token_log_ratio = logprobs - generator_logprobs.detach()
        seq_lengths = mask.sum(dim=-1).clamp(min=1)
        seq_log_ratio = (token_log_ratio * mask).sum(dim=-1) / seq_lengths

        # Reparameterization: forward uses seq ratio, backward uses token grads
        log_ratio = logprobs - logprobs.detach() + seq_log_ratio.detach().unsqueeze(-1)
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
                key="loss/kl_policy/mean",
                value=masked_mean(-log_ratio, mask),
                reduction=Reduce.MEAN,
            ),
        ]

    return ratio, log_ratio, metrics


def compute_kl(
    policy_logprobs: torch.Tensor,
    ref_logprobs: torch.Tensor,
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
        policy_logprobs (torch.Tensor): Log probs from current policy (B, S).
        ref_logprobs (torch.Tensor): Log probs from reference policy (B, S).
        mask (torch.Tensor): Valid token mask (B, S).
        kl_type (KLType): KL estimator type: "k1", "k2", or "k3" (default: "k3").

    Returns:
        tuple[torch.Tensor, list[Metric]]: Per-token KL (B, S) and loss/kl/mean metric.
    """
    log_ratio = policy_logprobs - ref_logprobs.detach()  # log(π_θ / π_ref)

    if kl_type == "k1":
        kl = log_ratio
    elif kl_type == "k2":
        kl = 0.5 * log_ratio.square()
    elif kl_type == "k3":
        neg_log_ratio = torch.clamp(-log_ratio, min=-10.0, max=10.0)
        ratio = torch.exp(neg_log_ratio)  # π_ref / π_θ
        kl = ratio - neg_log_ratio - 1
    else:
        raise ValueError(f"Unknown kl_type: {kl_type}")

    with torch.no_grad():
        metrics = [
            Metric(
                key="loss/kl_ref/mean",
                value=masked_mean(kl, mask),
                reduction=Reduce.MEAN,
            ),
        ]

    return kl, metrics


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
        # divide by (B * S), use max to avoid division by zero for empty inputs
        loss = (per_token_loss * mask).sum() / max(mask.numel(), 1)

    elif agg_type == "sequence_mean":
        seq_lengths = mask.sum(dim=-1).clamp(min=1.0)
        seq_means = (per_token_loss * mask).sum(dim=-1) / seq_lengths
        # Handle empty batch: mean of empty tensor returns nan, so use sum instead
        loss = seq_means.sum() / max(seq_means.numel(), 1)

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
        pos_adv = advantages > 0
        neg_adv = advantages < 0
        metrics = [
            Metric(
                key="loss/clip/clipped_ratio/mean",
                value=masked_mean(clipped_ratio, mask),
                reduction=Reduce.MEAN,
            ),
            Metric(
                key="loss/clip/high_fraction",
                value=masked_mean((clipped_high & pos_adv).float(), mask),
                reduction=Reduce.MEAN,
            ),
            Metric(
                key="loss/clip/low_fraction",
                value=masked_mean((clipped_low & neg_adv).float(), mask),
                reduction=Reduce.MEAN,
            ),
        ]

    return pg_loss, metrics
