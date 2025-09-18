# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn


class SimpleGRPOLoss(nn.Module):
    """
    Group Relative Policy Optimization (GRPO) Loss Function.

    GRPO is a policy gradient method that optimizes a policy relative to a reference policy
    using advantages computed from group-based rewards. This implementation computes the
    standard GRPO objective with KL divergence regularization.

    Mathematical Formulation:
        L = -E[log(π_θ/π_ref) * A] + β * KL(π_θ || π_ref)

    Where:
        - π_θ: Current policy being trained
        - π_ref: Reference policy (usually frozen copy of initial policy)
        - A: Advantage values (typically rewards or reward-to-go)
        - β: KL regularization coefficient

    The loss encourages:
        - Increasing probability of actions with positive advantages
        - Decreasing probability of actions with negative advantages
        - Staying close to reference policy (KL penalty prevents drift)
    """

    def __init__(self, beta: float = 0.1):
        """
        Initialize GRPO loss function.

        Args:
            beta (float): KL divergence penalty coefficient. Higher values prevent
                         the policy from deviating too far from the reference policy.
                         Typical range: 0.01 - 0.5
        """
        super().__init__()
        self.beta = beta

    def forward(self, logprobs, ref_logprobs, advantages, padding_mask):
        """
        Compute GRPO loss for a batch of token sequences.

        Args:
            logprobs (torch.Tensor): Log probabilities from current policy.
                                   Shape: [batch_size, seq_len]
            ref_logprobs (torch.Tensor): Log probabilities from reference policy.
                                        Shape: [batch_size, seq_len]
            advantages (torch.Tensor): Advantage values for each token position.
                                     Shape: [batch_size, seq_len]
            padding_mask (torch.Tensor): Binary mask indicating valid tokens (1) vs
                                        padding (0). Shape: [batch_size, seq_len]

        Returns:
            torch.Tensor: Scalar loss value averaged over all valid tokens.

        Note:
            - Loss is only computed on non-padded tokens (where padding_mask == 1)
            - Typically only response tokens should have non-zero mask values
            - All input tensors should be on the same device
        """
        # Compute log ratio: log(π_θ/π_ref) = log(π_θ) - log(π_ref)
        log_ratio = logprobs - ref_logprobs

        # Policy gradient loss: -log_ratio * advantages
        per_token_policy_loss = log_ratio * advantages

        # KL penalty: KL(π_θ || π_ref) ≈ π_θ * (log(π_θ) - log(π_ref))
        # For log probabilities, this becomes: exp(logprobs) * (logprobs - ref_logprobs)
        kl_penalty = torch.exp(logprobs) * (logprobs - ref_logprobs)

        # Total per-token loss
        per_token_loss = -(per_token_policy_loss - self.beta * kl_penalty)

        # Average over valid tokens
        loss = (
            (per_token_loss * padding_mask).sum(dim=1)
            / padding_mask.sum(dim=1).clamp(min=1.0)
        ).mean()
        return loss
