# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn


class ReinforceLoss(nn.Module):
    """Reinforce loss function with optional importance ratio clipping.

    Reinforce with importance ratio is NOT GRPO. GRPO uses ratio clipping, where
    tokens outside trust region don't have gradients. Reinforce with importance
    ratio clips a detached importance ratio, where tokens outside trust region
    still have gradients.

    This difference is importance when very bad things happens, e.g. SDC or
    expert selection mismatch between sampling and policy update due to
    numerical noise. GRPO is more resilient in this case.
    """

    def __init__(
        self, prob_ratio_min: float | None = None, prob_ratio_max: float | None = None
    ):
        super().__init__()
        self.prob_ratio_min = prob_ratio_min
        self.prob_ratio_max = prob_ratio_max

    def forward(self, logprobs, sampling_logprobs, advantages, padding_mask):
        prob_ratio = torch.exp(logprobs - sampling_logprobs)
        prob_ratio = torch.clamp(
            prob_ratio, min=self.prob_ratio_min, max=self.prob_ratio_max
        )
        advantages = advantages * prob_ratio

        per_token_loss = -logprobs * advantages
        sequence_length = padding_mask.sum(dim=1).clamp(min=1.0)
        per_sequence_loss = (per_token_loss * padding_mask).sum(dim=1) / sequence_length

        loss = per_sequence_loss.mean()

        return loss
