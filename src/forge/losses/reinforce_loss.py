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

    def __init__(self):
        super().__init__()

    def forward(
        self,
        logprobs,
        target_ids,
        padding_mask,
        advantages,
        sampling_logprobs,
    ):
        target_mask_sum = padding_mask.sum()
        target_mask_sum = torch.maximum(
            target_mask_sum, torch.ones_like(target_mask_sum)
        )

        # Importance sampling ratio
        logp_diff = logprobs - sampling_logprobs.detach()
        prob_ratio = torch.exp(logp_diff).detach()
        prob_ratio = torch.clamp(prob_ratio, min=0.1, max=10.0)
        advantages = advantages * prob_ratio

        numerator = (-logprobs * advantages * padding_mask).sum()

        denominator = target_mask_sum
        return numerator / denominator
