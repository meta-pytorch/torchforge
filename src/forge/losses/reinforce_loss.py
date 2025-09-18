# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from trl.trainer import utils as trl_utils


class RinforceLoss(nn.Module):
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
        self, trainer_logits, target_ids, target_mask, target_weights, target_log_probs
    ):
        trainer_log_probs = trl_utils.selective_log_softmax(trainer_logits, target_ids)
        target_mask = target_mask.detach()
        target_weights = target_weights
        target_mask_sum = target_mask.sum()
        target_mask_sum = torch.maximum(
            target_mask_sum, torch.ones_like(target_mask_sum)
        )
        sampler_log_probs = target_log_probs

        use_importance_sampling_ratio = True  # TODO: needs to come from config
        if use_importance_sampling_ratio:
            logp_diff = trainer_log_probs - sampler_log_probs.detach()
            importance_weights = torch.exp(logp_diff).detach()
            target_weights *= importance_weights

        numerator = (-trainer_log_probs * target_weights * target_mask).sum()

        denominator = target_mask_sum
        return numerator / denominator
