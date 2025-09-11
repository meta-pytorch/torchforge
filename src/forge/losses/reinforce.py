# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from forge.data_models.distributed_metric import Fraction, SumDistributedMetric
from forge.data_models.loss import LossInput, LossOutput
from trl.trainer import utils as trl_utils


class ReinforceLoss:
    """Reinforce loss function with optional importance ratio clipping.

    Reinforce with importance ratio is NOT GRPO. GRPO uses ratio clipping, where
    tokens outside trust region don't have gradients. Reinforce with importance
    ratio clips a detached importance ratio, where tokens outside trust region
    still have gradients.

    This difference is importance when very bad things happens, e.g. SDC or
    expert selection mismatch between sampling and policy update due to
    numerical noise. GRPO is more resilient in this case.
    """

    def loss(self, loss_input: LossInput) -> LossOutput:
        trainer_log_probs = trl_utils.selective_log_softmax(
            loss_input.trainer_logits, loss_input.minibatch.target_ids
        )
        target_mask = loss_input.minibatch.target_mask.detach()
        target_weights = loss_input.minibatch.target_weights
        target_mask_sum = loss_input.minibatch.target_mask.sum()
        target_mask_sum = torch.maximum(
            target_mask_sum, torch.ones_like(target_mask_sum)
        )
        sampler_log_probs = loss_input.minibatch.target_log_probs

        use_importance_sampling_ratio = True  # TODO: needs to come from config
        if use_importance_sampling_ratio:
            logp_diff = trainer_log_probs - sampler_log_probs.detach()
            importance_weights = torch.exp(logp_diff).detach()
            target_weights *= importance_weights

        numerator = SumDistributedMetric(
            (-trainer_log_probs * target_weights * target_mask).sum()
        )
        denominator = SumDistributedMetric(target_mask_sum)
        return LossOutput(loss=Fraction(numerator=numerator, denominator=denominator))
