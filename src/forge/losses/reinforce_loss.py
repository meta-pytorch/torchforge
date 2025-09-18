# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from trl.trainer import utils as trl_utils


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

    def selective_log_softmax(self, logits, index) -> torch.Tensor:
        """
        A memory-efficient implementation of the common `log_softmax -> gather` operation.

        This function is equivalent to the following naive implementation:
        ```python
        logps = torch.gather(logits.log_softmax(-1), dim=-1, index=index.unsqueeze(-1)).squeeze(-1)
        ```

        Args:
            logits (`torch.Tensor`):
                Logits tensor of shape `(..., num_classes)`.
            index (`torch.Tensor`):
                Index tensor of shape `(...)`, specifying the positions to gather from the log-softmax output.

        Returns:
            `torch.Tensor`:
                Gathered log probabilities with the same shape as `index`.
        """
        if logits.dtype in [torch.float32, torch.float64]:
            selected_logits = torch.gather(
                logits, dim=-1, index=index.unsqueeze(-1)
            ).squeeze(-1)
            # loop to reduce peak mem consumption
            logsumexp_values = torch.stack(
                [torch.logsumexp(lg, dim=-1) for lg in logits]
            )
            per_token_logps = (
                selected_logits - logsumexp_values
            )  # log_softmax(x_i) = x_i - logsumexp(x)
        else:
            # logsumexp approach is unstable with bfloat16, fall back to slightly less efficient approach
            per_token_logps = []
            for row_logits, row_labels in zip(
                logits, index
            ):  # loop to reduce peak mem consumption
                row_logps = F.log_softmax(row_logits, dim=-1)
                row_per_token_logps = row_logps.gather(
                    dim=-1, index=row_labels.unsqueeze(-1)
                ).squeeze(-1)
                per_token_logps.append(row_per_token_logps)
            per_token_logps = torch.stack(per_token_logps)
        return per_token_logps
