# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from forge.rl.types import Group
from forge.types import TrainBatch


def collate(batches: list[Group]) -> list[TrainBatch]:
    """
    Collates a list of batches into TrainBatch objects.
    Each batch is a list of episodes, and each episode is a dict of tensors.
    """
    result = []
    for batch in batches:
        request = [e.request_tensor for e in batch]
        request = torch.stack(request)  # [b x s]

        response = [e.response_tensor for e in batch]
        response = torch.stack(response)  # [b x s]

        input_ids = torch.cat([request, response], dim=1)
        seq_len = input_ids.shape[1]

        # ref_logprobs is optional - only stack if all episodes have it
        ref_logprobs = None
        if all(e.ref_logprobs is not None for e in batch):
            ref_logprobs = torch.stack([e.ref_logprobs for e in batch])

        advantages = [e.advantage for e in batch]
        advantages = torch.tensor(advantages).unsqueeze(-1)  # [b x 1]
        advantages = advantages.expand(-1, seq_len)  # [b x s]

        generator_logprobs = torch.stack([e.generator_logprobs for e in batch])
        loss_mask = torch.stack([e.loss_mask for e in batch])

        loss_inputs = {
            "generator_logprobs": generator_logprobs,
            "loss_mask": loss_mask,
            "advantages": advantages,
        }
        if ref_logprobs is not None:
            loss_inputs["ref_logprobs"] = ref_logprobs

        result.append(
            TrainBatch(
                model_inputs={"tokens": input_ids},
                loss_inputs=loss_inputs,
            )
        )
    return result
