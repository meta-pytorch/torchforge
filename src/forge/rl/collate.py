# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any

import torch
from forge.rl.types import Group


def collate(
    batches: list[Group],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Collates a list of batches into a single batch of inputs and targets.
    Each batch is a list of episodes, and each episode is a dict of tensors.
    """
    inputs = []
    targets = []
    for batch in batches:
        request = [e.request_tensor for e in batch]
        request = torch.stack(request)  # [b x s]

        response = [e.response_tensor for e in batch]
        response = torch.stack(response)  # [b x s]

        ref_logprobs = [e.ref_logprobs for e in batch]
        ref_logprobs = torch.stack(ref_logprobs).squeeze()  # [b x s]

        advantages = [e.advantage for e in batch]
        advantages = torch.tensor(advantages).unsqueeze(-1)  # [b x 1]

        pad_id = batch[0].pad_id
        mask = response != pad_id

        input = {"tokens": torch.cat([request, response], dim=1)}
        target = {
            "response": response,
            "ref_logprobs": ref_logprobs,
            "advantages": advantages,
            "padding_mask": mask,
        }
        inputs.append(input)
        targets.append(target)
    return inputs, targets
