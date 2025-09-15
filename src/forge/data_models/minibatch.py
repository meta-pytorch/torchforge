# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Sequence

import torch
from forge.data_models.experience import Experience


@dataclass
class Minibatch:
    """The minibatch that trainer will recieve."""

    # The input sequence token ids for the trainer forward pass.
    input_ids: torch.Tensor

    # The segment ids for the input sequence token ids. Same segment
    # ids respresent the same sequence.
    segment_ids: torch.Tensor

    # The targets required for loss computation, usually concatenated prompt and
    # sample token ids.
    target_ids: torch.Tensor

    # The mask for the target ids, 0 for prompt tokens, 1 for sample tokens.
    target_mask: torch.Tensor

    # The weight to apply to the loss of each target token. It's normally computed
    # from the advantage and the reward.
    target_weights: torch.Tensor

    # The log probabilities of the target tokens, for prompt part it's set to 0,
    # for generation part it's computed from the sampler.
    target_log_probs: torch.Tensor


def from_experiences(
    exps: Sequence[Experience], max_seq_len: int, pad_val: int = 0
) -> Minibatch:
    """
    Convert a list of experiences to a minibatch.
    """

    def pack_sequence(
        tensors: Sequence[torch.Tensor],
        pad_val: Any,
        dtype: torch.dtype,
        max_len: int,
    ) -> torch.Tensor:
        """Packs multiple tensors along the seq dim."""
        seq = torch.cat(tensors)
        pad_len = max_len - seq.size(0)
        if pad_len < 0:
            raise ValueError(
                f"Sequence lenth {seq.size(0)} exceeds the maximum length {max_len}"
            )
        return torch.nn.functional.pad(seq, (0, pad_len), value=pad_val)[None, ...].to(
            dtype
        )

    mini_batch = {}
    exp_list = defaultdict(list)
    for i, exp in enumerate(exps):
        input_ids = exp.ids[:-1]
        exp_list["input_ids"].append(input_ids)
        exp_list["target_ids"].append(exp.ids[1:])
        exp_list["segment_ids"].append(torch.ones_like(input_ids) * i)
        exp_list["target_mask"].append(exp.mask[1:])
        exp_list["target_weights"].append(exp.weights[1:])
        exp_list["target_log_probs"].append(exp.log_probs[1:])

    for k, v in exp_list.items():
        _dtype = torch.int64
        if k == "target_mask" or k == "target_weights" or k == "target_log_probs":
            _dtype = torch.float32

        mini_batch[k] = pack_sequence(v, pad_val, _dtype, max_seq_len)

    return Minibatch(**mini_batch)
