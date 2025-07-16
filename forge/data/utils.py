# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Literal

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from forge.data.common import CROSS_ENTROPY_IGNORE_IDX

Role = Literal[
    "system",  # Origin is system prompt
    "user",  # Origin is user
    "assistant",  # Origin is the model output
    "ipython",  # Origin is return from a tool call
    "tool",  # Origin is return from a tool call
]


def padded_collate_sft(
    batch: list[dict[str, Any]],
    padding_idx: int = 0,
    ignore_idx: int = CROSS_ENTROPY_IGNORE_IDX,
    pad_to_multiple_of: int = 1,
    stack_on_new_dim: bool = False,
) -> dict[str, torch.Tensor]:
    """Pad a batch of sequences to the longest sequence length in the batch, and
    convert integer lists to tensors.

    Args:
        batch (list[dict[str, Any]]): A list of dictionaries containing samples, including tokens and labels.
        padding_idx (int): Padding index for input ids. Defaults to 0.
        ignore_idx (int): Padding index for labels. Defaults to -100.
        pad_to_multiple_of (int): If > 1, pad the sequence to a multiple of this number.
            This is useful for proper sharding with e.g. SequenceParallel.
        stack_on_new_dim (bool): If True, stack any encoder tensors on a new dimension. Default is False

    Returns:
        dict[str, torch.Tensor]: Collated input and label tensors.

    Example:
        >>> token_pairs = [
        >>>    {"tokens": [1, 2, 3], "labels": [4, 5, 6]},
        >>>    {"tokens": [7,], "labels": [10,]},
        >>> ]
        >>> collated = padded_collate(
        >>>    batch=token_pairs,
        >>>    padding_idx=padding_idx,
        >>>    ignore_idx=ignore_idx,
        >>> )
        >>> collated["tokens"]
        >>> tensor([[1, 2, 3], [7, 0, 0]])
        >>> collated["labels"]
        >>> tensor([[4, 5, 6], [10, -100, -100]])
    """
    input_ids = pad_sequence(
        [torch.tensor(x["tokens"]) for x in batch],
        batch_first=True,
        padding_value=padding_idx,
    )
    labels = pad_sequence(
        [torch.tensor(x["labels"]) for x in batch],
        batch_first=True,
        padding_value=ignore_idx,
    )

    input_ids_seq_len = input_ids.shape[-1]
    labels_seq_len = labels.shape[-1]

    # Hack to pad correctly and not use max_seq_len, which is costly
    if input_ids_seq_len > labels_seq_len:
        labels = F.pad(
            labels, (0, input_ids_seq_len - labels_seq_len), value=ignore_idx
        )
    elif labels_seq_len > input_ids_seq_len:
        input_ids = F.pad(
            input_ids,
            (0, labels_seq_len - input_ids_seq_len),
            value=padding_idx,
        )

    # Pad to multiple of N
    if pad_to_multiple_of > 1:
        input_ids = F.pad(
            input_ids,
            (0, pad_to_multiple_of - (input_ids_seq_len % pad_to_multiple_of)),
            value=padding_idx,
        )
        labels = F.pad(
            labels,
            (0, pad_to_multiple_of - (labels_seq_len % pad_to_multiple_of)),
            value=ignore_idx,
        )
    batch_dict = {"tokens": input_ids.long(), "labels": labels.long()}
    if "encoder_input" in batch[0]:
        x = [x["encoder_input"] for x in batch]
        batched_encodings = _stack_encoder_input(x, new_dim=stack_on_new_dim)
        if batched_encodings != {}:
            batch_dict["encoder_input"] = batched_encodings
    return batch_dict
