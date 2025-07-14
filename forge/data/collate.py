# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import Any, Callable

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

CROSS_ENTROPY_IGNORE_IDX = -100


def padded_collate_sft(
    batch: list[dict[str, Any]],
    padding_idx: int = 0,
    ignore_idx: int = CROSS_ENTROPY_IGNORE_IDX,
    pad_to_multiple_of: int = 1,
) -> dict[str, torch.Tensor]:
    """Pad a batch of sequences to the longest sequence length in the batch, and
    convert integer lists to tensors.

    Args:
        batch (list[dict[str, Any]]): A list of dictionaries containing samples, including tokens and labels.
        padding_idx (int): Padding index for input ids. Defaults to 0.
        ignore_idx (int): Padding index for labels. Defaults to -100.
        pad_to_multiple_of (int): If > 1, pad the sequence to a multiple of this number.
            This is useful for proper sharding with e.g. SequenceParallel.

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

    return batch_dict


def collate_packed(
    batch: list[dict[str, Any]], mask_fn: Callable, device: str
) -> dict[str, Any]:
    """
    Generic collate function for packed samples from an IterablePackedDataset.

    - Stacks tensors from all samples in the batch. Tensors are expected to be packed,
    i.e. have the same shape.
    - Non tensors are appended to a list., e.g. [a,b] + [c] = [[a,b],[c]]
    - Metrics are extended to a single list, e.g. [a,b] + [c] = [a,b,c].
    - Delegates attention mask creation to a provided `mask_fn`to generate masks on-the-fly for
    packed sequences. For an example, check 'create_block_mask' in 'forge.datasets._packed.py'.

    Args:
        batch (list[dict[str, Any]]): A list of dictionaries containing samples.
        mask_fn (Callable): A function that generates attention masks for packed sequences.
        device (str): The device to use for the tensors.

    Returns:
        dict[str, Any]: A dictionary containing the collated samples.

    Raises:
        ValueError: If all samples do not have the same keys.
    """
    if not batch:
        return {}

    # Verify all samples have the same keys
    first_sample_keys = batch[0].keys()
    for sample in batch:
        if sample.keys() != first_sample_keys:
            raise ValueError(
                f"All samples must have the same keys. Expected {first_sample_keys}, got {sample.keys()}"
            )

    keys_to_stack = first_sample_keys
    collated = {}

    for key in keys_to_stack:
        if isinstance(batch[0][key], torch.Tensor):
            collated[key] = torch.stack([sample[key] for sample in batch], dim=0)
        elif key == "metrics":
            collated[key] = []
            for sample in batch:
                collated[key].extend(sample[key])
        else:
            collated[key] = [sample[key] for sample in batch]

    # TODO: investigate the need for device here. Device is needed
    # because mask is created on-the-fly, during forward, given how flex_attention
    # works.
    collated["mask"] = mask_fn(collated["document_ids"], device=device)

    return collated
