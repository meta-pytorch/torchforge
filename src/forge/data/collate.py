# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Callable

import torch
import torch.nn.functional as F

from forge.data.utils import CROSS_ENTROPY_IGNORE_IDX


def collate_padded(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Collate function that pads sequences to the longest sample in the batch.

    Pads 'tokens' with 0 and 'labels' with CROSS_ENTROPY_IGNORE_IDX (-100).
    Non-tensor fields (like metrics) are collected into lists and flattened
    if all items are lists.

    Args:
        batch: List of samples, each containing 'tokens' and 'labels' tensors

    Returns:
        Batched dict with padded tensors
    """
    if not batch:
        return {}

    # Find max length in batch
    max_len = max(sample["tokens"].size(0) for sample in batch)

    # Initialize lists for batched tensors
    tokens_list = []
    labels_list = []

    # Pad each sample to max_len
    for sample in batch:
        seq_len = sample["tokens"].size(0)
        pad_len = max_len - seq_len

        # Pad tokens with 0
        padded_tokens = F.pad(sample["tokens"], (0, pad_len), value=0)
        tokens_list.append(padded_tokens)

        # Pad labels with CROSS_ENTROPY_IGNORE_IDX (-100)
        padded_labels = F.pad(
            sample["labels"], (0, pad_len), value=CROSS_ENTROPY_IGNORE_IDX
        )
        labels_list.append(padded_labels)

    # Stack into batch
    result = {
        "tokens": torch.stack(tokens_list),
        "labels": torch.stack(labels_list),
    }

    # Collect non-tensor fields (like metrics)
    for key in batch[0].keys():
        if key not in ["tokens", "labels"]:
            result[key] = [sample[key] for sample in batch]
            # Flatten if all are lists
            if all(isinstance(item, list) for item in result[key]):
                result[key] = [item for sublist in result[key] for item in sublist]

    return result


def collate_packed(
    batch: list[dict[str, Any]], mask_fn: Callable, device: str
) -> dict[str, Any]:
    """
    Generic collate function for packed samples from an IterablePackedDataset.
    Stacks tensors from all samples in the batch, while keeping non-tensor values
    as lists. Handles metrics by extending them into a single list. Delegates
    attention mask creation to a provided `mask_fn` callable that expects
    `document_ids` and `device` parameters to generate masks on-the-fly for
    packed sequences.
    Args:
        batch (list[dict[str, Any]]): A list of dictionaries containing samples.
        mask_fn (callable): A function that generates attention masks for packed sequences.
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

    # Delegate mask creation to the provided specialized function
    # TODO: investigate the need for device here. Currently we hardcode it in utilities to cuda.
    # shouldnt we just send to device later?
    collated["mask"] = mask_fn(collated["document_ids"], device=device)

    return collated
