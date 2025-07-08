# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import Any, Callable

import torch


def collate_packed(
    batch: list[dict[str, Any]], mask_fn: Callable, device: str
) -> dict[str, Any]:
    """
    Generic collate function for packed samples from an IterablePackedDataset.

    - Stacks tensors from all samples in the batch, while keeping non-tensor values
    as lists.
    - Handles metrics by extending them into a single list.
    - Delegates attention mask creation to a provided `mask_fn` callable that expects
    `document_ids` and `device` parameters to generate masks on-the-fly for
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

    # Delegate mask creation to the provided specialized function
    # TODO: investigate the need for device here. Device is needed
    # because mask is created on-the-fly, but we cant pass it
    # to the model forward.
    collated["mask"] = mask_fn(collated["document_ids"], device=device)

    return collated
