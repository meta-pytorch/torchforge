# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Callable, Optional, TypeVar

import torch
from forge.data.common import CROSS_ENTROPY_IGNORE_IDX

from forge.data.hf import HfIterableDataset
from forge.data.metrics import DefaultTrainingMetricTransform

Transform = TypeVar("Transform")


class SFTOutputTransform:
    """Applied to each dataset sample to build the `"labels"` tensor for causal-LM SFT training.

    Expects sample to contain 1-D torch tensors
    "tokens": token IDs, dtype=torch.long
    "mask": bool/int where **True** marks positions to ignore

    If they are not tensors, they are converted to tensors.

    Produces ``"labels"`` of the same shape such that
        labels[t] =  tokens[t+1]                # shift left
        labels[t] =  IGNORE_IDX  if mask[t+1]   # respect mask
        labels[-1] = IGNORE_IDX                 # last token has no target

    All ops are vectorised; only one fresh tensor (`labels`) is allocated.
    """

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:

        # Sanity checks
        if not isinstance(sample["tokens"], torch.Tensor):
            sample["tokens"] = torch.tensor(sample["tokens"])
        if not isinstance(sample["mask"], torch.Tensor):
            sample["mask"] = torch.tensor(sample["mask"])

        tokens = sample["tokens"]
        mask = sample["mask"]

        if tokens.ndim != 1 or mask.ndim != 1:
            raise ValueError("Both 'tokens' and 'mask' must be 1-D tensors.")

        # build labels
        # pre-fill with IGNORE so we don’t need extra assignments later
        labels = tokens.new_full(tokens.shape, CROSS_ENTROPY_IGNORE_IDX)

        # left-shift via cheap views (no copy)
        labels[:-1].copy_(tokens[1:])

        # apply mask in-place (single fused kernel on GPU/CPU)
        labels[:-1].masked_fill_(mask[1:].bool(), CROSS_ENTROPY_IGNORE_IDX)

        out = dict(sample)
        out["labels"] = labels
        return out


def sft_iterable_dataset(
    model_transform: Transform,
    *,
    weight: int = 1,
    message_transform: Transform,
    shuffle_buffer_size: Optional[int] = 1000,
    seed: int = 42,
    num_shards_per_rank: int = 64,
    dataset_name: Optional[str] = None,
    filter_fn: Optional[Callable] = None,
    filter_kwargs: Optional[dict[str, Any]] = None,
    **load_dataset_kwargs: dict[str, Any],
) -> HfIterableDataset:
    """
    Creates an SFT-ready iterable dataset with appropriate output transform.

    Args:
        model_transform (Transform): Usually the tokenizer
        weight (int): Weight of the dataset. Used for sampling when interleaving datasets.
        message_transform (Transform): Transform to convert raw data to messages
        shuffle_buffer_size (Optional[int]): Buffer size for shuffling
        seed (int): Random seed for shuffling
        num_shards_per_rank (int): Target shards per worker
        dataset_name (Optional[str]): Name for metrics namespacing
        filter_fn (Optional[Callable]): Filter function
        filter_kwargs (Optional[dict[str, Any]]): Filter function kwargs
        **load_dataset_kwargs (dict[str, Any]): Args passed to load_dataset

    Returns:
        HfIterableDataset: Configured for SFT training

    Example:
        >>> from forge.data import AlpacaToMessages
        >>> message_transform = AlpacaToMessages(train_on_input=False)
        >>> ds = sft_iterable_dataset(
        ...     message_transform=message_transform,
        ...     model_transform=tokenizer,
        ...     path="tatsu-lab/alpaca"
        ... )
    """

    output_transform = SFTOutputTransform()

    return HfIterableDataset(
        message_transform=message_transform,
        model_transform=model_transform,
        output_transform=output_transform,
        metric_transform=DefaultTrainingMetricTransform(),
        shuffle_buffer_size=shuffle_buffer_size,
        weight=weight,
        seed=seed,
        num_shards_per_rank=num_shards_per_rank,
        dataset_name=dataset_name,
        filter_fn=filter_fn,
        filter_kwargs=filter_kwargs,
        **load_dataset_kwargs,
    )
