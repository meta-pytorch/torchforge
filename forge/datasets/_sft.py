# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Callable, Optional

from forge.data.metrics import DefaultTrainingMetricTransform

from forge.data.transforms import SFTOutputTransform, Transform
from forge.datasets._hf import HfIterableDataset


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
