# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Callable, Optional

from forge.data.transforms.tokenizers import ModelTokenizer
from forge.data.transforms.transforms import AlpacaToMessages
from forge.datasets.hf import HfIterableDataset
from forge.datasets.sft import sft_iterable_dataset


def alpaca_iterable_dataset(
    model_transform: ModelTokenizer,
    *,
    path: str = "tatsu-lab/alpaca",
    column_map: Optional[dict[str, str]] = None,
    masking_strategy: str = "train_on_all",
    shuffle_buffer_size: Optional[int] = 1000,
    seed: int = 42,
    dataset_name: Optional[str] = None,
    filter_fn: Optional[Callable] = None,
    split: str = "train",
    **load_dataset_kwargs: dict[str, Any],
) -> HfIterableDataset:
    """
    Support for iterable version of Alpaca-style datasets.
    This returns an infinite iterable dataset that supports checkpointing
    and metrics tracking, designed for step-based training.
    Args:
        model_transform (ModelTokenizer): Model tokenizer used to tokenize the messages.
        path (str): path to dataset repository on Hugging Face. Default is ``tatsu-lab/alpaca``.
        column_map (Optional[dict[str, str]]): a mapping from the expected columns in the message transform
            :class:`~torchtune.data.AlpacaToMessages` to the new column names in the dataset. Keys should be
            "instruction", "input", and "output" and values should be the actual column names.
        masking_strategy (str): masking strategy to use for model training.
            Must be one of: `train_on_all`, `train_on_assistant`, `train_on_last`.
            Default is "train_on_all".

            - ``train_on_all``: both user and assistant messages are unmasked
            - ``train_on_assistant``: user messages are masked, only assistant messages are unmasked
            - ``train_on_last``: only the last assistant message is unmasked
        train_on_input (bool): Whether the model is trained on the prompt or not. Default is True.
        shuffle_buffer_size (Optional[int]): Size of the shuffle buffer. If None or 0, no shuffling is done.
        seed (int): Seed for shuffling.
        dataset_name (Optional[str]): Name of the dataset for metrics tracking. If None, auto-generated.
        filter_fn (Optional[Callable]): callable used to filter the dataset prior to any pre-processing.
        split (str): ``split`` argument for ``datasets.load_dataset``. Default is "train".
        **load_dataset_kwargs (dict[str, Any]): additional keyword arguments to pass to ``load_dataset``.
    Returns:
        HfIterableDataset: iterable dataset configured with source data and transforms
    Example:
        >>> from torchdata.stateful_dataloader import StatefulDataLoader
        >>> alpaca_ds = alpaca_iterable_dataset(tokenizer=tokenizer)
        >>> dataloader = StatefulDataLoader(alpaca_ds, batch_size=8)
        >>> for batch in dataloader:
        >>>     print(f"Batch size: {len(batch)}")
        >>> Batch size: 8
    """
    message_transform = AlpacaToMessages(
        column_map=column_map, masking_strategy=masking_strategy
    )

    return sft_iterable_dataset(
        message_transform=message_transform,
        model_transform=model_transform,
        shuffle_buffer_size=shuffle_buffer_size,
        seed=seed,
        dataset_name=dataset_name,
        filter_fn=filter_fn,
        split=split,
        path=path,
        **load_dataset_kwargs,
    )
