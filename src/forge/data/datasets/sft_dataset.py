# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Callable, Optional

import torch

from forge.data import CROSS_ENTROPY_IGNORE_IDX
from forge.data.dataset_metrics import DefaultTrainingMetricTransform
from forge.data.utils import mask_messages, TuneMessage
from forge.interfaces import Transform

from .hf_dataset import HfIterableDataset


class AlpacaToMessages(Transform):
    """
    Message transform class for Alpaca-style datasets with "instruction", "input", and "output"
    (or equivalent fields specified in column_map) columns. User messages are formed from the
    instruction + input columns and assistant messages are formed from the output column. Prompt
    templating is conditional on the presence of the "input" column, and thus is handled directly
    in this transform class instead of a dedicated :class:`~torchtune.data.PromptTemplate` class
    due to this custom logic.

    Args:
        column_map (Optional[dict[str, str]]): a mapping to change the expected "instruction", "input",
            and "output" column names to the actual column names in the dataset. Default is None,
            keeping the default column names.
        masking_strategy (str): masking strategy to use for model training.
            Must be one of: `train_on_all`, `train_on_assistant`, `train_on_last`.
            Default is "train_on_all".

            - ``train_on_all``: both user and assistant messages are unmasked
            - ``train_on_assistant``: user messages are masked, only assistant messages are unmasked
            - ``train_on_last``: only the last assistant message is unmasked

    Raises:
        ValueError:
            If ``column_map`` is provided and ``instruction`` not in ``column_map``, or
                ``output`` not in ``column_map``
    """

    def __init__(
        self,
        column_map: Optional[dict[str, str]] = None,
        masking_strategy: str = "train_on_all",
    ):
        self.masking_strategy = masking_strategy
        if column_map:
            if "instruction" not in column_map:
                raise ValueError(
                    f"Expected a key of 'instruction' in column_map but found {column_map.keys()}."
                )
            # input is optional
            if "output" not in column_map:
                raise ValueError(
                    f"Expected a key of 'output' in column_map but found {column_map.keys()}."
                )
            self._column_map = column_map
        else:
            self._column_map = {
                "instruction": "instruction",
                "input": "input",
                "output": "output",
            }
        self.template = {
            "prompt_input": (
                "Below is an instruction that describes a task, paired with an input that provides further context. "
                "Write a response that appropriately completes the request.\n\n"
                "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
            ),
            "prompt_no_input": (
                "Below is an instruction that describes a task. "
                "Write a response that appropriately completes the request.\n\n"
                "### Instruction:\n{instruction}\n\n### Response:\n"
            ),
        }

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        key_input = self._column_map.get("input", "input")
        if key_input in sample and sample[key_input]:
            prompt = self.template["prompt_input"].format(
                instruction=sample[self._column_map["instruction"]],
                input=sample[key_input],
            )
        else:
            prompt = self.template["prompt_no_input"].format(
                instruction=sample[self._column_map["instruction"]]
            )

        messages = [
            TuneMessage(
                role="user",
                content=prompt,
                eot=True,
            ),
            TuneMessage(
                role="assistant",
                content=sample[self._column_map["output"]],
                eot=True,
            ),
        ]
        mask_messages(messages, self.masking_strategy)
        return {"messages": messages}


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
