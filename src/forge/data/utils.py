# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from enum import Enum
from typing import Any, Iterator, Literal, Union

import torch
import torch.distributed as dist

from torch.nn.attention.flex_attention import BlockMask

logger = logging.getLogger(__name__)

CROSS_ENTROPY_IGNORE_IDX = -100

Role = Literal[
    "system",  # Origin is system prompt
    "user",  # Origin is user
    "assistant",  # Origin is the model output
    "ipython",  # Origin is return from a tool call
    "tool",  # Origin is return from a tool call
]


class MaskingStrategy(Enum):
    TRAIN_ON_ALL = "train_on_all"
    TRAIN_ON_ASSISTANT = "train_on_assistant"
    TRAIN_ON_LAST = "train_on_last"


class TuneMessage:
    """
    This class represents individual messages in a fine-tuning dataset. It supports
    text-only content, text with interleaved images, and tool calls. The
    :class:`~forge.interfaces.ModelTokenizer` will tokenize
    the content of the message using ``tokenize_messages`` and attach the appropriate
    special tokens based on the flags set in this class.

    Args:
        role (Role): role of the message writer. Can be "system" for system prompts,
            "user" for human prompts, "assistant" for model responses, or "ipython"
            for tool call returns.
        content (Union[str, list[dict[str, Any]]]): content of the message. If it is text only content,
            you can pass in a string. If it is multimodal content, pass in a list of dictionaries formatted
            as follows::

                [
                    {"type": "image", "content": torch.Tensor},
                    {"type": "text", "content": "What is in this image?"},
                ]

        masked (bool): whether the message is masked in the sample. If True, do not use
            in loss calculation. Default: False
        ipython (bool): whether the message is a tool call. Default: False
        eot (bool): whether the message corresponds to the end of a turn, where control is handed over
            to the assistant from the user or the user from the assistant. Default: True. Should be true
            in most cases except for:

            - For multiple consecutive assistant messages (i.e., tool calls
              by assistant), only the last assistant message will have ``eot=True``
            - All ipython messages (tool call returns) should set ``eot=False``.

    Note:
        TuneMessage class expects any image content to be a ``torch.Tensor``.
    """

    def __init__(
        self,
        role: Role,
        content: Union[str, list[dict[str, Any]]],
        masked: bool = False,
        ipython: bool = False,
        eot: bool = True,
    ):
        self.role = role
        self.content = self._convert_to_list_of_dict(content)
        self.masked = masked
        self.ipython = ipython
        self.eot = eot

    def _convert_to_list_of_dict(self, content) -> list[dict[str, Any]]:
        """User is currently allowed to pass in a string for text-only content.
        This ensures that the content is formatted as a list of dictionaries."""
        if isinstance(content, str):
            return [{"type": "text", "content": content}]

        assert isinstance(
            content, list
        ), f"content must be of type list[dict[str, Any]], got {content}"

        return content

    @classmethod
    def from_dict(cls, d: dict) -> "TuneMessage":
        """
        Construct a TuneMessage from a dictionary.

        Args:
            d (dict): dictionary containing the fields of the TuneMessage.

        Returns:
            TuneMessage: constructed TuneMessage.
        """
        return cls(
            role=d["role"],
            content=d["content"],
            masked=d.get("masked", False),
            ipython=d.get("ipython", False),
            eot=d.get("eot", True),
        )

    def __repr__(self) -> str:
        content_only = [content["content"] for content in self.content]
        return f"Message(role='{self.role}', content={content_only!r})"


def truncate(
    tokens: list[Any],
    max_seq_len: int,
    eos_id: Any | None = None,
    truncation_type: str = "right",
) -> list[Any]:
    """
    Truncate a list of tokens to a maximum length. If eos_id is provided, the last
    token will be replaced with eos_id.

    Args:
        tokens (list[Any]): list of tokens to truncate
        max_seq_len (int): maximum length of the list
        eos_id (Any | None): token to replace the last token with. If None, the
            last token will not be replaced. Default is None.
        truncation_type (str): type of truncation to apply, either "left" or "right".
            Default is "right".

    Returns:
        list[Any]: truncated list of tokens

    Raises:
        ValueError: if truncation_type is not "left" or "right"
    """

    if truncation_type == "left":
        tokens_truncated = tokens[-max_seq_len:]  # Take the last max_seq_len tokens
    elif truncation_type == "right":
        tokens_truncated = tokens[:max_seq_len]  # Take the first max_seq_len tokens
    else:
        raise ValueError(
            f"truncation_type must be 'left' or 'right', got {truncation_type}"
        )

    # Replace the last token with eos_id if necessary
    if eos_id is not None and tokens_truncated and tokens_truncated[-1] != eos_id:
        tokens_truncated[-1] = eos_id

    return tokens_truncated


def mask_messages(
    messages: list[TuneMessage], masking_strategy: MaskingStrategy
) -> None:
    """
    Set the masked attribute for each message in the list based on the specified masking strategy.

    Args:
        messages (list[TuneMessage]): a list of messages to mask.
        masking_strategy (MaskingStrategy): masking strategy to use.
            Must be one of `train_on_all`, `train_on_assistant`, `train_on_last`.

            - ``train_on_all``: both user and assistant messages are unmasked
            - ``train_on_assistant``: user messages are masked, only assistant messages are unmasked
            - ``train_on_last``: only the last assistant message is unmasked
    """
    masking_strategy = MaskingStrategy(masking_strategy)
    marked_last_assistant_message = False
    for message in reversed(messages):
        # System messages are always masked
        if message.role == "system":
            message.masked = True
            continue
        if masking_strategy == MaskingStrategy.TRAIN_ON_LAST:
            if message.role == "assistant" and not marked_last_assistant_message:
                message.masked = False
                marked_last_assistant_message = True
            else:
                message.masked = True
        elif masking_strategy == MaskingStrategy.TRAIN_ON_ASSISTANT:
            message.masked = message.role != "assistant"


def batch_to_device(batch: dict, device: torch.device) -> None:
    """Function that takes a dictionary (or nested dictionary) of tensors and sets them
    all to the same device. This utility is intended to be used for batches of data to be
    moved to device, the update is inplace.

    Args:
        batch (dict): dict of Tensors or more nested dicts of tensors.
        device (torch.device): torch device to move the tensors to.

    Raises:
        ValueError: if batch dict contains anything other than ``torch.Tensor``.

    """
    for k, v in batch.items():
        if isinstance(v, dict):
            batch_to_device(v, device)
        elif isinstance(v, torch.Tensor):
            batch[k] = v.to(device)
        elif isinstance(v, BlockMask):
            batch[k] = v.to(device)
        else:
            raise ValueError(
                f"To use batch_to_device, all elements in the batch must be a dict, "
                f"Tensor, or BlockMask with flexattention enabled. "
                f'Got key "{k}" with value of type {type(v)}'
            )


class StopAfterOneEpoch:
    """Wraps an iterator, e.g. dataloader, and stops iterating after a rank shows that an epoch has been completed.

    In distributed eval, we may have len(dataset) % num_ranks != 0. This means that some ranks may be on epoch 0
    while others are already in epoch 1. To avoid hangs, all ranks *must* stop at the same time, requiring communication.

    This function minimzes this impact by fetching one batch in advance and perfoming overlapping async all_reduce.

    Assumes batch contains field "metrics" with at least one Metric containing "num_epochs" in its key, as it is done in
    `forge.src.data.datasets.HfIterableDataset`.

    Args:
        iter (Iterator): Iterator over dataloader batches
        device (torch.device): Device for synchronizing tensors
        dp_mesh (dist.ProcessGroup | None): Data parallel process group (None for single process)
    """

    def __init__(
        self,
        iter: Iterator,
        device: torch.device,
        dp_mesh: dist.ProcessGroup | None = None,
    ):
        self.iter = iter
        self.device = device
        self.dp_mesh = dp_mesh

        # Prefetch first batch for pipeline-style execution
        self._next_batch = next(iter)

        # Track pending async epoch sync
        self._epoch_tensor: torch.Tensor | None = None
        self._pending_work: Any = None
        self._should_stop = False

    def __iter__(self):
        return self

    def __next__(self) -> dict:
        """Get next batch from current epoch.

        Returns:
            Batch dict guaranteed to be from current epoch

        Raises:
            StopIteration: When epoch completes across all ranks
        """
        # Check if previous epoch sync completed
        if self._pending_work is not None:
            self._pending_work.wait()
            if self._epoch_tensor.item() > 0:
                self._should_stop = True
            self._pending_work = None
            self._epoch_tensor = None

        if self._should_stop:
            logger.debug("Eval epoch completed. Stopping data iterator.")
            raise StopIteration

        # Get current batch
        current_batch = self._next_batch
        current_epoch = extract_epoch_from_batch(current_batch)

        # Prefetch next batch and check for epoch change
        self._next_batch = next(self.iter)
        next_epoch = extract_epoch_from_batch(self._next_batch)
        epoch_changed = next_epoch > current_epoch

        # Start async epoch sync
        if torch.distributed.is_initialized():
            self._epoch_tensor = torch.tensor([int(epoch_changed)], device=self.device)
            self._pending_work = torch.distributed.all_reduce(
                self._epoch_tensor,
                op=torch.distributed.ReduceOp.MAX,
                group=self.dp_mesh,
                async_op=True,
            )
        elif epoch_changed:
            # if not distributed, just update the flag directly
            self._should_stop = True

        return current_batch


def extract_epoch_from_batch(batch: dict) -> int:
    """Extract epoch number from batch metrics. Useful to detect epoch changes during validation.

    Assumes batch contains field "metrics" with at least one Metric containing "num_epochs" in its key, as it is done in
    `forge.src.data.datasets.HfIterableDataset`.

    Args:
        batch (dict): Batch dictionary with 'metrics' field

    Returns:
        int: Max epoch number from metrics

    Raises:
        ValueError: If metrics key is missing or no metric with 'num_epochs' found
    """
    if "metrics" not in batch:
        raise ValueError(
            "Batch missing 'metrics' field. Cannot extract epoch from batch."
        )

    # Match metrics where 'num_epochs' appears in the key (handles prefixed keys like 'dataset/name/num_epochs')
    epochs = [metric.value for metric in batch["metrics"] if "num_epochs" in metric.key]
    if epochs:
        return max(epochs)

    raise ValueError(
        f"No 'num_epochs' metric found in batch. Got metrics: "
        f"{[m.key for m in batch['metrics']]}"
    )
