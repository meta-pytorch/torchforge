# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Utility functions for SFT training actors.

These utilities handle data loading, model setup, and common operations.
"""

import logging
import os
from functools import partial
from typing import Any, Optional

import torch
from forge.data.collate import collate_packed
from forge.data.datasets.packed import PackedDataset, TextPacker
from forge.data.datasets.sft_dataset import AlpacaToMessages, sft_iterable_dataset
from forge.data.tokenizer import HuggingFaceModelTokenizer
from torchdata.stateful_dataloader import StatefulDataLoader
from torchtitan.distributed import ParallelDims, utils as dist_utils

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def setup_tokenizer(
    hf_assets_path: str,
    tokenizer_filename: str = "tokenizer.json",
    tokenizer_config_filename: str = "tokenizer_config.json",
    generation_config_filename: str = "generation_config.json",
) -> HuggingFaceModelTokenizer:
    """
    Setup HuggingFace tokenizer from model assets.

    Args:
        hf_assets_path: Path to the directory containing tokenizer files
        tokenizer_filename: Name of the tokenizer JSON file
        tokenizer_config_filename: Name of the tokenizer config JSON file
        generation_config_filename: Name of the generation config JSON file

    Returns:
        Initialized HuggingFaceModelTokenizer
    """
    tokenizer_json_path = os.path.join(hf_assets_path, tokenizer_filename)
    tokenizer_config_path = os.path.join(hf_assets_path, tokenizer_config_filename)
    generation_config_path = os.path.join(hf_assets_path, generation_config_filename)

    logger.info(f"Loading tokenizer from: {tokenizer_json_path}")

    tokenizer = HuggingFaceModelTokenizer(
        tokenizer_json_path=tokenizer_json_path,
        tokenizer_config_json_path=tokenizer_config_path,
        generation_config_path=generation_config_path,
    )

    return tokenizer


def setup_sft_dataloader(
    tokenizer: HuggingFaceModelTokenizer,
    dataset_path: str,
    dataset_split: str,
    target_tokens_per_pack: int,
    batch_size: int,
    device: torch.device,
    padding_idx: int = 0,
    message_transform: Optional[Any] = None,
) -> StatefulDataLoader:
    """
    Setup dataloader for SFT training.

    Args:
        tokenizer: Tokenizer to use for processing text
        dataset_path: Path or name of the dataset (e.g., "yahma/alpaca-cleaned")
        dataset_split: Dataset split to use (e.g., "train", "validation")
        target_tokens_per_pack: Target sequence length for packing
        batch_size: Batch size for training
        device: Device to move tensors to
        padding_idx: Padding token index
        message_transform: Transform to convert dataset format to messages

    Returns:
        Configured StatefulDataLoader
    """
    if message_transform is None:
        message_transform = AlpacaToMessages()

    logger.info(f"Loading SFT dataset from: {dataset_path}, split: {dataset_split}")

    dataset = sft_iterable_dataset(
        model_transform=tokenizer,
        message_transform=message_transform,
        path=dataset_path,
        split=dataset_split,
    )

    packer = TextPacker(padding_idx=padding_idx)
    dataset = PackedDataset(
        dataset=dataset,
        packer=packer,
        target_tokens_per_pack=target_tokens_per_pack,
    )

    dataloader = StatefulDataLoader(
        dataset=dataset,
        batch_size=batch_size,
        collate_fn=partial(
            collate_packed, mask_fn=packer.create_block_mask, device=device
        ),
    )

    logger.info(
        f"Created dataloader with batch_size={batch_size}, target_tokens={target_tokens_per_pack}"
    )

    return dataloader


def create_context_parallel_context(
    parallel_dims: ParallelDims,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    model_parts: list,
    rotate_method: str,
):
    """
    Create context parallel context for distributed training.

    Args:
        parallel_dims: Parallel dimensions configuration
        inputs: Input tensor
        labels: Label tensor
        model_parts: List of model parts
        rotate_method: Context parallel rotation method

    Returns:
        Context parallel context or None if CP is not enabled
    """
    if not parallel_dims.cp_enabled:
        return None

    return dist_utils.create_context_parallel_ctx(
        cp_mesh=parallel_dims.world_mesh["cp"],
        cp_buffers=[inputs, labels] + [m.freqs_cis for m in model_parts],
        cp_seq_dims=[1, 1] + [0 for _ in model_parts],
        cp_no_restore_buffers={inputs, labels},
        cp_rotate_method=rotate_method,
    )


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """
    Move batch tensors to the specified device.

    Args:
        batch: Dictionary containing batch data
        device: Target device

    Returns:
        Batch with tensors moved to device
    """
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value.to(device)
    return batch


def log_training_step(
    step: int,
    total_steps: int,
    loss: torch.Tensor,
    logger: logging.Logger,
):
    """
    Log training step information.

    Args:
        step: Current training step
        total_steps: Total number of training steps
        loss: Current loss value
        logger: Logger instance
    """
    logger.info(f"Step {step}/{total_steps} | Loss: {loss.item():.4f}")
