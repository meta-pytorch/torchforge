# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Optional

import torch
from torch.utils.data import DataLoader


def collate_with_metrics(batch):
    """
    Simple collate function that preserves metrics for validation.
    Collects metrics from all samples in the batch and aggregates them.

    Uses a simple collation that doesn't enforce same sizes for lists/tokens.
    """
    # Collect metrics from all samples
    batch_metrics = []
    for sample in batch:
        if "metrics" in sample:
            batch_metrics.extend(sample.pop("metrics"))

    # Simple collation that handles variable-length sequences
    collated = {}
    if batch:
        for key in batch[0].keys():
            values = [sample[key] for sample in batch]
            if key == "tokens" or key == "labels":
                # Keep as list of lists for variable length sequences
                collated[key] = values
            else:
                # Use default collation for scalars
                collated[key] = torch.utils.data.default_collate(values)

    # Add batch-level metrics key for downstream processing
    if batch_metrics:
        collated["metrics"] = batch_metrics

    return collated


def generate_ckpt(
    dataloader: DataLoader,
    steps_before_checkpoint: int,
    steps_after_checkpoint: int,
    resume_dataloader: Optional[DataLoader] = None,
) -> dict[str, Any]:
    """
    Generates a checkpoint by running through data and saving checkpoint mid-stream.
    Optionally, a second dataloader can be given to resume from checkpoint
    and run steps_after_checkpoint to match the first one.

    Collects and aggregates metrics for test validation purposes.

    Args:
        dataloader (DataLoader): The dataloader to test
        steps_before_checkpoint (int): Number of steps to run before saving checkpoint
        steps_after_checkpoint (int): Number of steps to run after checkpoint
        resume_dataloader (Optional[DataLoader]): Optional new dataloader to test resuming.
            If None, returns empty resumed_batches.

    Returns:
        dict[str, Any]: Dict with batches and aggregated metrics for validation.
    """
    iterator = iter(dataloader)

    # Collect batches and metrics before and after checkpoint
    batches = []
    all_metrics = []  # All metrics collected during the run
    checkpoint_metrics = []  # Metrics collected only up to checkpoint
    checkpoint_state = None

    total_steps = steps_before_checkpoint + steps_after_checkpoint

    for idx, batch in enumerate(iterator):
        batches.append(batch)

        # Collect metrics for test validation
        if "metrics" in batch:
            batch_metrics = batch.pop("metrics")
            all_metrics.extend(batch_metrics)

            # If we haven't reached checkpoint yet, also add to checkpoint metrics
            if idx < steps_before_checkpoint:
                checkpoint_metrics.extend(batch_metrics)

        # Save checkpoint state after steps_before_checkpoint
        if idx == steps_before_checkpoint - 1:  # -1 because idx is 0-based
            checkpoint_state = {
                "loader": dataloader.state_dict(),
            }

        # Stop after total steps
        if idx == total_steps - 1:
            break

    # Split batches
    pre_checkpoint_batches = batches[:steps_before_checkpoint]
    post_checkpoint_batches = batches[steps_before_checkpoint:]

    # Resume with new instance if provided
    resumed_batches = []
    resumed_metrics = []

    if resume_dataloader is not None and checkpoint_state is not None:
        # Test resuming with new instance
        resume_dataloader.load_state_dict(checkpoint_state["loader"])
        resume_iterator = iter(resume_dataloader)

        # Collect only the post-checkpoint batches when resuming
        for idx, batch in enumerate(resume_iterator):
            resumed_batches.append(batch)

            # Collect metrics from resumed batches
            if "metrics" in batch:
                batch_metrics = batch.pop("metrics")
                resumed_metrics.extend(batch_metrics)

            # Stop after steps_after_checkpoint
            if idx == steps_after_checkpoint - 1:
                break

    return {
        # Original run
        "pre_checkpoint_batches": pre_checkpoint_batches,
        "post_checkpoint_batches": post_checkpoint_batches,
        "metrics_at_checkpoint": keep_last_metric(checkpoint_metrics),
        "final_metrics": keep_last_metric(all_metrics),
        # Resumed run
        "resumed_batches": resumed_batches,
        "resumed_metrics": keep_last_metric(resumed_metrics),
        # Internal state for loading - only if someone needs to manually load
        "_checkpoint_state": checkpoint_state,
    }


def keep_last_metric(metrics_list: list) -> dict[str, Any]:
    result = {}
    for metric in metrics_list:
        # Expect observability.Metric objects only
        key = metric.key
        value = metric.value

        # For test purposes, just keep the last value of each metric
        result[key] = value

    return result
