"""Utility functions for evaluation to make main.py more concise."""

import logging
from typing import Any, Callable, Iterator

import torch
from torch import nn

logger = logging.getLogger(__name__)


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """Move all tensors in batch to specified device.

    Args:
        batch: Dictionary containing batch data
        device: Target device

    Returns:
        Batch with tensors moved to device (modifies in-place and returns)
    """
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device)
    return batch


def extract_epoch_from_batch(batch: dict) -> int | None:
    """Extract epoch number from batch metrics.

    Args:
        batch: Batch dictionary with 'metrics' field

    Returns:
        Epoch number from metrics, or None if not found
    """
    if "metrics" in batch:
        for metric in batch["metrics"]:
            if hasattr(metric, "metric_name") and metric.metric_name == "num_epochs":
                return metric.value
    return None


def start_epoch_sync(
    epoch_increment: int,
    device: torch.device,
    dp_process_group: Any = None,
) -> tuple[torch.Tensor | None, Any]:
    """Start async all_reduce for epoch synchronization across ranks.

    Args:
        epoch_increment: Difference between current and starting epoch
        device: Device for tensor
        dp_process_group: Data parallel process group (None = default group)

    Returns:
        Tuple of (epoch_tensor, pending_work) for async operation, or (None, None) if not initialized
    """
    if not torch.distributed.is_initialized():
        return None, None

    epoch_tensor = torch.tensor([epoch_increment], dtype=torch.long, device=device)
    pending_work = torch.distributed.all_reduce(
        epoch_tensor,
        op=torch.distributed.ReduceOp.MAX,
        group=dp_process_group,
        async_op=True,
    )
    return epoch_tensor, pending_work


def check_epoch_complete(
    pending_work: Any,
    epoch_tensor: torch.Tensor | None,
) -> bool:
    """Wait for async epoch sync and check if epoch completed.

    Args:
        pending_work: Pending async all_reduce work
        epoch_tensor: Tensor containing epoch increment

    Returns:
        True if any rank completed an epoch, False otherwise
    """
    if pending_work is None:
        return False

    pending_work.wait()
    if epoch_tensor is not None:
        return bool((epoch_tensor > 0).any().item())
    return False


def eval_loop(
    dataloader_iter: Iterator,
    forward_fn: Callable[[dict, torch.Tensor], torch.Tensor],
    device: torch.device,
    eval_steps: int,
    dataset_name: str,
    dp_process_group: Any = None,
    extract_epoch_fn: Callable[[dict], int | None] = extract_epoch_from_batch,
    log_interval: int = 10,
) -> tuple[float, int]:
    """Run evaluation loop with epoch synchronization.

    Args:
        dataloader_iter: Iterator over validation data
        forward_fn: Function that takes (batch_dict, labels_tensor) and returns loss tensor
        device: Device for computation
        eval_steps: Maximum number of eval steps (0 = no limit)
        dataset_name: Name for logging
        dp_process_group: Data parallel process group for epoch sync
        extract_epoch_fn: Function to extract epoch from batch
        log_interval: Log every N batches

    Returns:
        Tuple of (avg_loss, num_batches)
    """
    total_loss = torch.tensor(0.0, device=device)
    num_batches, starting_epoch = 0, None

    # Prefetch first batch
    next_batch = next(dataloader_iter)
    should_break, pending_work, epoch_tensor = False, None, None

    with torch.no_grad():
        while True:
            # Check if previous epoch sync completed
            if pending_work is not None:
                should_break = check_epoch_complete(pending_work, epoch_tensor)
                pending_work = None

            if should_break:
                logger.info(
                    f"[{dataset_name}] Epoch completed across all ranks - stopping evaluation"
                )
                break

            if eval_steps > 0 and num_batches >= eval_steps:
                logger.info(f"[{dataset_name}] Reached eval_steps cap of {eval_steps}")
                break

            batch = next_batch

            # Track starting epoch
            current_epoch = extract_epoch_fn(batch)
            if starting_epoch is None:
                starting_epoch = current_epoch

            # Prefetch next batch and start async epoch check
            try:
                next_batch = next(dataloader_iter)
                next_epoch = extract_epoch_fn(next_batch)

                # Only check epochs if both are available
                if next_epoch is not None and starting_epoch is not None:
                    epoch_increment = next_epoch - starting_epoch
                    if torch.distributed.is_initialized():
                        epoch_tensor, pending_work = start_epoch_sync(
                            epoch_increment, device, dp_process_group
                        )
                    else:
                        should_break = epoch_increment > 0
            except StopIteration:
                should_break = True

            # Process current batch (overlaps with async all_reduce)
            move_batch_to_device(batch, device)
            labels = batch.pop("labels")
            loss = forward_fn(batch, labels)
            total_loss += loss
            num_batches += 1

            if num_batches % log_interval == 0:
                logger.info(
                    f"  [{dataset_name}] Eval batch {num_batches} | Loss: {loss:.4f}"
                )

    avg_loss = (total_loss / max(num_batches, 1)).item()
    logger.info(
        f"[{dataset_name}] COMPLETE | Val Loss: {avg_loss:.4f} | Batches: {num_batches}"
    )

    return avg_loss, num_batches


async def evaluate_single_dataset(
    val_dataloader: Any,
    dataset_name: str,
    forward_fn: Callable[[dict, torch.Tensor], torch.Tensor],
    device: torch.device,
    eval_steps: int,
    dp_process_group: Any = None,
    extract_epoch_fn: Callable[[dict], int | None] = extract_epoch_from_batch,
) -> dict[str, float]:
    """Evaluate on a single validation dataset with epoch synchronization.

    Args:
        val_dataloader: DataLoader for this validation dataset
        dataset_name: Name of the dataset (for logging)
        forward_fn: Function that takes (batch_dict, labels_tensor) and returns loss
        device: Device for computation
        eval_steps: Maximum number of eval steps
        dp_process_group: Data parallel process group
        extract_epoch_fn: Function to extract epoch from batch

    Returns:
        Dict with metrics: {"val_loss": float, "val_batches": int}
    """
    avg_loss, num_batches = eval_loop(
        dataloader_iter=iter(val_dataloader),
        forward_fn=forward_fn,
        device=device,
        eval_steps=eval_steps,
        dataset_name=dataset_name,
        dp_process_group=dp_process_group,
        extract_epoch_fn=extract_epoch_fn,
        log_interval=10,
    )

    return {"val_loss": avg_loss, "val_batches": num_batches}


async def run_evaluation(
    val_dataloaders: dict[str, Any],
    model_parts: list[nn.Module],
    forward_fn: Callable[[dict, torch.Tensor], torch.Tensor],
    device: torch.device,
    eval_steps: int,
    dp_process_group: Any = None,
    extract_epoch_fn: Callable[[dict], int | None] = extract_epoch_from_batch,
) -> dict[str, dict[str, float]]:
    """Run evaluation on multiple validation datasets.

    Evaluates on all configured validation datasets and returns per-dataset metrics.
    Sets models to eval mode before evaluation and back to train mode after.

    Args:
        val_dataloaders: Dict mapping dataset names to dataloaders
        model_parts: List of model parts (for setting eval/train mode)
        forward_fn: Function that takes (batch_dict, labels_tensor) and returns loss
        device: Device for computation
        eval_steps: Maximum number of eval steps per dataset
        dp_process_group: Data parallel process group
        extract_epoch_fn: Function to extract epoch from batch

    Returns:
        Dict mapping dataset name to metrics dict, e.g.:
        {
            "val_in_domain": {"val_loss": 2.5, "val_batches": 100},
            "val_out_domain": {"val_loss": 3.1, "val_batches": 100}
        }
    """
    logger.info("=" * 50)
    logger.info("STARTING EVALUATION")
    logger.info("=" * 50)

    # Set models to eval mode
    for model_part in model_parts:
        model_part.eval()

    all_metrics = {}

    # Evaluate on each dataset
    for dataset_name, val_dataloader in val_dataloaders.items():
        logger.info(f"\n{'='*50}")
        logger.info(f"Evaluating on dataset: {dataset_name}")
        logger.info(f"{'='*50}")

        dataset_metrics = await evaluate_single_dataset(
            val_dataloader=val_dataloader,
            dataset_name=dataset_name,
            forward_fn=forward_fn,
            device=device,
            eval_steps=eval_steps,
            dp_process_group=dp_process_group,
            extract_epoch_fn=extract_epoch_fn,
        )
        all_metrics[dataset_name] = dataset_metrics

    # Set models back to train mode
    for model_part in model_parts:
        model_part.train()

    logger.info("\n" + "=" * 50)
    logger.info("EVALUATION COMPLETE - Summary:")
    for dataset_name, metrics in all_metrics.items():
        logger.info(
            f"  {dataset_name}: Loss={metrics['val_loss']:.4f}, Batches={metrics['val_batches']}"
        )
    logger.info("=" * 50)

    return all_metrics


def get_dp_process_group(parallel_dims: Any) -> Any:
    """Get the Data Parallel process group for epoch synchronization.

    Returns the DP process group if DP parallelism is enabled, otherwise None.
    This ensures all_reduce only happens across ranks with different data.

    Args:
        parallel_dims: ParallelDims object containing parallel configuration

    Returns:
        DP process group or None if not available/needed
    """
    if not torch.distributed.is_initialized():
        return None

    if parallel_dims is None:
        return None

    # Check if DP is enabled
    if not parallel_dims.dp_enabled:
        # No DP parallelism, use default group (all ranks)
        return None

    try:
        # Get the "dp" submesh which contains only DP dimensions (dp_replicate + dp_shard)
        # This excludes TP and PP ranks which should already be synchronized
        dp_mesh = parallel_dims.world_mesh.get_group("dp")
        return dp_mesh
    except Exception as e:
        logger.warning(f"Could not get DP process group, using default: {e}")
        return None
