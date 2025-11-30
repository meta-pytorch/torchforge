# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Parallel log-probability computation for vocab-sharded tensors.

When logits are sharded across GPUs on the vocabulary dimension (tensor parallelism),
computing log_softmax naively requires gathering the full vocab to each GPU.
This is wasteful - for a [batch, seq_len, 150k_vocab] tensor in float32,
that's ~40GB of memory just for logits.

This module provides `compute_logprobs_parallel` which computes log probabilities
distributedly, never materializing the full vocab dimension on any single GPU.

Algorithm:
---------
log_softmax(x) = x - max(x) - log(sum(exp(x - max(x))))

For sharded vocab:
1. Compute local_max → All-reduce max → global_max
2. Compute local exp(x - global_max), sum → All-reduce sum → global_sum
3. log_normalizer = global_max + log(global_sum)
4. For each target token:
   - Check if it's in this rank's shard
   - If yes: logprob = logits[target] - log_normalizer
   - If no: logprob = 0
5. All-reduce sum logprobs (only one rank has the real value)

Memory: O(batch * seq_len) per GPU instead of O(batch * seq_len * vocab_size)
"""

import torch
import torch.distributed as dist

from forge.util.ops import compute_logprobs
from torch.distributed.tensor import DTensor


def compute_logprobs_parallel(
    logits: DTensor,
    target_ids: torch.Tensor,
    temperature: float = 1.0,
    align: bool = True,
) -> torch.Tensor:
    """
    Compute log probabilities for target tokens from vocab-sharded DTensor logits.

    This function computes log_softmax(logits)[target_ids] distributedly,
    without ever gathering the full vocabulary dimension.

    IMPORTANT: Only use this when logits is a DTensor sharded on vocab dimension.
    For regular tensors or non-vocab-sharded DTensors, use compute_logprobs instead.

    Args:
        logits: DTensor of shape [batch_size, seq_len, vocab_size], sharded on dim=-1.
        target_ids: Tensor of shape [batch_size, target_len] with target token IDs.
        temperature: Temperature for scaling logits (default 1.0).
        align: If True, slice logits to align with target_ids (default True).

    Returns:
        Tensor of shape [batch_size, target_len] with log probabilities.
    """
    # Get sharding info using helper
    tp_group, tp_rank, tp_size, vocab_start, vocab_end = get_vocab_shard_info(logits)

    if tp_group is None:
        # DTensor but not sharded on vocab (Replicate or other dim sharding)
        return compute_logprobs(logits.full_tensor(), target_ids, temperature, align)

    # Get the local shard
    local_logits = logits._local_tensor  # [batch, seq_len, vocab_size / tp_size]

    # Align logits with target if needed
    if align:
        # Slice to match target length: logits[:, -target_len-1:-1, :]
        target_len = target_ids.size(1)
        local_logits = local_logits[:, -target_len - 1 : -1, :]

    # Scale by temperature
    local_logits = local_logits / temperature

    batch_size, seq_len, local_vocab_size = local_logits.shape
    device = local_logits.device

    # Move target_ids to the same device
    target_ids = target_ids.to(device)

    # Cast to float32 for numerical stability
    local_logits_fp32 = local_logits.float()

    # ============================================================
    # Step 1: Compute global max for numerical stability
    # ============================================================
    local_max = local_logits_fp32.max(dim=-1, keepdim=True).values  # [batch, seq, 1]
    global_max = local_max.clone()
    dist.all_reduce(global_max, op=dist.ReduceOp.MAX, group=tp_group)

    # ============================================================
    # Step 2: Compute global sum(exp(x - max))
    # ============================================================
    local_exp = torch.exp(local_logits_fp32 - global_max)  # [batch, seq, local_vocab]
    local_sum_exp = local_exp.sum(dim=-1, keepdim=True)  # [batch, seq, 1]
    global_sum_exp = local_sum_exp.clone()
    dist.all_reduce(global_sum_exp, op=dist.ReduceOp.SUM, group=tp_group)

    # log_normalizer = global_max + log(global_sum_exp)
    log_normalizer = global_max + torch.log(global_sum_exp)  # [batch, seq, 1]
    log_normalizer = log_normalizer.squeeze(-1)  # [batch, seq]

    # ============================================================
    # Step 3: Extract logits at target positions (only on owning rank)
    # ============================================================
    # Create mask for tokens owned by this rank (vocab_start/vocab_end from helper)
    is_local = (target_ids >= vocab_start) & (target_ids < vocab_end)

    # Convert global indices to local indices (only valid where is_local=True)
    local_indices = target_ids - vocab_start
    local_indices = local_indices.clamp(0, local_vocab_size - 1)  # Clamp for safety

    # Gather logits at target positions
    # local_logits_fp32: [batch, seq, local_vocab]
    # local_indices: [batch, seq]
    # We need logits_fp32[b, s, local_indices[b, s]]
    target_logits = torch.gather(
        local_logits_fp32,
        dim=-1,
        index=local_indices.unsqueeze(-1).long(),
    ).squeeze(
        -1
    )  # [batch, seq]

    # Zero out logits where this rank doesn't own the token
    target_logits = target_logits * is_local.float()

    # ============================================================
    # Step 4: All-reduce to combine (only one rank has non-zero value)
    # ============================================================
    dist.all_reduce(target_logits, op=dist.ReduceOp.SUM, group=tp_group)

    # ============================================================
    # Step 5: Compute final log probability
    # ============================================================
    logprobs = target_logits - log_normalizer

    return logprobs


def get_vocab_shard_info(
    logits: DTensor,
) -> tuple[dist.ProcessGroup | None, int, int, int, int]:
    """
    Get vocabulary sharding information from a DTensor.

    Args:
        logits: DTensor with shape [..., vocab_size], potentially sharded on vocab dim.

    Returns:
        Tuple of (tp_group, tp_rank, tp_size, vocab_start, vocab_end).
        If not sharded, returns (None, 0, 1, 0, vocab_size).
    """
    from torch.distributed.tensor.placement_types import Shard

    local_logits = logits._local_tensor
    placements = logits.placements
    device_mesh = logits.device_mesh

    for i, p in enumerate(placements):
        if isinstance(p, Shard) and p.dim == 2:  # vocab dimension
            tp_group = device_mesh.get_group(mesh_dim=i)
            tp_size = dist.get_world_size(tp_group)
            tp_rank = dist.get_rank(tp_group)
            local_vocab_size = local_logits.shape[-1]
            vocab_start = tp_rank * local_vocab_size
            vocab_end = vocab_start + local_vocab_size
            return tp_group, tp_rank, tp_size, vocab_start, vocab_end

    # Not sharded
    return None, 0, 1, 0, local_logits.shape[-1]
