# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.distributed as dist
import torch.nn.functional as F

from torch.distributed.tensor import DTensor


def compute_logprobs(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    temperature: float = 1.0,
    align: bool = True,
) -> torch.Tensor:
    """
    Computes the log probabilities of the input tokens given the model logits and temperature.
    Always converts inputs to fp32 for numerical stability.

    This function handles two common usage patterns:

    **Pattern 1: Pre-aligned logits (align=False)**
    Use when logits are already aligned with input_ids, typically when you:
    - Pass input_ids to the model: model(input_ids) -> logits
    - The model outputs logits[i] that predict target_ids[i]
    - logits.shape[1] == input_ids.shape[1]

    Example:
        >>> input_ids = torch.tensor([[1, 2, 3, 4]])  # Model input
        >>> target_ids = torch.tensor([[2, 3, 4, 5]]) # Shifted by 1 (next-token prediction)
        >>> logits = model(input_ids)  # Shape: [1, 4, vocab_size]
        >>> # logits already aligned: logits[:, i] predicts target_ids[:, i]
        >>> logprobs = compute_logprobs(logits, target_ids, align=False)

    **Pattern 2: Full-sequence logits needing alignment (align=True, default)**
    Use when you have logits for the full sequence but only want log probs for a subset
    (e.g., just the response tokens, not the prompt). The function will:
    - Slice logits to match the length of input_ids
    - Take logits[:, -len(input_ids)-1:-1] to get positions that predict input_ids

    Example:
        >>> # Full sequence passed to model: [prompt + response]
        >>> full_input_ids = torch.tensor([[1, 2, 3, 4, 5, 6]])  # Prompt + response
        >>> logits = model(full_input_ids)  # Shape: [1, 6, vocab_size]
        >>> # Only want log probs for response tokens
        >>> response_tokens = torch.tensor([[4, 5, 6]])  # Just the response
        >>> logprobs = compute_logprobs(logits, response_tokens, align=True)
        >>> # Function slices logits[:, -4:-1] to get logits that predict tokens [4, 5, 6]

    The alignment logic ensures that when you have a full sequence but only want log
    probabilities for the response portion, you don't need to re-run the model. This
    is a key optimization in RL training where the prompt remains constant.

    Args:
        logits (`torch.Tensor`):
            The model output logits of shape `(batch_size, sequence_length, vocab_size)`.
        input_ids (`torch.Tensor`):
            The target token ids of shape `(batch_size, target_sequence_length)`.
            These are the tokens for which you want to compute log probabilities.
        temperature (`float`, *optional*, defaults to 1.0):
            The temperature value for scaling logits before computing log probabilities.
            Higher values make the distribution more uniform, lower values more peaked.
        align (`bool`, *optional*, defaults to True):
            If True (default), align logits with input_ids by slicing to extract the
            relevant positions from a longer sequence (Pattern 2).
            If False, assume logits are already aligned with input_ids (Pattern 1).

    Returns:
        torch.Tensor: Log probabilities of shape `(batch_size, target_sequence_length)`.
            Each element [b, i] is the log probability of input_ids[b, i] given the
            corresponding logits.

    Note:
        This function uses cross_entropy instead of log_softmax + gather for better
        numerical stability, especially important for fp16/bf16 training.
    """
    # Align logits with input_ids if requested
    if align:
        # Ignore the last token from logits because it predicts the next token (-1)
        # And align logits with the input tokens length.
        logits = logits[:, -input_ids.size(1) - 1 : -1, :].to(input_ids.device)

    scaled_logits = logits / temperature

    # Cast up to fp32 for numerical stability
    scaled_logits_fp32 = scaled_logits.float()

    # get per-token log probs
    batch_size, seq_len, vocab_size = scaled_logits_fp32.shape
    logprobs = -F.cross_entropy(
        scaled_logits_fp32.reshape(-1, vocab_size),
        input_ids.reshape(-1).long(),
        reduction="none",
    )

    return logprobs.reshape(batch_size, seq_len)


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
    tp_group, _, _, vocab_start, vocab_end = get_vocab_shard_info(logits)

    if tp_group is None:
        # DTensor but not sharded on vocab (Replicate or other dim sharding)
        return compute_logprobs(logits.full_tensor(), target_ids, temperature, align)

    local_logits = logits._local_tensor  # [batch, seq_len, vocab_size / tp_size]
    target_len = target_ids.size(1)

    if align:
        local_logits = local_logits[:, -target_len - 1 : -1, :]

    target_ids = target_ids.to(local_logits.device)
    local_logits_fp32 = local_logits.float() / temperature

    log_normalizer = _distributed_log_normalizer(local_logits_fp32, tp_group)

    local_vocab_size = local_logits_fp32.shape[-1]
    local_indices = (target_ids - vocab_start).clamp(0, local_vocab_size - 1)
    is_local = (target_ids >= vocab_start) & (target_ids < vocab_end)

    target_logits = torch.gather(
        local_logits_fp32,
        dim=-1,
        index=local_indices.unsqueeze(-1).long(),
    ).squeeze(-1)
    target_logits = target_logits.masked_fill(~is_local, 0.0)
    dist.all_reduce(target_logits, op=dist.ReduceOp.SUM, group=tp_group)

    return target_logits - log_normalizer


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


def _distributed_log_normalizer(
    local_logits_fp32: torch.Tensor,
    tp_group: dist.ProcessGroup,
) -> torch.Tensor:
    """
    Compute logsumexp across vocab shards without materializing the full vocab.
    """
    global_max = local_logits_fp32.max(dim=-1, keepdim=True).values
    dist.all_reduce(global_max, op=dist.ReduceOp.MAX, group=tp_group)

    sum_exp = torch.exp(local_logits_fp32 - global_max).sum(dim=-1, keepdim=True)
    dist.all_reduce(sum_exp, op=dist.ReduceOp.SUM, group=tp_group)

    return (global_max + torch.log(sum_exp)).squeeze(-1)
