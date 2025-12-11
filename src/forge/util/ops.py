# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F
from torch.distributed.tensor import DTensor


def compute_logprobs(
    logits: torch.Tensor | DTensor,
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

    **Tensor Parallelism Support:**
    When logits is a DTensor sharded on the vocab dimension (e.g., from tensor parallel
    training), wrap calls to this function with `loss_parallel()` context:

        >>> from torch.distributed.tensor.parallel import loss_parallel
        >>> with loss_parallel():
        ...     logprobs = compute_logprobs(logits, input_ids)

    The `loss_parallel` context ensures F.cross_entropy works correctly with
    vocab-sharded DTensors without needing to gather the full tensor.

    Args:
        logits (`torch.Tensor`):
            The model output logits of shape `(batch_size, sequence_length, vocab_size)`.
            Can be a regular Tensor or a DTensor (when using with loss_parallel context).
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
