# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import torch


def compute_logprobs(
    logits: torch.Tensor, input_ids: torch.Tensor, temperature: float = 1.0
) -> torch.Tensor:
    """
    Compute log probs of the completion input_ids given the logits of the whole sequence.
    Warning: only works if all prompts in the batch have the same length. TODO: support variable length prompts.

    Args:
        logits (torch.Tensor): (batch_size, seq_len, vocab_size), the logits output from the model.
        input_ids (torch.Tensor): (batch_size, completion_len), the token ids for the completion.

    Returns:
        torch.Tensor: (batch_size, completion_len), the log probabilities of the completion tokens.

    Raises:
        ValueError: If the inferred context length is less than or equal to 0.
    """
    context_len = logits.shape[1] - input_ids.shape[1]
    completion_len = input_ids.shape[1]
    if context_len <= 0:
        raise ValueError(
            "Context length must be greater than 0. Otherwise the probability of the first token is undefined."
        )

    # (bsz, completion_len, vocab_size)
    logits = logits[:, context_len - 1 : -1, :]
    assert logits.shape == (
        input_ids.shape[0],
        completion_len,
        logits.shape[-1],
    ), f"logits shape incorrect, {logits.shape=}, {input_ids.shape=}, {logits.shape[-1]=}"
    token_logprobs = torch.log_softmax(logits / temperature, dim=-1)
    # (bsz, completion_len, 1)
    logprobs = torch.gather(token_logprobs, 2, input_ids.unsqueeze(-1))
    # (bsz, completion_len)
    logprobs = logprobs.squeeze(-1)

    return logprobs
