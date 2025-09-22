# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch


def compute_advantages(rewards: list[float]) -> list[float]:
    """Compute advantages for GRPO using reward signals"""
    # TODO: add batch processing
    if not rewards:
        return []
    rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
    mean = rewards_tensor.mean()
    std = rewards_tensor.std()
    advantages = (rewards_tensor - mean) / (std + 1e-4)
    return advantages.tolist()


def compute_logprobs(
    logits: torch.Tensor, input_ids: torch.Tensor, temperature: float = 1.0
) -> torch.Tensor:
    context_length = logits.shape[1] - input_ids.shape[1]
    logits = logits[:, context_length - 1 : -1]
    logprobs = torch.log_softmax(logits / temperature, dim=-1).to(input_ids.device)
    logprobs = torch.gather(logprobs, 2, input_ids.unsqueeze(-1)).squeeze(-1)
    return logprobs


def simple_grpo_loss(
    logits: torch.Tensor,
    response: torch.Tensor,
    ref_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    padding_mask: torch.Tensor,
    beta: float = 0.1,
) -> torch.Tensor:
    logprobs = compute_logprobs(logits, response)
    kl = torch.exp(ref_logprobs - logprobs) - (ref_logprobs - logprobs) - 1
    per_token_policy_loss = torch.exp(logprobs - logprobs.detach()) * advantages
    per_token_loss = -(per_token_policy_loss - beta * kl)
    loss = (
        ((per_token_loss * padding_mask).sum(dim=1))
        / (padding_mask.sum(dim=1).clamp(min=1.0))
    ).mean()
    return loss
