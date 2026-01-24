# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from forge.rl.loss import compute_logprobs, compute_ratio


def assert_close(actual, expected, atol=1e-4, rtol=1e-4):
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


def get_metric(metrics, key: str):
    for m in metrics:
        if m.key == key:
            return m.value
    raise KeyError(f"Metric '{key}' not found")


@pytest.fixture
def inputs():
    """Expected values generated using P2120280632"""
    torch.manual_seed(42)
    B, S, V = 2, 4, 10

    logits = torch.randn(B, S, V)
    target_ids = torch.randint(0, V, (B, S))

    # Seq 0: mild divergence, Seq 1: high divergence (triggers clipping)
    generator_logprobs = torch.tensor(
        [
            [-2.0, -2.1, -1.9, -2.0],
            [-6.0, -1.0, -5.0, -0.5],
        ]
    )
    ref_logprobs = torch.randn(B, S) * 0.5 - 2.0
    advantages = torch.randn(B, S)

    # Interleaved mask (multi-turn pattern)
    loss_mask = torch.tensor([[1, 0, 1, 0], [1, 1, 0, 0]], dtype=torch.float)

    # Pre-compute common values
    logprobs, _ = compute_logprobs(logits, target_ids)
    ratio, log_ratio, _ = compute_ratio(logprobs, generator_logprobs, loss_mask)

    return {
        "B": B,
        "S": S,
        "V": V,
        "logits": logits,
        "target_ids": target_ids,
        "generator_logprobs": generator_logprobs,
        "ref_logprobs": ref_logprobs,
        "advantages": advantages,
        "loss_mask": loss_mask,
        "logprobs": logprobs,
        "ratio": ratio,
        "log_ratio": log_ratio,
    }
