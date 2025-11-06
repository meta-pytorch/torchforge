# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Trainer protocol.

This file defines the unified training interface compatible
with all supported torchforge trainers.
"""

from typing import Any, Protocol, runtime_checkable

import torch


@runtime_checkable
class Trainer(Protocol):
    """Protocol for all trainers in torchforge."""

    async def accumulate_gradients(
        self, microbatch: dict[str, torch.Tensor]
    ) -> dict[str, Any]:
        """Accumulate gradients from one microbatch.

        Does NOT clear gradients - they accumulate on top of existing.
        Can be called multiple times before optim_step().

        Returns:
            dict with keys:
                - loss: float
                - metrics: dict[str, float]
        """
        ...

    async def optim_step(self, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Apply optimizer step and clear gradients after.

        Returns:
            dict with keys:
                - step: int
                - learning_rate: float
                - accumulated_microbatches: int
        """
        ...

    async def clear_gradients(self) -> None:
        """Clear accumulated gradients without applying."""
        ...

    async def forward(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Run forward pass, no backward.

        Returns:
            dict with key:
                - logits: torch.Tensor
        """
        ...

    async def forward_backward(
        self, data: list[dict[str, torch.Tensor]]
    ) -> dict[str, Any]:
        """Clear first, then forward+backward on all items in data.

        Convenience wrapper equivalent to:
            clear_gradients() + accumulate_gradients() for each item

        Does NOT call optim_step() - you must call it separately.

        Returns:
            dict with keys:
                - loss: float
                - metrics: dict[str, float]
        """
        ...

    async def save_state(self, name: str) -> dict[str, Any]:
        """Save the checkpoint.

        Returns:
            dict with keys:
                - path: str
                - step: int
        """
        ...

    async def load_state(self, path: str) -> dict[str, Any]:
        """Load checkpoint.

        Returns:
            dict with keys:
                - step: int
                - learning_rate: float
        """
        ...

    async def save_weights_for_sampler(self, name: str) -> dict[str, Any]:
        """Export weights for inference.

        Returns:
            dict with keys:
                - path: str
                - version: str or int
        """
        ...

    def get_tokenizer(self):
        """Get the tokenizer.

        Returns:
            PreTrainedTokenizer
        """
        ...
