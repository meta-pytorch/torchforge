# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Protocol

import torch

from forge.data.common import CROSS_ENTROPY_IGNORE_IDX


class Transform(Protocol):
    """
    Loose interface for all data and model transforms. Transforms operate at the
    sample level and perform operations on a sample dict, returning the updated dict.
    """

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        ...


class SFTOutputTransform(Transform):
    """Applied to each dataset sample to build the `"labels"` tensor for causal-LM SFT training.

    Expects sample to contain 1-D torch tensors
    "tokens": token IDs, dtype=torch.long
    "mask": bool/int where **True** marks positions to ignore

    If they are not tensors, they are converted to tensors.

    Produces ``"labels"`` of the same shape such that
        labels[t] =  tokens[t+1]                # shift left
        labels[t] =  IGNORE_IDX  if mask[t+1]   # respect mask
        labels[-1] = IGNORE_IDX                 # last token has no target

    All ops are vectorised; only one fresh tensor (`labels`) is allocated.
    """

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:

        tokens = sample["tokens"]
        mask = sample["mask"]

        # Sanity checks
        if not isinstance(tokens, torch.Tensor):
            tokens = torch.tensor(tokens)
        if not isinstance(mask, torch.Tensor):
            mask = torch.tensor(mask)

        if tokens.ndim != 1 or mask.ndim != 1:
            raise ValueError("Both 'tokens' and 'mask' must be 1-D tensors.")

        # build labels
        # pre-fill with IGNORE so we don’t need extra assignments later
        labels = tokens.new_full(tokens.shape, CROSS_ENTROPY_IGNORE_IDX)

        # left-shift via cheap views (no copy)
        labels[:-1].copy_(tokens[1:])

        # apply mask in-place (single fused kernel on GPU/CPU)
        labels[:-1].masked_fill_(mask[1:].bool(), CROSS_ENTROPY_IGNORE_IDX)

        # return a shallow-copied mapping so the original sample stays intact
        out = dict(sample)
        out["labels"] = labels
        return out
