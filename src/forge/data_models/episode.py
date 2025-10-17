# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Sequence

import torch

from forge.data_models.scored_completion import ScoredCompletion


@dataclass
class Episode:
    """
    The Episode data class to be used by the trainer.

    Episodes are usually generated from a scored completion and running various post processing steps.
    """

    # Concatenated prompt and sample token ids.
    ids: torch.Tensor

    # The mask for the target ids, 0 for prompt tokens, 1 for sample tokens.
    mask: torch.Tensor

    # The weight to apply to the loss of each target token. It's normally computed
    # from the advantage and the reward.
    weights: torch.Tensor

    # The log probabilities of the target tokens, for prompt part it's set to 0,
    # for generation part it's computed from the Generator/Sampler.
    log_probs: torch.Tensor | None = None

    # TODO: add more fields as required
    state: str = ""


def from_scored_completion(scored_completion: ScoredCompletion) -> Episode:
    """Converts a ScoredCompletion to an Episode."""
    prompt_ids = scored_completion.completion.prompt_ids
    token_ids = scored_completion.completion.token_ids
    log_probs = scored_completion.completion.log_probs
    ids = torch.cat([prompt_ids, token_ids])
    mask = torch.cat(
        [
            torch.zeros(prompt_ids.shape, dtype=torch.float32),
            torch.ones_like(token_ids, dtype=torch.float32),
        ]
    )
    advantage = scored_completion.score
    weights = mask * advantage
    log_probs = torch.cat(
        [
            torch.zeros(prompt_ids.shape, dtype=torch.float32),
            # TODO: this only works if sample.log_probs is 1
            log_probs,
        ]
    )
    return Episode(ids=ids, mask=mask, weights=weights, log_probs=log_probs)


def from_scored_completions(
    scored_completions: Sequence[ScoredCompletion],
) -> Sequence[Episode]:
    """Converts a sequence of ScoredCompletion to a sequence of Episodes."""
    return [from_scored_completion(sc) for sc in scored_completions]
