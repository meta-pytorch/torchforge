# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Optional

import torch
from forge.data_models.completion import Completion
from forge.util.ops import pad_sequence


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
    advantage_weights: torch.Tensor

    # The log probabilities of the target tokens, for prompt part it's set to 0,
    # for generation part it's computed from the Generator/Sampler.
    logprobs: Optional[torch.Tensor] = None

    # The reward score for this episode
    reward: float = 0.0

    # the pad_id token for the tokenizer
    pad_id: int = 0

    # The log probabilities of the target tokens, generated from the reference model
    ref_logprobs: Optional[torch.Tensor] = None

    # max sequence length of the episode
    max_seq_len: int = 512

    @property
    def episode_ids(self) -> torch.Tensor:
        """
        Just another alias for ids.
        """
        return self.ids

    @property
    def input_ids(self) -> torch.Tensor:
        """
        Get model input tokens for next-token prediction.

        Returns:
            torch.Tensor: Episode trajectory with EOS truncated for model input.
                         Shape: [max_seq_len - 1]
        """
        input_ids = self.episode_ids[:-1]  # truncate EOS
        return input_ids

    @property
    def target_ids(self) -> torch.Tensor:
        """
        Get target tokens for next-token prediction training.

        Returns:
            torch.Tensor: Episode trajectory shifted by 1 position (BOS truncated).
                         Aligned with input_ids for teacher forcing.
                         Shape: [max_seq_len - 1]
        """
        target_ids = self.episode_ids[1:]  # truncate BOS
        return target_ids

    @property
    def input_ids_paded(self) -> torch.Tensor:
        """
        Get model input tokens for next-token prediction, but with padding added

        Returns:
            torch.Tensor: Episode trajectory with EOS truncated for model input.
                         Shape: [max_seq_len - 1]
        """
        return pad_sequence(self.input_ids, self.max_seq_len - 1, self.pad_id)

    @property
    def target_ids_paded(self) -> torch.Tensor:
        """
        Get target tokens for next-token prediction training.

        Returns:
            torch.Tensor: Episode trajectory shifted by 1 position (BOS truncated).
                         Aligned with input_ids for teacher forcing.
                         Shape: [max_seq_len - 1]
        """
        return pad_sequence(self.target_ids, self.max_seq_len - 1, self.pad_id)

    @property
    def input_ids_attn_mask_padded(self) -> torch.Tensor:
        """
        Get's the attention mask for the input_ids with padding

        Returns:
            torch.Tensor: Attention mask for the input_ids, 1 for tokens all the tokens except pad_id
        """
        return self.input_ids_paded != self.pad_id

    @property
    def loss_mask(self) -> torch.Tensor:
        """
        Get mask for computing loss only on response tokens.

        Returns:
            torch.Tensor: Binary mask (0 for prompt, 1 for response) shifted to align
                         with target_ids. Shape: [max_seq_len - 1]
        """
        return self.loss_mask[1:]  # Shift to align with target_ids (truncates BOS)

    @property
    def loss_mask_paded(self) -> torch.Tensor:
        """
        Get mask for computing loss only on response tokens with padding.

        Returns:
            torch.Tensor: Binary mask (0 for prompt, 1 for response) shifted to align
                         with target_ids. Shape: [max_seq_len - 1]
        """
        return pad_sequence(self.loss_mask, self.max_seq_len - 1, self.pad_id)

    @property
    def weighted_advantages_padded(self) -> torch.Tensor:
        """
        Get advantages weighted by loss mask.

        Returns:
            torch.Tensor: Advantage values masked to response tokens only.
                         Zero for prompt positions, advantage value for response positions.
                         Shape: [max_seq_len - 1]
        """
        if self.advantage_weights is None:
            return torch.zeros_like(self.loss_mask)
        return self.loss_mask * self.advantage_weights


def from_completion(
    completion: Completion,
    reward: float = 0.0,
    pad_id: int = 0,
    max_seq_len: int = 512,
    ref_logprobs: Optional[torch.Tensor] = None,
) -> Episode:
    """Converts a ScoredCompletion to an Episode."""
    prompt_ids = completion.prompt_ids
    token_ids = completion.token_ids
    logprobs = completion.logprobs
    ids = torch.cat([prompt_ids, token_ids])
    mask = torch.cat(
        [
            torch.zeros(prompt_ids.shape, dtype=torch.float32),
            torch.ones_like(token_ids, dtype=torch.float32),
        ]
    )
    advantage = reward  # simple case
    advantage_weights = mask * advantage
    logprobs = torch.cat(
        [
            torch.zeros(prompt_ids.shape, dtype=torch.float32),
            # TODO: this only works if sample.log_probs is 1
            logprobs,
        ]
    )
    return Episode(
        ids=ids,
        mask=mask,
        advantage_weights=advantage_weights,
        logprobs=logprobs,
        pad_id=pad_id,
        max_seq_len=max_seq_len,
        ref_logprobs=ref_logprobs,
    )
