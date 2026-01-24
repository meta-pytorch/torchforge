# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from forge.data_models.completion import Completion


@dataclass
class Episode:
    episode_id: str
    pad_id: int
    request_len: int
    response_len: int
    target: Any | None = None
    request: str | None = None
    response: str | None = None
    # Processed data
    completion: Completion | None = None
    generator_logprobs: torch.Tensor | None = None  # [seq_len]
    ref_logprobs: torch.Tensor | None = None  # [seq_len]
    reward: float | None = None
    reward_breakdown: dict[str, float] | None = None
    advantage: float | None = None
    loss_mask: torch.Tensor | None = None

    @property
    def policy_version(self) -> int | None:
        return self.completion.generator_version

    @property
    def request_tensor(self) -> torch.Tensor:
        tensor: torch.Tensor = self.completion.prompt_ids.to(torch.long)
        if tensor.shape[0] < self.request_len:  # left pad
            diff = self.request_len - tensor.shape[0]
            tensor = F.pad(tensor, (diff, 0), value=self.pad_id)
        return tensor

    @property
    def response_tensor(self) -> torch.Tensor:
        tensor: torch.Tensor = self.completion.token_ids.to(torch.long)
        if tensor.shape[0] < self.response_len:  # right pad
            diff = self.response_len - tensor.shape[0]
            tensor = F.pad(tensor, (0, diff), value=self.pad_id)
        return tensor

    def to_dict(self, exclude: list[str] | None = None) -> dict[str, Any]:
        """Convert episode to dict, optionally excluding specified fields."""
        result = {
            "episode_id": self.episode_id,
            "policy_version": self.policy_version,
            "prompt": self.request,
            "response": self.response,
            "target": str(self.target),
            "reward": self.reward,
            "advantage": self.advantage,
            "request_len": self.request_len,
            "response_len": self.response_len,
            "pad_id": self.pad_id,
            "ref_logprobs": self.ref_logprobs,
            "completion": self.completion,
        }

        if self.reward_breakdown is not None and "reward_breakdown" not in exclude:
            result.update(self.reward_breakdown)

        if exclude:
            for key in exclude:
                result.pop(key, None)

        return result


# Represents the group (G) of episodes in GRPO
Group = list[Episode]
