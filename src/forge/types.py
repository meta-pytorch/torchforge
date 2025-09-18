# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import uuid
from dataclasses import dataclass, field
from typing import Any, TypedDict, Union

import torch
import torch.nn.functional as F


class Message(TypedDict):
    role: str
    content: str | dict[str, Any]
    tools: dict[str, Any] | None


@dataclass
class ForgeEnvInfo:
    """Environment info returned with observations."""

    episode_id: str | None = None
    step_count: int = 0
    metadata: dict | None = None


@dataclass(kw_only=True)
class Observation:
    """Base class for environment observations.

    Contract:
    - Should contain all information needed by an agent to make decisions
    - Should be serializable/deserializable
    - Should be immutable (or treated as such)
    Args:
        done: Whether the episode/conversation is complete
        reward: Optional reward signal (can be boolean, int, or float)
        metadata: Additional data that doesn't affect agent decisions but may be useful
                 for transforms, logging, evaluation, etc.
    """

    done: bool = False
    reward: bool | int | float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(kw_only=True)
class Action:
    """Base class for environment actions.

    Contract:
    - Should contain all information needed to execute a step in the environment
    - Should be serializable/deserializable
    - Should be immutable (or treated as such)

    Args:
        metadata: Additional data that may be useful for logging, debugging, or transforms
    """

    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Trajectory:
    """A trajectory containing a sequence of states, actions, etc."""

    policy_version: int
    states: list[Observation] = field(default_factory=list)
    actions: list[Action] = field(default_factory=list)

    def __post_init__(self):
        assert self.policy_version >= 0


@dataclass(kw_only=True)
class State:
    """Base class for environment state.

    Contract:
    - Should contain all information needed to restore the environment
    - Should be serializable/deserializable
    - May contain information not exposed in observations

    Args:
        metadata: Additional state information that may be useful for debugging or analysis
    """

    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProcessConfig:
    """A proc_mesh config for the torchx scheduler."""

    num_procs: int = 1
    with_gpus: bool = False
    num_hosts: int | None = None


@dataclass
class ServiceConfig:
    """A service config."""

    procs_per_replica: int
    num_replicas: int
    with_gpus: bool = False
    hosts_per_replica: int | None = None
    # ServiceConfig-specific fields
    health_poll_rate: float = 0.2
    replica_max_concurrent_requests: int = 10
    return_first_rank_result: bool = (
        True  # Whether or not to auto-unwrap ValueMesh to first rank's result
    )

    def to_process_config(self) -> ProcessConfig:
        """Extract ProcessConfig from this ServiceConfig.
        Maps procs_per_replica to num_procs for ProcessConfig.
        """
        return ProcessConfig(
            num_procs=self.procs_per_replica,
            with_gpus=self.with_gpus,
            num_hosts=self.hosts_per_replica,
        )


Scalar = Union[int, float]


@dataclass
class Episode:
    # TODO: add adtional layer for multi-turn
    episode_id: str
    request: str
    policy_version: int
    pad_id: int
    request_len: int
    response_len: int
    target: Any | None = None
    # processed data
    response: str | None = None
    request_tokens: list[int] | None = None
    response_tokens: list[int] | None = None
    ref_logprobs: torch.Tensor | None = None
    reward: float | None = None
    advantage: float | None = None
    response_logprobs: torch.Tensor | None = None

    @property
    def request_tensor(self):
        tensor = torch.tensor(self.request_tokens, dtype=torch.long)
        if tensor.shape[0] < self.request_len:  # left pad
            diff = self.request_len - tensor.shape[0]
            tensor = F.pad(tensor, (diff, 0), value=self.pad_id)
        return tensor

    @property
    def response_tensor(self):
        tensor = torch.tensor(self.response_tokens, dtype=torch.long)
        if tensor.shape[0] < self.response_len:  # right pad
            diff = self.response_len - tensor.shape[0]
            tensor = F.pad(tensor, (0, diff), value=self.pad_id)
        return tensor

    @property
    def max_seq_len(self) -> int:
        """
        Get maximum sequence length for this episode.

        Returns:
            int: Total length (request_len + response_len) before any truncation
        """
        return self.request_len + self.response_len

    @property
    def episode_ids(self) -> torch.Tensor:
        """
        Get complete episode trajectory as concatenated token sequence.

        Returns:
            torch.Tensor: Full sequence [request_tokens + response_tokens].
                         Shape: [request_len + response_len]
        """
        prompt_ids = torch.LongTensor(self.request_tokens)
        token_ids = torch.LongTensor(self.response_tokens)
        ids = torch.cat([prompt_ids, token_ids])
        return ids

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
    def loss_mask(self) -> torch.Tensor:
        """
        Get mask for computing loss only on response tokens.

        Returns:
            torch.Tensor: Binary mask (0 for prompt, 1 for response) shifted to align
                         with target_ids. Shape: [max_seq_len - 1]
        """
        prompt_ids = torch.LongTensor(self.request_tokens)
        token_ids = torch.LongTensor(self.response_tokens)
        loss_mask = torch.cat(
            [
                torch.zeros(
                    len(prompt_ids), dtype=torch.float32
                ),  # Don't compute loss on prompt
                torch.ones(
                    len(token_ids), dtype=torch.float32
                ),  # Compute loss on response
            ]
        )

        loss_mask = loss_mask[1:]  # Shift to align with target_ids (truncates BOS)
        return loss_mask

    @property
    def sampling_log_probs(self) -> torch.Tensor:
        """
        Get log probabilities from the sampling policy (for importance sampling).

        Returns:
            torch.Tensor: Log probabilities from policy that generated the response,
                         with zeros for prompt positions. Shifted to align with target_ids.
                         Shape: [max_seq_len - 1]
        """
        if self.response_logprobs is None:
            return torch.zeros(self.max_seq_len - 1, dtype=torch.float32)
        prompt_ids = torch.LongTensor(self.request_tokens)
        sampling_log_probs = torch.cat(
            [
                torch.zeros(prompt_ids.shape, dtype=torch.float32),
                self.response_logprobs,
            ]
        )
        sampling_log_probs = sampling_log_probs[1:]  # Shift log probs
        return sampling_log_probs

    @property
    def weighted_advantages(self) -> torch.Tensor:
        """
        Get advantages weighted by loss mask for REINFORCE training.

        Returns:
            torch.Tensor: Advantage values masked to response tokens only.
                         Zero for prompt positions, advantage value for response positions.
                         Shape: [max_seq_len - 1]
        """
        if self.advantage is None:
            return torch.zeros_like(self.loss_mask)
        return self.loss_mask * self.advantage


@dataclass
class Group:
    group_id: str
    episodes: list[Episode]

    @classmethod
    def new_group(
        cls,
        group_id: int,
        group_size: int,
        request: str,
        policy_version: int,
        pad_id: int,
        request_len: int,
        response_len: int,
        target: Any = None,
    ):
        episodes = []
        for _ in range(group_size):
            episodes.append(
                Episode(
                    episode_id=str(uuid.uuid4()),
                    request=request,
                    policy_version=policy_version,
                    pad_id=pad_id,
                    request_len=request_len,
                    response_len=response_len,
                    target=target,
                )
            )
        return cls(str(group_id), episodes)
