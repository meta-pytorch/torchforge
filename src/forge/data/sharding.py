# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod

import torch


class BaseSharding(ABC):
    """
    Abstract base class for tensor parallel sharding strategies.
    """

    def __init__(self, tensor_parallel_size: int, rank: int):
        self.tensor_parallel_size = tensor_parallel_size
        self.rank = rank

    @abstractmethod
    def load_from_source_to_target(
        self,
        param_name: str,
        source_tensor: torch.Tensor,
        target_tensor: torch.Tensor,
    ) -> None:
        """
        Copy a source tensor to a target tensor, handling sharding and replication.

        Args:
            param_name: Name of the parameter being loaded
            source_tensor: Source tensor to load from
            target_tensor: Target tensor to load into
        """
        pass


class Llama3vLLMSharding(BaseSharding):
    """
    Llama3 vLLM specific tensor parallel sharding strategy.
    """

    def load_from_source_to_target(
        self,
        param_name: str,
        source_tensor: torch.Tensor,
        target_tensor: torch.Tensor,
    ) -> None:
        """
        Copy a source tensor to a target tensor, handling sharding and replication.
        """
        # Determine sharding strategy for this parameter
        shard_dim, is_sharded = self._get_tensor_parallel_sharding_strategy(param_name)

        if not is_sharded:
            # Parameter is replicated - shapes should match exactly
            if source_tensor.shape != target_tensor.shape:
                raise ValueError(
                    f"Replicated parameter {param_name} has mismatched shapes: "
                    f"{source_tensor.shape} vs {target_tensor.shape}, skipping"
                )

            # Direct copy for replicated parameters
            target_tensor.copy_(source_tensor)
        else:
            # Need to shard the full tensor
            sharded_tensor = self._calculate_tensor_shard(
                source_tensor, shard_dim, self.tensor_parallel_size, self.rank
            )

            if sharded_tensor.shape != target_tensor.shape:
                raise ValueError(
                    f"Calculated shard for {param_name} has wrong shape: "
                    f"{sharded_tensor.shape} vs expected {target_tensor.shape}, skipping"
                )

            target_tensor.copy_(sharded_tensor)

    def _get_tensor_parallel_sharding_strategy(
        self, param_name: str
    ) -> tuple[int, bool]:
        """
        Determine the sharding strategy for a parameter in tensor parallel setup.

        Returns:
            tuple[int, bool]: (shard_dimension, is_sharded)
                - shard_dimension: Which dimension to shard (0 or 1)
                - is_sharded: Whether this parameter should be sharded at all

        Based on vLLM's tensor parallel implementation for LLaMA models:
        - Embedding layers: shard along vocab dimension (dim 0)
        - Attention projections: qkv_proj shard along hidden dimension (dim 0), o_proj along input dimension (dim 1)
        - MLP projections: gate/up_proj shard along hidden dimension (dim 0), down_proj along input dimension (dim 1)
        - Layer norms: not sharded (replicated)
        - Output layer: shard along vocab dimension (dim 0)
        """
        # Parameters that are not sharded (replicated across all tensor parallel ranks)
        if any(keyword in param_name for keyword in ["norm", "bias", "rotary_emb"]):
            return 0, False

        # Embedding layers - shard along vocab dimension (dim 0)
        if "embed_tokens" in param_name or "lm_head" in param_name:
            return 0, True

        # Attention projections
        if "qkv_proj" in param_name:
            # Input projections: shard output dimension (dim 0)
            return 0, True
        elif "o_proj" in param_name:
            # Output projection: shard input dimension (dim 1)
            return 1, True

        # MLP projections
        elif any(
            proj in param_name for proj in ["gate_proj", "up_proj", "gate_up_proj"]
        ):
            # Input projections: shard output dimension (dim 0)
            return 0, True
        elif "down_proj" in param_name:
            # Output projection: shard input dimension (dim 1)
            return 1, True

        # Default: try to infer from tensor shape patterns
        return 0, True

    def _calculate_tensor_shard(
        self,
        full_tensor: torch.Tensor,
        shard_dim: int,
        tensor_parallel_size: int,
        rank: int,
    ) -> torch.Tensor:
        """
        Calculate the shard of a full tensor for the current tensor parallel rank.

        Args:
            full_tensor: The full tensor to shard
            shard_dim: Which dimension to shard along (0 or 1)
            tensor_parallel_size: Number of tensor parallel ranks
            rank: Current rank (will be modulo by tensor_parallel_size)

        Returns:
            torch.Tensor: The sharded tensor for this rank
        """
        tp_rank = rank % tensor_parallel_size
        tensor_size = full_tensor.shape[shard_dim]

        if tensor_size % tensor_parallel_size != 0:
            raise ValueError(
                f"Cannot shard tensor dimension {shard_dim} with size {tensor_size} "
                f"across {tensor_parallel_size} ranks: not evenly divisible"
            )

        shard_size = tensor_size // tensor_parallel_size
        start_idx = tp_rank * shard_size
        end_idx = start_idx + shard_size

        # Create index tensor for the shard range
        indices = torch.arange(start_idx, end_idx, device=full_tensor.device)

        if shard_dim == 0:
            return torch.index_select(full_tensor, 0, indices)
        elif shard_dim == 1:
            return torch.index_select(full_tensor, 1, indices)
        else:
            raise ValueError(f"Unsupported shard dimension: {shard_dim}")
