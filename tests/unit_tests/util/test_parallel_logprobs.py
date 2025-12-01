# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Tests for parallel logprobs computation.

Verifies that compute_logprobs_parallel produces identical results to
compute_logprobs when the logits are sharded across GPUs.
"""

import torch
import torch.distributed as dist

from forge.util.ops import (
    compute_logprobs,
    compute_logprobs_parallel,
    get_vocab_shard_info,
)
from tests.test_utils import gpu_test
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Shard
from torch.testing._internal.common_fsdp import FSDPTest


class TestParallelLogprobs(FSDPTest):
    """Test parallel logprobs against reference implementation."""

    @property
    def world_size(self) -> int:
        return 2

    @gpu_test(gpu_count=2)
    def test_parallel_logprobs_matches_sequential(self):
        """Verify parallel logprobs produces same results as sequential version."""
        torch.manual_seed(42)

        batch_size = 4
        seq_len = 16
        vocab_size = 1000  # Must be divisible by world_size
        target_len = 8

        rank = dist.get_rank()
        device = torch.device(f"cuda:{rank}")

        # Create test data on rank 0 and broadcast to ensure consistency
        if rank == 0:
            # Full logits tensor (what we'd have without sharding)
            full_logits = torch.randn(
                batch_size, seq_len, vocab_size, dtype=torch.float32, device=device
            )
            # Target tokens for logprob computation
            target_ids = torch.randint(
                0, vocab_size, (batch_size, target_len), device=device
            )
        else:
            full_logits = torch.empty(
                batch_size, seq_len, vocab_size, dtype=torch.float32, device=device
            )
            target_ids = torch.empty(
                batch_size, target_len, dtype=torch.int64, device=device
            )

        # Broadcast to all ranks
        dist.broadcast(full_logits, src=0)
        dist.broadcast(target_ids, src=0)

        # Compute reference result using sequential version
        expected = compute_logprobs(full_logits, target_ids, align=True)

        # Create device mesh for tensor parallel
        mesh = init_device_mesh("cuda", (self.world_size,), mesh_dim_names=("tp",))

        # Create DTensor sharded on vocab dimension (dim=2)
        # Each rank gets vocab_size // world_size columns
        dtensor_logits = DTensor.from_local(
            full_logits[
                :, :, rank * (vocab_size // 2) : (rank + 1) * (vocab_size // 2)
            ],
            mesh,
            placements=[Shard(2)],  # Shard on vocab dimension
        )

        # Compute parallel result
        result = compute_logprobs_parallel(dtensor_logits, target_ids, align=True)

        # Verify results match
        torch.testing.assert_close(
            result,
            expected,
            atol=1e-5,
            rtol=1e-5,
            msg="Parallel logprobs should match sequential version",
        )

    @gpu_test(gpu_count=2)
    def test_parallel_logprobs_with_temperature(self):
        """Test parallel logprobs with different temperature values."""
        torch.manual_seed(123)

        batch_size = 2
        seq_len = 10
        vocab_size = 500
        target_len = 5

        rank = dist.get_rank()
        device = torch.device(f"cuda:{rank}")

        if rank == 0:
            full_logits = torch.randn(
                batch_size, seq_len, vocab_size, dtype=torch.float32, device=device
            )
            target_ids = torch.randint(
                0, vocab_size, (batch_size, target_len), device=device
            )
        else:
            full_logits = torch.empty(
                batch_size, seq_len, vocab_size, dtype=torch.float32, device=device
            )
            target_ids = torch.empty(
                batch_size, target_len, dtype=torch.int64, device=device
            )

        dist.broadcast(full_logits, src=0)
        dist.broadcast(target_ids, src=0)

        mesh = init_device_mesh("cuda", (self.world_size,), mesh_dim_names=("tp",))
        local_vocab = vocab_size // self.world_size
        dtensor_logits = DTensor.from_local(
            full_logits[:, :, rank * local_vocab : (rank + 1) * local_vocab],
            mesh,
            placements=[Shard(2)],
        )

        for temperature in [0.5, 1.0, 2.0]:
            expected = compute_logprobs(
                full_logits, target_ids, temperature=temperature, align=True
            )
            result = compute_logprobs_parallel(
                dtensor_logits, target_ids, temperature=temperature, align=True
            )
            torch.testing.assert_close(
                result,
                expected,
                atol=1e-5,
                rtol=1e-5,
                msg=f"Failed with temperature={temperature}",
            )

    @gpu_test(gpu_count=2)
    def test_parallel_logprobs_align_false(self):
        """Test parallel logprobs with align=False."""
        torch.manual_seed(456)

        batch_size = 3
        seq_len = 8
        vocab_size = 200

        rank = dist.get_rank()
        device = torch.device(f"cuda:{rank}")

        if rank == 0:
            full_logits = torch.randn(
                batch_size, seq_len, vocab_size, dtype=torch.float32, device=device
            )
            # With align=False, target_ids same length as seq_len
            target_ids = torch.randint(
                0, vocab_size, (batch_size, seq_len), device=device
            )
        else:
            full_logits = torch.empty(
                batch_size, seq_len, vocab_size, dtype=torch.float32, device=device
            )
            target_ids = torch.empty(
                batch_size, seq_len, dtype=torch.int64, device=device
            )

        dist.broadcast(full_logits, src=0)
        dist.broadcast(target_ids, src=0)

        expected = compute_logprobs(full_logits, target_ids, align=False)

        mesh = init_device_mesh("cuda", (self.world_size,), mesh_dim_names=("tp",))
        local_vocab = vocab_size // self.world_size
        dtensor_logits = DTensor.from_local(
            full_logits[:, :, rank * local_vocab : (rank + 1) * local_vocab],
            mesh,
            placements=[Shard(2)],
        )

        result = compute_logprobs_parallel(dtensor_logits, target_ids, align=False)

        torch.testing.assert_close(
            result,
            expected,
            atol=1e-5,
            rtol=1e-5,
            msg="Parallel logprobs with align=False should match",
        )

    @gpu_test(gpu_count=2)
    def test_parallel_logprobs_numerical_stability(self):
        """Test parallel logprobs handles extreme values correctly."""
        torch.manual_seed(789)

        batch_size = 2
        seq_len = 4
        vocab_size = 100
        target_len = 2

        rank = dist.get_rank()
        device = torch.device(f"cuda:{rank}")

        # Test with large values
        if rank == 0:
            full_logits = torch.randn(
                batch_size, seq_len, vocab_size, dtype=torch.float32, device=device
            )
            # Add some extreme values
            full_logits[:, :, 0] = 1000.0
            full_logits[:, :, 50] = -1000.0
            target_ids = torch.randint(
                0, vocab_size, (batch_size, target_len), device=device
            )
        else:
            full_logits = torch.empty(
                batch_size, seq_len, vocab_size, dtype=torch.float32, device=device
            )
            target_ids = torch.empty(
                batch_size, target_len, dtype=torch.int64, device=device
            )

        dist.broadcast(full_logits, src=0)
        dist.broadcast(target_ids, src=0)

        expected = compute_logprobs(full_logits, target_ids, align=True)

        mesh = init_device_mesh("cuda", (self.world_size,), mesh_dim_names=("tp",))
        local_vocab = vocab_size // self.world_size
        dtensor_logits = DTensor.from_local(
            full_logits[:, :, rank * local_vocab : (rank + 1) * local_vocab],
            mesh,
            placements=[Shard(2)],
        )

        result = compute_logprobs_parallel(dtensor_logits, target_ids, align=True)

        # Should not have NaN or Inf
        assert torch.isfinite(result).all(), "Result contains NaN or Inf"
        assert torch.isfinite(expected).all(), "Expected contains NaN or Inf"

        torch.testing.assert_close(
            result,
            expected,
            atol=1e-4,  # Slightly relaxed for extreme values
            rtol=1e-4,
            msg="Parallel logprobs should be numerically stable",
        )

    @gpu_test(gpu_count=2)
    def test_get_vocab_shard_info(self):
        """Test vocab shard info extraction."""
        torch.manual_seed(111)

        batch_size = 2
        seq_len = 4
        vocab_size = 100

        rank = dist.get_rank()
        device = torch.device(f"cuda:{rank}")

        full_logits = torch.randn(
            batch_size, seq_len, vocab_size, dtype=torch.float32, device=device
        )

        mesh = init_device_mesh("cuda", (self.world_size,), mesh_dim_names=("tp",))
        local_vocab = vocab_size // self.world_size
        dtensor_logits = DTensor.from_local(
            full_logits[:, :, rank * local_vocab : (rank + 1) * local_vocab],
            mesh,
            placements=[Shard(2)],
        )

        tp_group, tp_rank, tp_size, vocab_start, vocab_end = get_vocab_shard_info(
            dtensor_logits
        )

        assert tp_group is not None, "Should have TP group for sharded tensor"
        assert tp_rank == rank, f"TP rank should be {rank}, got {tp_rank}"
        assert tp_size == self.world_size, f"TP size should be {self.world_size}"
        assert vocab_start == rank * local_vocab, "Vocab start incorrect"
        assert vocab_end == (rank + 1) * local_vocab, "Vocab end incorrect"


class TestParallelLogprobs4GPU(FSDPTest):
    """Test parallel logprobs with 4 GPUs."""

    @property
    def world_size(self) -> int:
        return 4

    @gpu_test(gpu_count=4)
    def test_parallel_logprobs_4_way_sharding(self):
        """Test with 4-way vocab sharding."""
        torch.manual_seed(999)

        batch_size = 8
        seq_len = 32
        vocab_size = 1000  # Divisible by 4
        target_len = 16

        rank = dist.get_rank()
        device = torch.device(f"cuda:{rank}")

        if rank == 0:
            full_logits = torch.randn(
                batch_size, seq_len, vocab_size, dtype=torch.float32, device=device
            )
            target_ids = torch.randint(
                0, vocab_size, (batch_size, target_len), device=device
            )
        else:
            full_logits = torch.empty(
                batch_size, seq_len, vocab_size, dtype=torch.float32, device=device
            )
            target_ids = torch.empty(
                batch_size, target_len, dtype=torch.int64, device=device
            )

        dist.broadcast(full_logits, src=0)
        dist.broadcast(target_ids, src=0)

        expected = compute_logprobs(full_logits, target_ids, align=True)

        mesh = init_device_mesh("cuda", (self.world_size,), mesh_dim_names=("tp",))
        local_vocab = vocab_size // self.world_size
        dtensor_logits = DTensor.from_local(
            full_logits[:, :, rank * local_vocab : (rank + 1) * local_vocab],
            mesh,
            placements=[Shard(2)],
        )

        result = compute_logprobs_parallel(dtensor_logits, target_ids, align=True)

        torch.testing.assert_close(
            result,
            expected,
            atol=1e-5,
            rtol=1e-5,
            msg="4-way parallel logprobs should match sequential",
        )
