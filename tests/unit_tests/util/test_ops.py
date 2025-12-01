# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F

from forge.util.ops import (
    compute_logprobs,
    compute_logprobs_parallel,
    get_vocab_shard_info,
)

from tests.test_utils import gpu_test
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Shard
from torch.testing._internal.common_fsdp import FSDPTest


def _textbook_log_softmax(logits: torch.Tensor, input_ids: torch.Tensor):
    # Helper: Textbook Log Softmax
    log_probs = F.log_softmax(logits, dim=-1)
    return torch.gather(log_probs, dim=-1, index=input_ids.unsqueeze(-1)).squeeze(-1)


class TestComputeLogprobs:
    def test_single_batch_item(self):
        """Test with single batch item."""
        # Shape: (1, 2, 3)
        logits = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])
        # Shape: (1, 1)
        input_ids = torch.tensor([[1]])
        result = compute_logprobs(logits, input_ids)

        # Manual calculation
        expected_logits = torch.tensor([[[1.0, 2.0, 3.0]]])
        expected = _textbook_log_softmax(expected_logits, input_ids)

        assert torch.allclose(result, expected, atol=1e-5)
        assert result.shape == (1, 1)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        # Shape: (1, 3, 3)
        logits = torch.tensor([[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]])
        # Shape: (1, 2)
        input_ids = torch.tensor([[2, 0]])
        result = compute_logprobs(logits, input_ids)

        # Manual calculation
        expected_logits = torch.tensor([[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]])
        expected = _textbook_log_softmax(expected_logits, input_ids)

        assert torch.allclose(result, expected, atol=1e-5)
        assert result.shape == (1, 2)

    @pytest.mark.timeout(10)
    def test_multi_batch(self):
        """Test with multiple batch items."""
        # Shape: (2, 2, 3)
        logits = torch.tensor(
            [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[0.5, 1.5, 2.5], [3.5, 4.5, 5.5]]]
        )
        # Shape: (2, 1)
        input_ids = torch.tensor([[1], [2]])
        result = compute_logprobs(logits, input_ids)

        # Manual calculation
        expected_logits = torch.tensor([[[1.0, 2.0, 3.0]], [[0.5, 1.5, 2.5]]])
        expected = _textbook_log_softmax(expected_logits, input_ids)

        assert torch.allclose(result, expected, atol=1e-5)
        assert result.shape == (2, 1)

    @pytest.mark.timeout(10)
    def test_temperature(self):
        """Test with different temperature values."""
        batch_size, seq_len, vocab_size = 2, 4, 6
        logits = torch.randn(batch_size, seq_len, vocab_size)
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len - 1))

        # Manual calculation with temperature scaling
        def _manual(temperature: float):
            expected_logits = logits[:, 0:-1] / temperature
            return _textbook_log_softmax(expected_logits, input_ids)

        temperatures = [1.0, 2.0, 4.5]
        for temperature in temperatures:
            result = compute_logprobs(logits, input_ids, temperature=temperature)
            expected = _manual(temperature)
            assert torch.allclose(result, expected, atol=1e-5)
            assert result.shape == input_ids.shape

    @pytest.mark.timeout(10)
    def test_edge_cases(self):
        """Test edge cases."""
        # Test with very large values (numerical stability)
        logits = torch.tensor([[[1000.0, 2000.0], [1500.0, 2500.0]]])
        input_ids = torch.tensor([[0]])
        result = compute_logprobs(logits, input_ids)
        # Should not be NaN or inf
        assert torch.isfinite(result).all()

        # Test with very small values
        logits = torch.tensor([[[-1000.0, -2000.0], [-1500.0, -2500.0]]])
        input_ids = torch.tensor([[1]])
        result = compute_logprobs(logits, input_ids)
        # Should not be NaN or inf
        assert torch.isfinite(result).all()

    def test_compute_logprobs_empty_response(self):
        """Test logprobs computation with empty response."""
        batch_size, seq_len, vocab_size = 1, 5, 1000
        logits = torch.randn(batch_size, seq_len, vocab_size)
        input_ids = torch.tensor([[]])

        result = compute_logprobs(logits, input_ids)
        assert result.shape == (batch_size, 0)

    @pytest.mark.timeout(10)
    def test_align_parameter_false(self):
        """Test with align=False (pre-aligned logits)."""
        # When align=False, logits are already aligned with input_ids
        # logits[:, i] predicts input_ids[:, i]
        batch_size, seq_len, vocab_size = 2, 3, 5
        logits = torch.randn(batch_size, seq_len, vocab_size)
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

        result = compute_logprobs(logits, input_ids, align=False)

        # Manual calculation without slicing
        expected = _textbook_log_softmax(logits, input_ids)

        assert torch.allclose(result, expected, atol=1e-5)
        assert result.shape == input_ids.shape

    @pytest.mark.timeout(10)
    def test_align_parameter_true(self):
        """Test with align=True (default, needs slicing)."""
        # When align=True, logits need to be sliced to align with input_ids
        batch_size, full_seq_len, vocab_size = 2, 6, 5
        logits = torch.randn(batch_size, full_seq_len, vocab_size)

        # We want log probs for just the last 3 tokens
        target_len = 3
        input_ids = torch.randint(0, vocab_size, (batch_size, target_len))

        result = compute_logprobs(logits, input_ids, align=True)

        # Manual calculation: align=True slices logits[:, -target_len-1:-1]
        sliced_logits = logits[:, -target_len - 1 : -1, :]
        expected = _textbook_log_softmax(sliced_logits, input_ids)

        assert torch.allclose(result, expected, atol=1e-5)
        assert result.shape == input_ids.shape

    @pytest.mark.timeout(10)
    def test_align_comparison(self):
        """Test that align=True properly slices logits."""
        batch_size, seq_len, vocab_size = 1, 4, 10
        logits = torch.randn(batch_size, seq_len, vocab_size)
        input_ids = torch.randint(0, vocab_size, (batch_size, 2))

        result_aligned = compute_logprobs(logits, input_ids, align=True)

        # Manually slice the same way align=True does
        sliced_logits = logits[:, -input_ids.size(1) - 1 : -1, :]
        result_manual = compute_logprobs(sliced_logits, input_ids, align=False)

        # Both should give the same result
        assert torch.allclose(result_aligned, result_manual, atol=1e-5)


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
