# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F

from forge.util.ops import compute_logprobs

from tests.test_utils import gpu_test
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Shard
from torch.distributed.tensor.parallel import loss_parallel
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


class TestComputeLogprobsWithLossParallel(FSDPTest):
    """Test compute_logprobs with loss_parallel context for vocab-sharded DTensors."""

    @property
    def world_size(self) -> int:
        return 2

    @gpu_test(gpu_count=2)
    def test_loss_parallel_matches_sequential(self):
        """Verify compute_logprobs under loss_parallel matches non-sharded version."""
        torch.manual_seed(42)

        batch_size, seq_len, vocab_size, target_len = 4, 16, 1000, 8
        rank = dist.get_rank()
        device = torch.device(f"cuda:{rank}")

        # Create and broadcast test data
        if rank == 0:
            full_logits = torch.randn(batch_size, seq_len, vocab_size, device=device)
            target_ids = torch.randint(
                0, vocab_size, (batch_size, target_len), device=device
            )
        else:
            full_logits = torch.empty(batch_size, seq_len, vocab_size, device=device)
            target_ids = torch.empty(
                batch_size, target_len, dtype=torch.int64, device=device
            )

        dist.broadcast(full_logits, src=0)
        dist.broadcast(target_ids, src=0)

        # Reference: non-sharded computation
        expected = compute_logprobs(full_logits, target_ids, align=True)

        # Create vocab-sharded DTensor
        mesh = init_device_mesh("cuda", (self.world_size,), mesh_dim_names=("tp",))
        local_vocab = vocab_size // self.world_size
        dtensor_logits = DTensor.from_local(
            full_logits[:, :, rank * local_vocab : (rank + 1) * local_vocab],
            mesh,
            placements=[Shard(2)],
        )

        # Compute with loss_parallel context
        with loss_parallel():
            result = compute_logprobs(dtensor_logits, target_ids, align=True)

        # Convert DTensor result to regular tensor for comparison
        if isinstance(result, DTensor):
            result = result.full_tensor()

        torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)
