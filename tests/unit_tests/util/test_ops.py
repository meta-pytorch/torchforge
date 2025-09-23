# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
import torch.nn.functional as F
from forge.util.ops import selective_log_softmax


class TestOps:
    @pytest.mark.timeout(10)
    def test_basic_2d(self):
        """Test basic 2D case."""
        logits = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        index = torch.tensor([0, 2])  # Select positions 0 and 2
        result = selective_log_softmax(logits, index)
        # Compare with torch's implementation
        expected = torch.gather(
            F.log_softmax(logits, dim=-1), dim=-1, index=index.unsqueeze(-1)
        ).squeeze(-1)
        assert torch.allclose(result, expected, atol=1e-5)
        assert result.shape == (2,)  # Same shape as index

    @pytest.mark.timeout(10)
    def test_single_row(self):
        """Test with single row."""
        logits = torch.tensor([[1.0, 2.0, 3.0]])
        index = torch.tensor([1])  # Select middle element
        result = selective_log_softmax(logits, index)
        # Manual calculation: log_softmax then select index 1
        log_probs = F.log_softmax(logits, dim=-1)
        expected = log_probs[0, 1]
        assert torch.allclose(result, expected)
        assert result.shape == (1,)

    @pytest.mark.timeout(10)
    def test_different_dtypes(self):
        """Test with different data types."""
        logits_f32 = torch.randn(2, 4, dtype=torch.float32)
        logits_bf16 = torch.randn(2, 4, dtype=torch.bfloat16)
        index = torch.tensor([0, 3])
        result_f32 = selective_log_softmax(logits_f32, index)
        result_bf16 = selective_log_softmax(logits_bf16, index)
        # Check output dtypes match input dtypes
        assert result_f32.dtype == torch.float32
        assert result_bf16.dtype == torch.bfloat16
        # Check shapes
        assert result_f32.shape == (2,)
        assert result_bf16.shape == (2,)

    @pytest.mark.timeout(10)
    def test_3d_tensor(self):
        """Test with 3D tensor."""
        batch, seq, vocab = 2, 3, 5
        logits = torch.randn(batch, seq, vocab)
        index = torch.randint(0, vocab, (batch, seq))
        result = selective_log_softmax(logits, index)
        # Should have same shape as index
        assert result.shape == (batch, seq)
        # All values should be negative (log probabilities)
        assert (result <= 0).all()

    @pytest.mark.timeout(10)
    def test_known_values(self):
        """Test with known values for manual verification."""
        # Simple case where we can calculate by hand
        logits = torch.tensor([[0.0, 0.0]])  # Equal logits
        index = torch.tensor([0])
        result = selective_log_softmax(logits, index)
        # log_softmax of [0, 0] gives [-log(2), -log(2)]
        # Selecting index 0 should give -log(2)
        expected = -torch.log(torch.tensor(2.0))
        assert torch.allclose(result, expected, atol=1e-6)

    @pytest.mark.timeout(10)
    def test_edge_cases(self):
        """Test edge cases."""
        # Test with single class
        logits = torch.tensor([[5.0]])
        index = torch.tensor([0])
        result = selective_log_softmax(logits, index)
        # log_softmax of single element is 0
        assert torch.allclose(result, torch.tensor([0.0]))
        # Test with large values (numerical stability)
        logits = torch.tensor([[100.0, 200.0]])
        index = torch.tensor([1])
        result = selective_log_softmax(logits, index)
        # Should not be NaN or inf
        assert torch.isfinite(result).all()
