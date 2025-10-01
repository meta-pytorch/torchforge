# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
import torch.nn.functional as F
from forge.util.ops import compute_logprobs


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
