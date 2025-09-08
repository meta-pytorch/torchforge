# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Tests for reference_actor.py - compute_logprobs function
"""

import unittest

import pytest
import torch


def _module_not_found():
    try:
        import forge.actors.reference_actor  # noqa: F401

        return False
    except ModuleNotFoundError:
        return True


class TestComputeLogprobs(unittest.TestCase):
    """Test the compute_logprobs utility function."""

    @pytest.mark.skipif(
        _module_not_found(),
        reason="Moduel not found error, likely due to missing dependencies on CI.",
    )
    def test_compute_logprobs_basic(self):
        """Test basic logprobs computation."""
        from forge.actors.reference_actor import compute_logprobs

        batch_size = 1
        seq_len = 5
        vocab_size = 1000
        response_len = 3

        logits = torch.randn(batch_size, seq_len, vocab_size)

        # Create mock input_ids for response tokens
        input_ids = torch.randint(0, vocab_size, (batch_size, response_len))

        result = compute_logprobs(logits, input_ids)

        # Verify output shape and properties
        assert isinstance(result, torch.Tensor)
        assert result.shape == (batch_size, response_len)
        assert torch.all(result <= 0)  # Log probabilities should be <= 0

    @pytest.mark.skipif(
        _module_not_found(),
        reason="Moduel not found error, likely due to missing dependencies on CI.",
    )
    def test_compute_logprobs_with_temperature(self):
        """Test logprobs computation with temperature scaling."""
        from forge.actors.reference_actor import compute_logprobs

        batch_size = 1
        seq_len = 5
        vocab_size = 1000
        response_len = 3
        temperature = 0.1

        logits = torch.randn(batch_size, seq_len, vocab_size)
        input_ids = torch.randint(0, vocab_size, (batch_size, response_len))

        result = compute_logprobs(logits, input_ids, temperature)

        assert isinstance(result, torch.Tensor)
        assert result.shape == (batch_size, response_len)
        assert torch.all(result <= 0)
        default_result = compute_logprobs(logits, input_ids)
        assert not torch.allclose(result, default_result)

    @pytest.mark.skipif(
        _module_not_found(),
        reason="Moduel not found error, likely due to missing dependencies on CI.",
    )
    def test_compute_logprobs_single_token(self):
        """Test logprobs computation with single token response."""
        from forge.actors.reference_actor import compute_logprobs

        batch_size = 1
        seq_len = 5
        vocab_size = 1000
        response_len = 1

        logits = torch.randn(batch_size, seq_len, vocab_size)
        input_ids = torch.randint(0, vocab_size, (batch_size, response_len))

        result = compute_logprobs(logits, input_ids)

        assert result.shape == (batch_size, response_len)
        assert result.numel() == 1  # Single element

    @pytest.mark.skipif(
        _module_not_found(),
        reason="Moduel not found error, likely due to missing dependencies on CI.",
    )
    def test_compute_logprobs_empty_response(self):
        """Test logprobs computation with empty response."""
        from forge.actors.reference_actor import compute_logprobs

        batch_size = 1
        seq_len = 5
        vocab_size = 1000
        response_len = 0

        logits = torch.randn(batch_size, seq_len, vocab_size)
        input_ids = torch.randint(0, vocab_size, (batch_size, response_len))

        result = compute_logprobs(logits, input_ids)

        assert result.shape == (batch_size, response_len)

    @pytest.mark.skipif(
        _module_not_found(),
        reason="Moduel not found error, likely due to missing dependencies on CI.",
    )
    def test_compute_logprobs_empty_prompt(self):
        """Test logprobs computation with empty prompt."""
        from forge.actors.reference_actor import compute_logprobs

        batch_size = 1
        vocab_size = 1000
        prompt_len = 0
        response_len = 5
        seq_len = prompt_len + response_len

        logits = torch.randn(batch_size, seq_len, vocab_size)
        input_ids = torch.randint(0, vocab_size, (batch_size, response_len))
        with pytest.raises(ValueError, match=r"(?i).*context length.*"):
            _ = compute_logprobs(logits, input_ids)
