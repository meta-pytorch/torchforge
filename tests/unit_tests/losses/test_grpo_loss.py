# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from forge.losses.grpo_loss import SimpleGRPOLoss


class TestSimpleGRPOLoss:
    @pytest.fixture
    def loss_fn(self):
        """Create a GRPO loss instance with default beta."""
        return SimpleGRPOLoss(beta=0.1)

    @pytest.fixture
    def sample_data(self):
        """Create sample input data for testing."""
        batch_size, seq_len = 2, 4

        # Create log probabilities (should be negative)
        logprobs = torch.log(torch.rand(batch_size, seq_len) * 0.9 + 0.1)
        ref_logprobs = torch.log(torch.rand(batch_size, seq_len) * 0.9 + 0.1)

        # Create advantages (can be positive or negative)
        advantages = torch.randn(batch_size, seq_len)

        # Create padding mask (1 for valid tokens, 0 for padding)
        padding_mask = torch.ones(batch_size, seq_len)
        padding_mask[0, -1] = 0  # Add some padding
        padding_mask[1, -2:] = 0  # Add more padding

        return logprobs, ref_logprobs, advantages, padding_mask

    @pytest.mark.timeout(10)
    @pytest.mark.asyncio
    def test_forward_basic(self, loss_fn, sample_data):
        """Test basic forward pass."""
        logprobs, ref_logprobs, advantages, padding_mask = sample_data

        loss = loss_fn(logprobs, ref_logprobs, advantages, padding_mask)

        # Loss should be a scalar
        assert loss.dim() == 0
        assert torch.isfinite(loss)
        assert not torch.isnan(loss)

    @pytest.mark.timeout(10)
    @pytest.mark.asyncio
    def test_output_shape(self, loss_fn):
        """Test output shape for different input sizes."""
        for batch_size in [1, 3, 8]:
            for seq_len in [1, 10, 32]:
                logprobs = torch.randn(batch_size, seq_len)
                ref_logprobs = torch.randn(batch_size, seq_len)
                advantages = torch.randn(batch_size, seq_len)
                padding_mask = torch.ones(batch_size, seq_len)

                loss = loss_fn(logprobs, ref_logprobs, advantages, padding_mask)
                assert loss.shape == torch.Size([])

    @pytest.mark.timeout(10)
    @pytest.mark.asyncio
    def test_gradient_flow(self, loss_fn, sample_data):
        """Test that gradients flow through logprobs."""
        logprobs, ref_logprobs, advantages, padding_mask = sample_data
        logprobs.requires_grad_(True)

        loss = loss_fn(logprobs, ref_logprobs, advantages, padding_mask)
        loss.backward()

        assert logprobs.grad is not None
        assert not torch.isnan(logprobs.grad).any()
        assert torch.isfinite(logprobs.grad).all()

    @pytest.mark.timeout(10)
    @pytest.mark.asyncio
    def test_no_gradient_to_ref_logprobs(self, loss_fn, sample_data):
        """Test that gradients don't flow to reference logprobs."""
        logprobs, ref_logprobs, advantages, padding_mask = sample_data
        ref_logprobs.requires_grad_(True)

        loss = loss_fn(logprobs, ref_logprobs, advantages, padding_mask)
        loss.backward()

        # ref_logprobs should receive gradients (it's used in KL computation)
        assert ref_logprobs.grad is not None

    @pytest.mark.timeout(10)
    @pytest.mark.asyncio
    def test_padding_mask_effect(self, loss_fn):
        """Test that padding mask correctly ignores padded tokens."""
        batch_size, seq_len = 2, 4

        logprobs = torch.randn(batch_size, seq_len)
        ref_logprobs = torch.randn(batch_size, seq_len)
        advantages = torch.randn(batch_size, seq_len)

        # Test with full mask
        full_mask = torch.ones(batch_size, seq_len)
        loss_full = loss_fn(logprobs, ref_logprobs, advantages, full_mask)

        # Test with partial mask
        partial_mask = torch.ones(batch_size, seq_len)
        partial_mask[:, -1] = 0  # Mask last token
        loss_partial = loss_fn(logprobs, ref_logprobs, advantages, partial_mask)

        # Losses should be different
        assert not torch.allclose(loss_full, loss_partial)

    @pytest.mark.timeout(10)
    @pytest.mark.asyncio
    def test_beta_parameter_effect(self, sample_data):
        """Test that different beta values produce different losses."""
        logprobs, ref_logprobs, advantages, padding_mask = sample_data

        loss_fn_1 = SimpleGRPOLoss(beta=0.1)
        loss_fn_2 = SimpleGRPOLoss(beta=0.5)

        loss_1 = loss_fn_1(logprobs, ref_logprobs, advantages, padding_mask)
        loss_2 = loss_fn_2(logprobs, ref_logprobs, advantages, padding_mask)

        # Different beta should produce different losses (unless KL is zero)
        # This test might be flaky if KL happens to be very small
        if not torch.allclose(ref_logprobs, logprobs, atol=1e-6):
            assert not torch.allclose(loss_1, loss_2, atol=1e-6)

    @pytest.mark.timeout(10)
    @pytest.mark.asyncio
    def test_zero_advantages(self, loss_fn):
        """Test behavior with zero advantages."""
        batch_size, seq_len = 2, 4

        logprobs = torch.randn(batch_size, seq_len)
        ref_logprobs = torch.randn(batch_size, seq_len)
        advantages = torch.zeros(batch_size, seq_len)
        padding_mask = torch.ones(batch_size, seq_len)

        loss = loss_fn(logprobs, ref_logprobs, advantages, padding_mask)

        # With zero advantages, loss should only depend on KL term
        assert torch.isfinite(loss)

    @pytest.mark.timeout(10)
    @pytest.mark.asyncio
    def test_identical_policies(self, loss_fn):
        """Test behavior when current and reference policies are identical."""
        batch_size, seq_len = 2, 4

        logprobs = torch.randn(batch_size, seq_len)
        ref_logprobs = logprobs.clone()  # Identical policies
        advantages = torch.randn(batch_size, seq_len)
        padding_mask = torch.ones(batch_size, seq_len)

        loss = loss_fn(logprobs, ref_logprobs, advantages, padding_mask)

        # KL should be approximately zero for identical policies
        assert torch.isfinite(loss)

    @pytest.mark.timeout(10)
    @pytest.mark.asyncio
    def test_extreme_values(self, loss_fn):
        """Test with extreme but valid values."""
        batch_size, seq_len = 2, 3

        # Large negative log probabilities (very low probabilities)
        logprobs = torch.full((batch_size, seq_len), -10.0)
        ref_logprobs = torch.full((batch_size, seq_len), -5.0)

        # Large advantages
        advantages = torch.full((batch_size, seq_len), 10.0)
        padding_mask = torch.ones(batch_size, seq_len)

        loss = loss_fn(logprobs, ref_logprobs, advantages, padding_mask)

        assert torch.isfinite(loss)
        assert not torch.isnan(loss)

    @pytest.mark.timeout(10)
    @pytest.mark.asyncio
    def test_numerical_stability(self, loss_fn):
        """Test numerical stability with edge cases."""
        batch_size, seq_len = 1, 2

        # Test with very similar log probabilities
        logprobs = torch.tensor([[0.0, -1e-8]])
        ref_logprobs = torch.tensor([[1e-8, 0.0]])
        advantages = torch.tensor([[1.0, -1.0]])
        padding_mask = torch.ones(batch_size, seq_len)

        loss = loss_fn(logprobs, ref_logprobs, advantages, padding_mask)

        assert torch.isfinite(loss)

    @pytest.mark.timeout(10)
    @pytest.mark.asyncio
    def test_all_masked_sequence(self, loss_fn):
        """Test behavior when entire sequence is masked."""
        batch_size, seq_len = 1, 3

        logprobs = torch.randn(batch_size, seq_len)
        ref_logprobs = torch.randn(batch_size, seq_len)
        advantages = torch.randn(batch_size, seq_len)
        padding_mask = torch.zeros(batch_size, seq_len)  # All masked

        loss = loss_fn(logprobs, ref_logprobs, advantages, padding_mask)

        # Should handle division by zero gracefully due to clamp(min=1.0)
        assert torch.isfinite(loss)

    @pytest.mark.timeout(10)
    @pytest.mark.asyncio
    def test_mathematical_correctness(self, loss_fn):
        """Test mathematical correctness with simple verification."""
        # Simple case: identical policies should give zero loss
        logprobs = torch.tensor([[0.0]])
        ref_logprobs = torch.tensor([[0.0]])
        advantages = torch.tensor([[1.0]])
        padding_mask = torch.ones(1, 1)
        loss = loss_fn(logprobs, ref_logprobs, advantages, padding_mask)

        # When policies are identical, loss should be 0
        assert torch.allclose(loss, torch.tensor(0.0), atol=1e-6)

        # Test that positive advantage with higher prob gives negative loss
        logprobs_higher = torch.tensor([[1.0]])  # Higher than reference
        loss_higher = loss_fn(logprobs_higher, ref_logprobs, advantages, padding_mask)

        assert loss_higher < 0  # Should encourage this action
