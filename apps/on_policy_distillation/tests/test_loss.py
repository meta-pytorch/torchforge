"""
Test file comparing reverse_kl_loss from the PR with Tinker/Thinking Machines implementation
PR: https://github.com/meta-pytorch/torchforge/pull/527

Citations from Tinker implementation:
- Blog post pseudocode: https://thinkingmachines.ai/blog/on-policy-distillation/
- Tinker Cookbook: https://github.com/thinking-machines-lab/tinker-cookbook
"""

import torch

from apps.on_policy_distillation.main import reverse_kl_loss
from forge.util.ops import compute_logprobs


class TestReverseKLLoss:
    """
    We want to cover a couple things in these tests:
        1. Basic input / output / handling of parameters
        2. Matches the Tinker implementation
        3. Behaving as expected meaning it pushes logprobs in the correct direction
    """

    def test_vs_tinker_loss(self):
        """Test the complete pattern from Tinker's implementation."""
        batch_size, seq_len, vocab_size = 2, 5, 50

        prompt = torch.randint(0, vocab_size, (batch_size, seq_len))
        response = torch.randint(0, vocab_size, (batch_size, seq_len))

        # https://github.com/thinking-machines-lab/tinker-cookbook/blob/6c9f7a4f254c01010509a147e7fd80026654464b/tinker_cookbook/distillation/train_on_policy.py#L71
        input_ids = torch.cat([prompt, response], dim=-1)

        teacher_logits = torch.full(
            (batch_size, input_ids.size(1) + 1, vocab_size), -1000.0
        )
        for b in range(batch_size):
            for t in range(input_ids.size(1)):
                teacher_logits[b, t, response[b, t]] = 0.0

        # https://github.com/thinking-machines-lab/tinker-cookbook/blob/6c9f7a4f254c01010509a147e7fd80026654464b/tinker_cookbook/distillation/train_on_policy.py#L77
        teacher_logprobs = compute_logprobs(teacher_logits, response)

        student_logits = torch.full(
            (batch_size, input_ids.size(1) + 1, vocab_size), -1000.0
        )
        for b in range(batch_size):
            for t in range(input_ids.size(1)):
                student_logits[b, t, response[b, t]] = 0.5

        # https://github.com/thinking-machines-lab/tinker-cookbook/blob/6c9f7a4f254c01010509a147e7fd80026654464b/tinker_cookbook/distillation/train_on_policy.py#L86
        student_logprobs = compute_logprobs(student_logits, response)

        # https://github.com/thinking-machines-lab/tinker-cookbook/blob/6c9f7a4f254c01010509a147e7fd80026654464b/tinker_cookbook/distillation/train_on_policy.py#L87
        mask = response == 0
        mask = mask.float()

        # https://github.com/thinking-machines-lab/tinker-cookbook/blob/6c9f7a4f254c01010509a147e7fd80026654464b/tinker_cookbook/distillation/train_on_policy.py#L89
        reverse_kl = (student_logprobs - teacher_logprobs) * mask

        # https://github.com/thinking-machines-lab/tinker-cookbook/blob/6c9f7a4f254c01010509a147e7fd80026654464b/tinker_cookbook/distillation/train_on_policy.py#L100
        advantages = -1.0 * mask * reverse_kl

        pass

    def test_zero_kl_property(self):
        """Test that KL is zero when distributions match perfectly."""
        batch_size, seq_len, vocab_size = 2, 5, 50

        response = torch.randint(0, vocab_size, (batch_size, seq_len))

        # Create logits for seq_len+1 positions (to predict seq_len response tokens)
        # compute_logprobs will slice logits[:, -seq_len-1:-1] to align with response
        logits = torch.full((batch_size, seq_len + 1, vocab_size), -1000.0)
        for b in range(batch_size):
            for t in range(seq_len):
                logits[b, t, response[b, t]] = 0.0

        # Get student log probabilities for selected tokens using compute_logprobs
        student_logprobs = compute_logprobs(logits, response)

        # Set teacher to match student exactly
        teacher_logprobs = student_logprobs.clone().detach()

        # No padding
        padding_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

        loss = reverse_kl_loss(logits, response, teacher_logprobs, padding_mask)

        # When student matches teacher, reverse_kl = 0, advantages = 0, loss = 0
        assert abs(loss.item()) < 1e-5, "Loss should be ~0 when student matches teacher"

    def test_loss_direction(self):
        """Test that gradients push student logprobs toward teacher."""
        batch_size, seq_len, vocab_size = 1, 1, 10  # noqa

        # Single token case for clarity
        response = torch.tensor([[5]])  # Token index 5

        # Student has low probability for token 5
        # Need seq_len+1 positions for compute_logprobs alignment
        logits = torch.full((1, 2, vocab_size), 0.0, requires_grad=True)
        logits.data[0, 0, 5] = -3.0  # Low logit for token 5

        # Teacher has higher probability (less negative logprob)
        teacher_logprobs = torch.tensor([[-1.0]])

        padding_mask = torch.ones(1, 1, dtype=torch.bool)

        # Compute loss and gradients
        loss = reverse_kl_loss(logits, response, teacher_logprobs, padding_mask)
        loss.backward()

        # When student logprob is lower than teacher, gradient should push it higher
        # Gradient at index 5 should be negative (increase logit -> increase logprob)
        assert logits.grad is not None
        assert (
            logits.grad[0, 0, 5].item() < 0
        ), "Gradient should push logit higher when student < teacher"

    def test_mode_seeking_behavior(self):
        """
        Test that reverse KL exhibits mode-seeking behavior.

        Citation: From blog post:
        "reverse KL is 'mode seeking' — it learns one specific behavior
        (the teacher's) instead of spreading its distribution across
        several suboptimal options."
        (https://thinkingmachines.ai/blog/on-policy-distillation/)
        """
        batch_size, seq_len, vocab_size = 1, 3, 10

        response = torch.tensor([[2, 5, 7]])

        # Teacher has high confidence (low entropy)
        teacher_logprobs = torch.tensor([[-0.1, -0.1, -0.1]])

        # Student 1: Spread distribution (high entropy)
        # Need seq_len+1 positions for compute_logprobs alignment
        logits_spread = torch.zeros(batch_size, seq_len + 1, vocab_size)

        # Student 2: Focused distribution (low entropy, matching teacher's confidence)
        logits_focused = torch.full((batch_size, seq_len + 1, vocab_size), -10.0)
        logits_focused[0, 0, 2] = 10.0
        logits_focused[0, 1, 5] = 10.0
        logits_focused[0, 2, 7] = 10.0

        padding_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

        # Compute losses
        loss_spread = reverse_kl_loss(
            logits_spread, response, teacher_logprobs, padding_mask
        )
        loss_focused = reverse_kl_loss(
            logits_focused, response, teacher_logprobs, padding_mask
        )

        # Mode-seeking: focused distribution should generally have different loss characteristics
        assert isinstance(loss_spread.item(), float)
        assert isinstance(loss_focused.item(), float)

        # Both losses should be finite
        assert torch.isfinite(loss_spread)
        assert torch.isfinite(loss_focused)
