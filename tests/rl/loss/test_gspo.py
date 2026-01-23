# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from forge.rl.loss import GSPOLoss

from .conftest import assert_close


class TestGSPOLoss:

    def test_forward(self, inputs):
        d = inputs
        logits = d["logits"].clone().requires_grad_(True)

        loss_fn = GSPOLoss(clip_low=0.2, clip_high=0.2)
        output = loss_fn(
            logits=logits,
            target_ids=d["target_ids"],
            advantages=d["advantages"],
            generator_logprobs=d["generator_logprobs"],
            loss_mask=d["loss_mask"],
        )

        assert_close(output.loss, torch.tensor(0.242948))

    def test_backward(self, inputs):
        d = inputs
        logits = d["logits"].clone().requires_grad_(True)

        loss_fn = GSPOLoss(clip_low=0.2, clip_high=0.2)
        output = loss_fn(
            logits=logits,
            target_ids=d["target_ids"],
            advantages=d["advantages"],
            generator_logprobs=d["generator_logprobs"],
            loss_mask=d["loss_mask"],
        )

        output.loss.backward()
        assert_close(logits.grad.norm(), torch.tensor(0.158423))

    def test_zero_advantages(self, inputs):
        d = inputs
        advantages = torch.zeros_like(d["advantages"])

        loss_fn = GSPOLoss()
        output = loss_fn(
            logits=d["logits"],
            target_ids=d["target_ids"],
            advantages=advantages,
            generator_logprobs=d["generator_logprobs"],
            loss_mask=d["loss_mask"],
        )

        assert output.loss.isfinite()

    def test_empty_mask(self, inputs):
        """Loss should be finite (zero) when mask is all zeros (no trainable tokens)."""
        d = inputs
        empty_mask = torch.zeros_like(d["loss_mask"])

        loss_fn = GSPOLoss()
        output = loss_fn(
            logits=d["logits"],
            target_ids=d["target_ids"],
            advantages=d["advantages"],
            generator_logprobs=d["generator_logprobs"],
            loss_mask=empty_mask,
        )

        assert output.loss.isfinite()
        assert output.loss == 0.0

    def test_empty_sequence(self):
        """Loss should be zero when sequence length is 0."""
        B, V = 2, 10
        logits = torch.empty(B, 0, V)
        target_ids = torch.empty(B, 0, dtype=torch.long)
        advantages = torch.empty(B, 0)
        generator_logprobs = torch.empty(B, 0)
        loss_mask = torch.empty(B, 0)

        loss_fn = GSPOLoss()
        output = loss_fn(
            logits=logits,
            target_ids=target_ids,
            advantages=advantages,
            generator_logprobs=generator_logprobs,
            loss_mask=loss_mask,
        )

        assert output.loss.isfinite()
        assert output.loss == 0.0
