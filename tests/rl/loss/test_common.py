# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from forge.rl.loss import CISPOLoss, DAPOLoss, GRPOLoss, GSPOLoss, SAPOLoss


ALL_LOSSES = [
    pytest.param(GRPOLoss, {"beta": 0.0}, id="GRPO"),
    pytest.param(DAPOLoss, {}, id="DAPO"),
    pytest.param(GSPOLoss, {}, id="GSPO"),
    pytest.param(CISPOLoss, {}, id="CISPO"),
    pytest.param(SAPOLoss, {}, id="SAPO"),
]


class TestCommonBehavior:

    @pytest.mark.parametrize("loss_cls,kwargs", ALL_LOSSES)
    def test_zero_advantages(self, inputs, loss_cls, kwargs):
        d = inputs
        advantages = torch.zeros_like(d["advantages"])

        loss_fn = loss_cls(**kwargs)
        output = loss_fn(
            logits=d["logits"],
            target_ids=d["target_ids"],
            advantages=advantages,
            generator_logprobs=d["generator_logprobs"],
            loss_mask=d["loss_mask"],
        )

        assert output.loss.isfinite()

    @pytest.mark.parametrize("loss_cls,kwargs", ALL_LOSSES)
    def test_empty_mask(self, inputs, loss_cls, kwargs):
        """Loss should be finite (zero) when mask is all zeros (no trainable tokens)."""
        d = inputs
        empty_mask = torch.zeros_like(d["loss_mask"])

        loss_fn = loss_cls(**kwargs)
        output = loss_fn(
            logits=d["logits"],
            target_ids=d["target_ids"],
            advantages=d["advantages"],
            generator_logprobs=d["generator_logprobs"],
            loss_mask=empty_mask,
        )

        assert output.loss.isfinite()
        assert output.loss == 0.0

    @pytest.mark.parametrize("loss_cls,kwargs", ALL_LOSSES)
    def test_empty_sequence(self, loss_cls, kwargs):
        """Loss should be zero when sequence length is 0."""
        B, V = 2, 10
        logits = torch.empty(B, 0, V)
        target_ids = torch.empty(B, 0, dtype=torch.long)
        advantages = torch.empty(B, 0)
        generator_logprobs = torch.empty(B, 0)
        loss_mask = torch.empty(B, 0)

        loss_fn = loss_cls(**kwargs)
        output = loss_fn(
            logits=logits,
            target_ids=target_ids,
            advantages=advantages,
            generator_logprobs=generator_logprobs,
            loss_mask=loss_mask,
        )

        assert output.loss.isfinite()
        assert output.loss == 0.0
