# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from forge.rl.loss import DAPOLoss

from .conftest import assert_close


class TestDAPOLoss:

    def test_forward(self, inputs):
        d = inputs
        logits = d["logits"].clone().requires_grad_(True)

        loss_fn = DAPOLoss(clip_low=0.2, clip_high=0.28, dual_clip_c=3.0)
        output = loss_fn(
            logits=logits,
            target_ids=d["target_ids"],
            advantages=d["advantages"],
            generator_logprobs=d["generator_logprobs"],
            loss_mask=d["loss_mask"],
        )

        assert_close(output.loss, torch.tensor(0.445464))

    def test_backward(self, inputs):
        d = inputs
        logits = d["logits"].clone().requires_grad_(True)

        loss_fn = DAPOLoss(clip_low=0.2, clip_high=0.28, dual_clip_c=3.0)
        output = loss_fn(
            logits=logits,
            target_ids=d["target_ids"],
            advantages=d["advantages"],
            generator_logprobs=d["generator_logprobs"],
            loss_mask=d["loss_mask"],
        )

        output.loss.backward()
        assert_close(logits.grad.norm(), torch.tensor(0.191675))
