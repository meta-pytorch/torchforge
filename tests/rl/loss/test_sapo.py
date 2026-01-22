# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from forge.rl.loss import SAPOLoss

from .conftest import assert_close


class TestSAPOLoss:

    def test_forward(self, inputs):
        d = inputs
        logits = d["logits"].clone().requires_grad_(True)

        loss_fn = SAPOLoss(tau_pos=1.0, tau_neg=1.05)
        output = loss_fn(
            logits=logits,
            target_ids=d["target_ids"],
            advantages=d["advantages"],
            generator_logprobs=d["generator_logprobs"],
            loss_mask=d["loss_mask"],
        )

        assert_close(output.loss, torch.tensor(0.376388))

    def test_backward(self, inputs):
        d = inputs
        logits = d["logits"].clone().requires_grad_(True)

        loss_fn = SAPOLoss(tau_pos=1.0, tau_neg=1.05)
        output = loss_fn(
            logits=logits,
            target_ids=d["target_ids"],
            advantages=d["advantages"],
            generator_logprobs=d["generator_logprobs"],
            loss_mask=d["loss_mask"],
        )

        output.loss.backward()
        assert_close(logits.grad.norm(), torch.tensor(0.437776))
