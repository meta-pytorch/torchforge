# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Expected values generated using P2120280632"""

import pytest
import torch

from forge.rl.losses import (
    aggregate,
    CISPOLoss,
    compute_entropy,
    compute_kl,
    compute_logprobs,
    compute_ratio,
    create_shifted_targets,
    CROSS_ENTROPY_IGNORE_IDX,
    DAPOLoss,
    # Loss classes
    GRPOLoss,
    GSPOLoss,
    # Primitives
    masked_mean,
    SAPOLoss,
)


def assert_close(actual, expected, atol=1e-4, rtol=1e-4):
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


def get_metric(metrics, key: str):
    for m in metrics:
        if m.key == key:
            return m.value
    raise KeyError(f"Metric '{key}' not found")


@pytest.fixture
def inputs():
    torch.manual_seed(42)
    B, S, V = 2, 4, 10

    logits = torch.randn(B, S, V)
    target_ids = torch.randint(0, V, (B, S))

    # Seq 0: mild divergence, Seq 1: high divergence (triggers clipping)
    old_logprobs = torch.tensor(
        [
            [-2.0, -2.1, -1.9, -2.0],
            [-6.0, -1.0, -5.0, -0.5],
        ]
    )
    ref_logprobs = torch.randn(B, S) * 0.5 - 2.0
    advantages = torch.randn(B, S)

    # Interleaved mask (multi-turn pattern)
    loss_mask = torch.tensor([[1, 0, 1, 0], [1, 1, 0, 0]], dtype=torch.float)

    # Pre-compute common values
    logprobs, _ = compute_logprobs(logits, target_ids)
    ratio, log_ratio, _ = compute_ratio(logprobs, old_logprobs, loss_mask)

    return {
        "B": B,
        "S": S,
        "V": V,
        "logits": logits,
        "target_ids": target_ids,
        "old_logprobs": old_logprobs,
        "ref_logprobs": ref_logprobs,
        "advantages": advantages,
        "loss_mask": loss_mask,
        "logprobs": logprobs,
        "ratio": ratio,
        "log_ratio": log_ratio,
    }


class TestPrimitives:

    def test_masked_mean(self, inputs):
        d = inputs

        # Basic: sum(values * mask) / sum(mask)
        result = masked_mean(d["advantages"], d["loss_mask"])
        assert_close(result, torch.tensor(-0.348463))

        # Zero mask: returns 0 (clamped divisor)
        result_zero = masked_mean(d["advantages"], torch.zeros_like(d["loss_mask"]))
        assert_close(result_zero, torch.tensor(0.0))

        # With loss_scale: divides by scale instead of mask.sum()
        result_scaled = masked_mean(
            d["advantages"], d["loss_mask"], loss_scale=torch.tensor(8.0)
        )
        assert_close(result_scaled, torch.tensor(-0.174231))

    def test_create_shifted_targets(self, inputs):
        input_ids = torch.tensor([[10, 20, 30, 40], [50, 60, 70, 80]])

        # Without mask
        targets = create_shifted_targets(input_ids)
        expected = torch.tensor(
            [
                [20, 30, 40, CROSS_ENTROPY_IGNORE_IDX],
                [60, 70, 80, CROSS_ENTROPY_IGNORE_IDX],
            ]
        )
        assert_close(targets, expected)

        # With mask: masked positions become ignore_index
        loss_mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]])
        targets_masked = create_shifted_targets(input_ids, loss_mask)
        expected_masked = torch.tensor(
            [
                [20, 30, CROSS_ENTROPY_IGNORE_IDX, CROSS_ENTROPY_IGNORE_IDX],
                [60, 70, 80, CROSS_ENTROPY_IGNORE_IDX],
            ]
        )
        assert_close(targets_masked, expected_masked)

    def test_compute_logprobs(self, inputs):
        d = inputs

        # Forward
        logits = d["logits"].clone().requires_grad_(True)
        logprobs, _ = compute_logprobs(logits, d["target_ids"])

        expected_logprobs = torch.tensor(
            [
                [-2.455715, -3.950112, -2.637205, -3.512223],
                [-3.542688, -2.388949, -3.638923, -4.686581],
            ]
        )
        assert_close(logprobs, expected_logprobs)

        # Backward
        loss = (logprobs * d["loss_mask"]).sum()
        loss.backward()
        assert_close(logits.grad.norm(), torch.tensor(2.077044))

    def test_compute_entropy(self, inputs):
        d = inputs

        # Forward
        logits = d["logits"].clone().requires_grad_(True)
        entropy, metrics = compute_entropy(logits, d["loss_mask"])

        expected_entropy = torch.tensor(
            [
                [1.801453, 1.862737, 2.120112, 1.875997],
                [1.429505, 2.056069, 1.953664, 1.997996],
            ]
        )
        assert_close(entropy, expected_entropy)
        assert (entropy >= 0).all()
        assert_close(get_metric(metrics, "loss/entropy/mean"), torch.tensor(1.851785))

        # Backward
        loss = masked_mean(entropy, d["loss_mask"])
        loss.backward()
        assert_close(logits.grad.norm(), torch.tensor(0.164508))

    def test_compute_ratio_token(self, inputs):
        d = inputs

        # Forward
        logprobs = d["logprobs"].clone().requires_grad_(True)
        ratio, log_ratio, _ = compute_ratio(
            logprobs, d["old_logprobs"], d["loss_mask"], ratio_type="token"
        )

        expected_ratio = torch.tensor(
            [
                [0.633994, 0.157220, 0.478449, 0.220419],
                [11.673395, 0.249337, 3.900393, 0.015198],
            ]
        )
        expected_log_ratio = torch.tensor(
            [
                [-0.455715, -1.850112, -0.737205, -1.512223],
                [2.457312, -1.388949, 1.361077, -4.186581],
            ]
        )
        assert_close(ratio, expected_ratio)
        assert_close(log_ratio, expected_log_ratio)

        # Backward
        loss = masked_mean(ratio, d["loss_mask"])
        loss.backward()
        assert_close(logprobs.grad.norm(), torch.tensor(2.925761))

    def test_compute_ratio_sequence(self, inputs):
        d = inputs

        logprobs = d["logprobs"].clone().requires_grad_(True)
        ratio, log_ratio, _ = compute_ratio(
            logprobs, d["old_logprobs"], d["loss_mask"], ratio_type="sequence"
        )

        expected_ratio = torch.tensor(
            [
                [0.550758, 0.550758, 0.550758, 0.550758],
                [1.706051, 1.706051, 1.706051, 1.706051],
            ]
        )
        expected_log_ratio = torch.tensor(
            [
                [-0.596460, -0.596460, -0.596460, -0.596460],
                [0.534181, 0.534181, 0.534181, 0.534181],
            ]
        )
        assert_close(ratio, expected_ratio)
        assert_close(log_ratio, expected_log_ratio)

        # Backward
        loss = masked_mean(ratio, d["loss_mask"])
        loss.backward()
        assert_close(logprobs.grad.norm(), torch.tensor(0.633832))

    @pytest.mark.parametrize(
        "kl_type,expected_kl,expected_mean,expected_grad_norm",
        [
            pytest.param(
                "k1",
                torch.tensor(
                    [
                        [-1.415665, -1.837418, -0.466356, -1.664230],
                        [-1.198181, 0.174410, -1.496045, -2.139825],
                    ]
                ),
                -0.726448,
                0.500000,
                id="k1",
            ),
            pytest.param(
                "k2",
                torch.tensor(
                    [
                        [1.002053, 1.688052, 0.108744, 1.384830],
                        [0.717819, 0.015209, 1.119076, 2.289426],
                    ]
                ),
                0.460956,
                0.480081,
                id="k2",
            ),
            pytest.param(
                "k3",
                torch.tensor(
                    [
                        [1.703559, 3.442883, 0.127818, 2.617373],
                        [1.115902, 0.014362, 1.967954, 5.358127],
                    ]
                ),
                0.740411,
                0.983082,
                id="k3",
            ),
        ],
    )
    def test_compute_kl(
        self, inputs, kl_type, expected_kl, expected_mean, expected_grad_norm
    ):
        d = inputs
        logprobs = d["logprobs"].clone().requires_grad_(True)
        kl, metrics = compute_kl(
            logprobs, d["ref_logprobs"], d["loss_mask"], kl_type=kl_type
        )

        assert_close(kl, expected_kl)
        assert_close(
            get_metric(metrics, "loss/kl_ref/mean"), torch.tensor(expected_mean)
        )

        # Backward
        loss = masked_mean(kl, d["loss_mask"])
        loss.backward()
        assert_close(logprobs.grad.norm(), torch.tensor(expected_grad_norm))

    @pytest.mark.parametrize(
        "agg_type,expected_loss,expected_grad_norm",
        [
            pytest.param(
                "token_mean", torch.tensor(3.258794), 0.500000, id="token_mean"
            ),
            pytest.param(
                "fixed_horizon", torch.tensor(1.629397), 0.250000, id="fixed_horizon"
            ),
            pytest.param(
                "sequence_mean", torch.tensor(3.258794), 0.500000, id="sequence_mean"
            ),
        ],
    )
    def test_aggregate(self, inputs, agg_type, expected_loss, expected_grad_norm):
        d = inputs
        per_token_loss = d["ratio"].clone().requires_grad_(True)
        loss, metrics = aggregate(per_token_loss, d["loss_mask"], agg_type=agg_type)

        assert_close(loss, expected_loss)
        assert_close(
            get_metric(metrics, "loss/aggregate/active_fraction"), torch.tensor(0.5)
        )

        # Backward
        loss.backward()
        assert_close(per_token_loss.grad.norm(), torch.tensor(expected_grad_norm))


class TestLosses:

    def test_grpo(self, inputs):
        d = inputs
        logits = d["logits"].clone().requires_grad_(True)

        loss_fn = GRPOLoss(clip_low=0.2, clip_high=0.2, beta=0.1)
        output = loss_fn(
            logits=logits,
            target_ids=d["target_ids"],
            advantages=d["advantages"],
            old_logprobs=d["old_logprobs"],
            loss_mask=d["loss_mask"],
            ref_logprobs=d["ref_logprobs"],
        )

        # Forward
        assert_close(output.loss, torch.tensor(0.260804))

        # Backward
        output.loss.backward()
        assert_close(logits.grad.norm(), torch.tensor(0.137857))

    def test_dapo(self, inputs):
        d = inputs
        logits = d["logits"].clone().requires_grad_(True)

        loss_fn = DAPOLoss(clip_low=0.2, clip_high=0.28, dual_clip_c=3.0)
        output = loss_fn(
            logits=logits,
            target_ids=d["target_ids"],
            advantages=d["advantages"],
            old_logprobs=d["old_logprobs"],
            loss_mask=d["loss_mask"],
        )

        # Forward
        assert_close(output.loss, torch.tensor(0.445464))

        # Backward
        output.loss.backward()
        assert_close(logits.grad.norm(), torch.tensor(0.191675))

    def test_gspo(self, inputs):
        d = inputs
        logits = d["logits"].clone().requires_grad_(True)

        loss_fn = GSPOLoss(clip_low=1.0, clip_high=0.2)
        output = loss_fn(
            logits=logits,
            target_ids=d["target_ids"],
            advantages=d["advantages"],
            old_logprobs=d["old_logprobs"],
            loss_mask=d["loss_mask"],
        )

        # Forward
        assert_close(output.loss, torch.tensor(0.018975))

        # Backward
        output.loss.backward()
        assert_close(logits.grad.norm(), torch.tensor(0.517415))

    def test_cispo(self, inputs):
        d = inputs
        logits = d["logits"].clone().requires_grad_(True)

        loss_fn = CISPOLoss(clip_low=1.0, clip_high=4.0)
        output = loss_fn(
            logits=logits,
            target_ids=d["target_ids"],
            advantages=d["advantages"],
            old_logprobs=d["old_logprobs"],
            loss_mask=d["loss_mask"],
        )

        # Forward
        assert_close(output.loss, torch.tensor(-0.083283))

        # Backward
        output.loss.backward()
        assert_close(logits.grad.norm(), torch.tensor(0.492673))

    def test_sapo(self, inputs):
        d = inputs
        logits = d["logits"].clone().requires_grad_(True)

        loss_fn = SAPOLoss(tau_pos=1.0, tau_neg=1.05)
        output = loss_fn(
            logits=logits,
            target_ids=d["target_ids"],
            advantages=d["advantages"],
            old_logprobs=d["old_logprobs"],
            loss_mask=d["loss_mask"],
        )

        # Forward
        assert_close(output.loss, torch.tensor(0.376388))

        # Backward
        output.loss.backward()
        assert_close(logits.grad.norm(), torch.tensor(0.437776))

    @pytest.mark.parametrize(
        "loss_cls,kwargs",
        [
            pytest.param(GRPOLoss, {"beta": 0.0}, id="GRPO"),
            pytest.param(DAPOLoss, {}, id="DAPO"),
            pytest.param(GSPOLoss, {}, id="GSPO"),
            pytest.param(CISPOLoss, {}, id="CISPO"),
            pytest.param(SAPOLoss, {}, id="SAPO"),
        ],
    )
    def test_zero_advantages(self, inputs, loss_cls, kwargs):
        d = inputs
        advantages = torch.zeros_like(d["advantages"])

        loss_fn = loss_cls(**kwargs)
        output = loss_fn(
            logits=d["logits"],
            target_ids=d["target_ids"],
            advantages=advantages,
            old_logprobs=d["old_logprobs"],
            loss_mask=d["loss_mask"],
        )

        assert output.loss.isfinite()

    @pytest.mark.parametrize(
        "loss_cls,kwargs",
        [
            pytest.param(GRPOLoss, {"beta": 0.0}, id="GRPO"),
            pytest.param(DAPOLoss, {}, id="DAPO"),
            pytest.param(GSPOLoss, {}, id="GSPO"),
            pytest.param(CISPOLoss, {}, id="CISPO"),
            pytest.param(SAPOLoss, {}, id="SAPO"),
        ],
    )
    def test_empty_mask(self, inputs, loss_cls, kwargs):
        """Loss should be finite (zero) when mask is all zeros (no trainable tokens)."""
        d = inputs
        empty_mask = torch.zeros_like(d["loss_mask"])

        loss_fn = loss_cls(**kwargs)
        output = loss_fn(
            logits=d["logits"],
            target_ids=d["target_ids"],
            advantages=d["advantages"],
            old_logprobs=d["old_logprobs"],
            loss_mask=empty_mask,
        )

        assert output.loss.isfinite()
        assert output.loss == 0.0
