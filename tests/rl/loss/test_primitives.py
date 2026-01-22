# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from forge.rl.loss import (
    aggregate,
    compute_entropy,
    compute_kl,
    compute_logprobs,
    compute_ratio,
    create_shifted_targets,
    CROSS_ENTROPY_IGNORE_IDX,
    masked_mean,
)

from .conftest import assert_close, get_metric


class TestMaskedMean:

    def test_basic(self, inputs):
        d = inputs
        result = masked_mean(d["advantages"], d["loss_mask"])
        assert_close(result, torch.tensor(-0.348463))

    def test_zero_mask(self, inputs):
        d = inputs
        result = masked_mean(d["advantages"], torch.zeros_like(d["loss_mask"]))
        assert_close(result, torch.tensor(0.0))

    def test_with_loss_scale(self, inputs):
        d = inputs
        result = masked_mean(
            d["advantages"], d["loss_mask"], loss_scale=torch.tensor(8.0)
        )
        assert_close(result, torch.tensor(-0.174231))


class TestCreateShiftedTargets:

    def test_without_mask(self):
        input_ids = torch.tensor([[10, 20, 30, 40], [50, 60, 70, 80]])
        targets = create_shifted_targets(input_ids)
        expected = torch.tensor(
            [
                [20, 30, 40, CROSS_ENTROPY_IGNORE_IDX],
                [60, 70, 80, CROSS_ENTROPY_IGNORE_IDX],
            ]
        )
        assert_close(targets, expected)

    def test_with_mask(self):
        input_ids = torch.tensor([[10, 20, 30, 40], [50, 60, 70, 80]])
        loss_mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]])
        targets = create_shifted_targets(input_ids, loss_mask)
        expected = torch.tensor(
            [
                [20, 30, CROSS_ENTROPY_IGNORE_IDX, CROSS_ENTROPY_IGNORE_IDX],
                [60, 70, 80, CROSS_ENTROPY_IGNORE_IDX],
            ]
        )
        assert_close(targets, expected)


class TestComputeLogprobs:

    def test_forward(self, inputs):
        d = inputs
        logits = d["logits"].clone().requires_grad_(True)
        logprobs, _ = compute_logprobs(logits, d["target_ids"])

        expected = torch.tensor(
            [
                [-2.455715, -3.950112, -2.637205, -3.512223],
                [-3.542688, -2.388949, -3.638923, -4.686581],
            ]
        )
        assert_close(logprobs, expected)

    def test_backward(self, inputs):
        d = inputs
        logits = d["logits"].clone().requires_grad_(True)
        logprobs, _ = compute_logprobs(logits, d["target_ids"])
        loss = (logprobs * d["loss_mask"]).sum()
        loss.backward()
        assert_close(logits.grad.norm(), torch.tensor(2.077044))


class TestComputeEntropy:

    def test_forward(self, inputs):
        d = inputs
        logits = d["logits"].clone().requires_grad_(True)
        entropy, metrics = compute_entropy(logits, d["loss_mask"])

        expected = torch.tensor(
            [
                [1.801453, 1.862737, 2.120112, 1.875997],
                [1.429505, 2.056069, 1.953664, 1.997996],
            ]
        )
        assert_close(entropy, expected)
        assert (entropy >= 0).all()
        assert_close(get_metric(metrics, "loss/entropy/mean"), torch.tensor(1.851785))

    def test_backward(self, inputs):
        d = inputs
        logits = d["logits"].clone().requires_grad_(True)
        entropy, _ = compute_entropy(logits, d["loss_mask"])
        loss = masked_mean(entropy, d["loss_mask"])
        loss.backward()
        assert_close(logits.grad.norm(), torch.tensor(0.164508))


class TestComputeRatio:

    def test_token_forward(self, inputs):
        d = inputs
        logprobs = d["logprobs"].clone().requires_grad_(True)
        ratio, log_ratio, _ = compute_ratio(
            logprobs, d["generator_logprobs"], d["loss_mask"], ratio_type="token"
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

    def test_token_backward(self, inputs):
        d = inputs
        logprobs = d["logprobs"].clone().requires_grad_(True)
        ratio, _, _ = compute_ratio(
            logprobs, d["generator_logprobs"], d["loss_mask"], ratio_type="token"
        )
        loss = masked_mean(ratio, d["loss_mask"])
        loss.backward()
        assert_close(logprobs.grad.norm(), torch.tensor(2.925761))

    def test_sequence_forward(self, inputs):
        d = inputs
        logprobs = d["logprobs"].clone().requires_grad_(True)
        ratio, log_ratio, _ = compute_ratio(
            logprobs, d["generator_logprobs"], d["loss_mask"], ratio_type="sequence"
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

    def test_sequence_backward(self, inputs):
        d = inputs
        logprobs = d["logprobs"].clone().requires_grad_(True)
        ratio, _, _ = compute_ratio(
            logprobs, d["generator_logprobs"], d["loss_mask"], ratio_type="sequence"
        )
        loss = masked_mean(ratio, d["loss_mask"])
        loss.backward()
        assert_close(logprobs.grad.norm(), torch.tensor(0.633832))


class TestComputeKl:

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
    def test_kl_types(
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

        loss = masked_mean(kl, d["loss_mask"])
        loss.backward()
        assert_close(logprobs.grad.norm(), torch.tensor(expected_grad_norm))


class TestAggregate:

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
    def test_agg_types(self, inputs, agg_type, expected_loss, expected_grad_norm):
        d = inputs
        per_token_loss = d["ratio"].clone().requires_grad_(True)
        loss, metrics = aggregate(per_token_loss, d["loss_mask"], agg_type=agg_type)

        assert_close(loss, expected_loss)
        assert_close(
            get_metric(metrics, "loss/aggregate/active_fraction"), torch.tensor(0.5)
        )

        loss.backward()
        assert_close(per_token_loss.grad.norm(), torch.tensor(expected_grad_norm))
