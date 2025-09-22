# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch

from apps.grpo.algorithms import compute_advantages


class TestComputeAdvantages(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_empty_rewards_list(self):
        advantages = compute_advantages([])
        self.assertEqual(advantages, [])

    def test_identical_rewards(self):
        rewards = [3.0, 3.0, 3.0, 3.0]
        advantages = compute_advantages(rewards)

        # All rewards are identical, so std=0 and all advantages should be ~0
        self.assertEqual(len(advantages), 4)
        for advantage in advantages:
            self.assertAlmostEqual(advantage, 0.0, places=3)

    def test_negative_rewards(self):
        rewards = [-2.0, -1.0, 0.0, 1.0, 2.0]
        advantages = compute_advantages(rewards)

        self.assertEqual(len(advantages), 5)
        # Mean = 0.0, std = sqrt(2) ≈ 1.5811
        mean = 0.0
        std = torch.tensor(rewards).std().item()

        expected_advantages = [(r - mean) / std for r in rewards]
        for i, expected in enumerate(expected_advantages):
            self.assertAlmostEqual(advantages[i], expected, places=3)

    def test_large_rewards_list(self):
        rewards = [i * 0.5 for i in range(10)]  # [0.0, 0.5, 1.0, ..., 4.5]
        advantages = compute_advantages(rewards)

        self.assertEqual(len(advantages), 10)

        # Verify normalization properties
        advantages_tensor = torch.tensor(advantages)
        mean_advantage = advantages_tensor.mean()
        std_advantage = advantages_tensor.std()

        # The normalized advantages should have mean ≈ 0 and std ≈ 1
        self.assertAlmostEqual(mean_advantage.item(), 0.0, places=3)
        self.assertAlmostEqual(std_advantage.item(), 1.0, places=3)


if __name__ == "__main__":
    unittest.main()
