# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


"""
Tests for sample filters:
- RandomRatioFilter: checks probabilistic acceptance of samples.
- RewardThresholdFilter: checks filtering by reward thresholds.
- TopBottomKFilter: checks top-k and bottom-k heap behavior.
"""

import random
from typing import Dict

from forge.observability.metrics import (
    RandomRatioFilter,
    RewardThresholdFilter,
    TopBottomKFilter,
)


def make_sample(reward: float) -> Dict:
    return {"reward": reward, "prompt": "Q", "response": "A"}


def test_random_ratio_filter_deterministic(monkeypatch):
    # Force randomness to be deterministic
    monkeypatch.setattr(random, "random", lambda: 0.1)

    f = RandomRatioFilter(ratio=0.2)
    # 0.1 < 0.2, so should accept
    assert f.filter_append(make_sample(0.5)) is True

    monkeypatch.setattr(random, "random", lambda: 0.5)
    # 0.5 > 0.2, so should reject
    assert f.filter_append(make_sample(0.5)) is False


def test_reward_threshold_filter_lt():
    f = RewardThresholdFilter(lt=0.5)
    assert f.filter_append(make_sample(0.2)) is True  # keep < 0.5
    assert f.filter_append(make_sample(0.5)) is False  # drop >= 0.5
    assert f.filter_append(make_sample(0.8)) is False


def test_reward_threshold_filter_gt():
    f = RewardThresholdFilter(gt=0.5)
    assert f.filter_append(make_sample(0.8)) is True  # keep > 0.5
    assert f.filter_append(make_sample(0.5)) is False  # drop <= 0.5
    assert f.filter_append(make_sample(0.2)) is False


def test_top_bottom_k_filter_simple():
    f = TopBottomKFilter(top_k=2, bottom_k=2, key="reward")
    rewards = [0.1, 0.9, 0.5, 0.7, 0.3]
    for r in rewards:
        f.filter_append(make_sample(r))

    samples = f.filter_flush([])
    values = sorted(s["reward"] for s in samples)

    # Expect bottom-2 (0.1, 0.3) and top-2 (0.7, 0.9)
    assert values == [0.1, 0.3, 0.7, 0.9]
