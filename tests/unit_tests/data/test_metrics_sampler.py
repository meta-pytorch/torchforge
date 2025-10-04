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
    Reduce,
    RewardThresholdFilter,
    SampleAccumulator,
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


def test_sample_accumulator_with_topbottom_filter():
    """Ensure SampleAccumulator integrates with TopBottomKFilter correctly."""
    f = TopBottomKFilter(top_k=2, bottom_k=1, key="reward")
    acc = SampleAccumulator(Reduce.SAMPLE, filter=f)

    rewards = [0.1, 0.9, 0.5, 0.7, 0.3]
    for r in rewards:
        acc.append(make_sample(r))

    result = acc.get_value()
    result_rewards = sorted(s["reward"] for s in result)

    # Expect bottom-1 (0.1) and top-2 (0.7, 0.9)
    assert result_rewards == [0.1, 0.7, 0.9]


def test_sample_accumulator_no_filter_returns_all():
    """Ensure SampleAccumulator without a filter returns all samples."""
    acc = SampleAccumulator(Reduce.SAMPLE, filter=None)

    samples = [make_sample(r) for r in [0.2, 0.4, 0.6]]
    for s in samples:
        acc.append(s)

    result = acc.get_value()
    assert len(result) == len(samples)
    assert [s["reward"] for s in result] == [0.2, 0.4, 0.6]
