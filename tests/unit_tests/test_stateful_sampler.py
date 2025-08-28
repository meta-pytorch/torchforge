# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
from forge.data.raw_buffer import SimpleRawBuffer
from forge.data.stateful_sampler import RandomStatefulSampler

from forge.interfaces import RawBuffer


class TestRandomStatefulSampler:
    @pytest.fixture
    def raw_buffer(self) -> RawBuffer[int, int]:
        buffer = SimpleRawBuffer[int, int]()
        for n in range(1000):
            buffer.add(n, n)
        return buffer

    def test_init(self):
        sampler = RandomStatefulSampler()
        assert True

    def test_init_with_seed(self):
        sampler1 = RandomStatefulSampler(seed=42)
        sampler2 = RandomStatefulSampler(seed=41)
        assert str(sampler1.state_dict()) != str(sampler2.state_dict())

    def test_state_dict(self):
        sampler = RandomStatefulSampler()
        state_dict = sampler.state_dict()
        assert "random_state" in state_dict
        assert state_dict["random_state"] is not None

    def test_set_state_dict_no_random_state(self):
        sampler = RandomStatefulSampler()
        state_dict = {}
        with pytest.raises(ValueError, match="Missing 'random_state'"):
            sampler.set_state_dict(state_dict)

    def test_deterministic(self, raw_buffer):
        sampler1 = RandomStatefulSampler(seed=42)
        sampler2 = RandomStatefulSampler()
        sampler2.set_state_dict(sampler1.state_dict())
        for _ in range(10):
            batch1 = sampler1.sample_keys(raw_buffer, 5)
            batch2 = sampler2.sample_keys(raw_buffer, 5)
            assert batch1 == batch2

    def test_deterministic_resume(self, raw_buffer):
        sampler1 = RandomStatefulSampler(seed=42)
        sampler2 = RandomStatefulSampler()
        for _ in range(10):
            sampler2.set_state_dict(sampler1.state_dict())
            batch1 = sampler1.sample_keys(raw_buffer, 5)
            batch2 = sampler2.sample_keys(raw_buffer, 5)
            assert batch1 == batch2
