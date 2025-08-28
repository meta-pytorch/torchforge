# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Test for actors/replay_buffer.py"""

import threading
from random import Random

import pytest
import pytest_asyncio

from forge.actors.replay_buffer import ReplayBuffer
from forge.data.stateful_sampler import RandomStatefulSampler
from forge.interfaces import StatefulSampler

from forge.test_util.udp_trace import add_udp_callback, receive_udp_packet
from forge.types import State, Trajectory

from monarch.actor import proc_mesh


class TestReplayBuffer:
    @pytest_asyncio.fixture
    async def replay_buffer(self) -> ReplayBuffer:
        mesh = await proc_mesh(gpus=1)
        replay_buffer = await mesh.spawn(
            "replay_buffer", ReplayBuffer, batch_size=2, max_policy_age=1
        )
        await replay_buffer.setup.call()
        return replay_buffer

    @pytest.mark.asyncio
    async def test_setup_accepts_sampler(self) -> None:
        # This test is flaky and only works if it is run on a single machine.
        # However, it's impossible to directly mock a function called by monarch
        # because it is first pickled and then unpickled.

        sampler = RandomStatefulSampler()
        TEST_PORT = 34958
        sampler.sample_keys = add_udp_callback(
            sampler.sample_keys, port=TEST_PORT, message=b"sample_keys"
        )

        mesh = await proc_mesh(gpus=1)
        replay_buffer = await mesh.spawn(
            "replay_buffer", ReplayBuffer, batch_size=1, max_policy_age=1
        )
        received = []
        server_thread = threading.Thread(
            target=receive_udp_packet,
            args=(TEST_PORT, received),
            kwargs={"timeout": 15},
        )
        server_thread.start()
        await replay_buffer.setup.call(sampler=sampler)
        await replay_buffer.add.call_one(Trajectory(policy_version=0))
        await replay_buffer.sample.call_one(curr_policy_version=0)
        server_thread.join()
        assert b"".join(received) == b"sample_keys"

    @pytest.mark.asyncio
    async def test_add(self, replay_buffer: ReplayBuffer) -> None:
        trajectory = Trajectory(policy_version=0)
        await replay_buffer.add.call_one(trajectory)
        assert replay_buffer._numel.call_one().get() == 1
        assert replay_buffer.sample.call_one(
            curr_policy_version=0, batch_size=1
        ).get() == [trajectory]
        replay_buffer.clear.call_one().get()

    @pytest.mark.asyncio
    async def test_add_multiple(self, replay_buffer) -> None:
        trajectory_0 = Trajectory(policy_version=0)
        trajectory_1 = Trajectory(policy_version=1)
        await replay_buffer.add.call_one(trajectory_0)
        await replay_buffer.add.call_one(trajectory_1)
        assert replay_buffer._numel.call_one().get() == 2
        all_samples = replay_buffer.sample.call_one(
            curr_policy_version=0, batch_size=2
        ).get()
        assert all_samples == [trajectory_0, trajectory_1] or all_samples == [
            trajectory_1,
            trajectory_0,
        ]
        replay_buffer.clear.call_one().get()

    @pytest.mark.asyncio
    async def test_state_dict_save_load(self, replay_buffer) -> None:
        trajectory = Trajectory(policy_version=0)
        await replay_buffer.add.call_one(trajectory)
        state_dict = replay_buffer.state_dict.call_one().get()
        replay_buffer.clear.call_one().get()
        assert replay_buffer._numel.call_one().get() == 0
        await replay_buffer.load_state_dict.call_one(state_dict)
        assert replay_buffer._numel.call_one().get() == 1
        replay_buffer.clear.call_one().get()

    @pytest.mark.asyncio
    async def test_evict(self, replay_buffer) -> None:
        trajectory_0 = Trajectory(policy_version=0)
        trajectory_1 = Trajectory(policy_version=1)
        await replay_buffer.add.call_one(trajectory_0)
        await replay_buffer.add.call_one(trajectory_1)
        assert replay_buffer._numel.call_one().get() == 2
        await replay_buffer.evict.call_one(curr_policy_version=2)
        assert replay_buffer._numel.call_one().get() == 1
        replay_buffer.clear.call_one().get()

    @pytest.mark.asyncio
    async def test_sample(self, replay_buffer) -> None:
        trajectory_0 = Trajectory(policy_version=0)
        trajectory_1 = Trajectory(policy_version=1)
        await replay_buffer.add.call_one(trajectory_0)
        await replay_buffer.add.call_one(trajectory_1)
        assert replay_buffer._numel.call_one().get() == 2

        # Test a simple sampling w/ no evictions
        samples = await replay_buffer.sample.call_one(curr_policy_version=1)
        assert samples is not None
        assert len(samples) == 2

        # Test sampling with overriding batch size
        await replay_buffer.add.call_one(trajectory_0)
        samples = await replay_buffer.sample.call_one(
            curr_policy_version=1, batch_size=1
        )
        assert samples is not None
        assert len(samples) == 1

        # Test sampling w/ overriding batch size (not enough samples in buffer, returns None)
        await replay_buffer.add.call_one(trajectory_0)
        samples = await replay_buffer.sample.call_one(
            curr_policy_version=1, batch_size=3
        )
        assert samples is None
        replay_buffer.clear.call_one().get()

    @pytest.mark.asyncio
    async def test_sample_with_evictions(self, replay_buffer) -> None:
        trajectory_0 = Trajectory(policy_version=0)
        trajectory_1 = Trajectory(policy_version=1)
        await replay_buffer.add.call_one(trajectory_0)
        await replay_buffer.add.call_one(trajectory_1)
        assert replay_buffer._numel.call_one().get() == 2
        samples = await replay_buffer.sample.call_one(
            curr_policy_version=2, batch_size=1
        )
        assert samples is not None
        assert len(samples) == 1
        assert samples[0] == trajectory_1
        replay_buffer.clear.call_one().get()
