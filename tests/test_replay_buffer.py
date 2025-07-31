"""Test for data/replay_buffer.py"""

import re

from src.forge.data.replay_buffer import ReplayBuffer
from src.forge.interfaces import Trajectory


class TestReplayBuffer:

    async def test_add(self) -> None:
        replay_buffer = ReplayBuffer(batch_size=2, max_policy_age=1)
        trajectory = Trajectory(policy_version=0)
        await replay_buffer.add(trajectory)
        assert len(replay_buffer) == 1
        assert replay_buffer[0] == trajectory

    async def test_add_multiple(self) -> None:
        replay_buffer = ReplayBuffer(batch_size=2, max_policy_age=1)
        trajectory_0 = Trajectory(policy_version=0)
        trajectory_1 = Trajectory(policy_version=1)
        await replay_buffer.add(trajectory_0)
        await replay_buffer.add(trajectory_1)
        assert len(replay_buffer) == 2
        assert replay_buffer[0] == trajectory_0
        assert replay_buffer[1] == trajectory_1

    async def test_clear(self) -> None:
        replay_buffer = ReplayBuffer(batch_size=2, max_policy_age=1)
        trajectory = Trajectory(policy_version=0)
        await replay_buffer.add(trajectory)
        assert len(replay_buffer) == 1
        replay_buffer.clear()
        assert len(replay_buffer) == 0

    async def test_state_dict_save_load(self) -> None:
        replay_buffer = ReplayBuffer(batch_size=2, max_policy_age=1)
        trajectory = Trajectory(policy_version=0)
        await replay_buffer.add(trajectory)
        state_dict = replay_buffer.state_dict()
        replay_buffer.clear()
        assert len(replay_buffer) == 0
        replay_buffer.load_state_dict(state_dict)
        assert len(replay_buffer) == 1

    async def test_evict(self) -> None:
        replay_buffer = ReplayBuffer(batch_size=2, max_policy_age=1)
        trajectory_0 = Trajectory(policy_version=0)
        trajectory_1 = Trajectory(policy_version=1)
        await replay_buffer.add(trajectory_0)
        await replay_buffer.add(trajectory_1)
        assert len(replay_buffer) == 2
        await replay_buffer.evict(curr_policy_version=1)
        assert len(replay_buffer) == 1

    async def test_sample(self) -> None:
        replay_buffer = ReplayBuffer(batch_size=2, max_policy_age=2)
        trajectory_0 = Trajectory(policy_version=0)
        trajectory_1 = Trajectory(policy_version=1)
        await replay_buffer.add(trajectory_0)
        await replay_buffer.add(trajectory_1)
        assert len(replay_buffer) == 2
        samples = await replay_buffer.sample(curr_policy_version=1)
        assert samples is not None
        assert len(samples) == 2

    async def test_sample_with_evictions(self) -> None:
        replay_buffer = ReplayBuffer(batch_size=2, max_policy_age=1)
        trajectory_0 = Trajectory(policy_version=0)
        trajectory_1 = Trajectory(policy_version=1)
        await replay_buffer.add(trajectory_0)
        await replay_buffer.add(trajectory_1)
        assert len(replay_buffer) == 2
        samples = await replay_buffer.sample(curr_policy_version=1)
        assert samples is not None
        assert len(samples) == 1
        assert samples[0] == trajectory_1
