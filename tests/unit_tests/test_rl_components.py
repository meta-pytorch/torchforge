# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for RL components."""

import asyncio
from functools import partial

import pytest
import torch
from forge.data.environments import (
    ToyAction,
    ToyEnvironment,
    ToyObservation,
    ToyRewarder,
    ToyState,
)
from forge.data.policies import ToyPolicy
from forge.rl.collector import Collector
from forge.rl.interfaces import Trajectory
from forge.rl.replay_buffer import ReplayBuffer

# local_proc_mesh is an implementation of proc_mesh for
# testing purposes. It lacks some features of the real proc_mesh
# but spawns much quicker
from monarch.actor import Actor, endpoint, local_proc_mesh


class TestToyEnvironment:
    @pytest.mark.asyncio
    async def test_toy_environment(self):
        proc = await local_proc_mesh(gpus=1)
        rewarder = await proc.spawn("rewarder", ToyRewarder)
        env = ToyEnvironment(name="test_env", rewarder=rewarder, max_steps=10)

        observation = env.reset()
        assert isinstance(observation, ToyObservation)
        assert observation.step == 0
        assert torch.equal(observation.data, torch.tensor([0.0]))

        action = ToyAction(data=torch.tensor([2.5]))
        observation, reward = env.step(action)

        assert isinstance(observation, ToyObservation)
        assert observation.step == 1
        assert torch.equal(observation.data, torch.tensor([2.5]))

        # ToyRewarder should return next_state_value + 1.0
        expected_reward = 2.5 + 1.0
        assert reward == expected_reward

    @pytest.mark.asyncio
    async def test_toy_environment_multiple_steps(self):
        """Test the toy environment for multiple steps."""
        rewarder_procs = await local_proc_mesh(gpus=1)
        rewarder = await rewarder_procs.spawn("rewarder", ToyRewarder)
        env = ToyEnvironment(name="test_env", rewarder=rewarder, max_steps=10)

        env.reset()
        action1 = ToyAction(data=torch.tensor([1.0]))
        action2 = ToyAction(data=torch.tensor([2.0]))
        action3 = ToyAction(data=torch.tensor([-0.5]))

        obs1, reward1 = env.step(action1)
        obs2, reward2 = env.step(action2)
        obs3, reward3 = env.step(action3)

        # Check observations
        assert obs1.step == 1 and torch.equal(obs1.data, torch.tensor([1.0]))
        assert obs2.step == 2 and torch.equal(obs2.data, torch.tensor([3.0]))
        assert obs3.step == 3 and torch.equal(obs3.data, torch.tensor([2.5]))

        # Check rewards (next_state_value + 1.0)
        assert reward1 == 2.0  # 1.0 + 1.0
        assert reward2 == 4.0  # 3.0 + 1.0
        assert reward3 == 3.5  # 2.5 + 1.0


class TestToyRewarder:
    """Test the ToyRewarder component."""

    @pytest.mark.asyncio
    async def test_toy_rewarder_compute_reward(self):
        """Test that ToyRewarder computes rewards correctly."""
        proc = await local_proc_mesh(gpus=1)
        rewarder = await proc.spawn("rewarder", ToyRewarder)

        state = ToyState(step=0, data=torch.tensor([1.0]))
        action = ToyAction(data=torch.tensor([0.5]))
        next_state = ToyState(step=1, data=torch.tensor([1.5]))

        reward = await rewarder.compute_reward.choose(state, action, next_state)

        # ToyRewarder should return next_state_value + 1.0
        expected_reward = 1.5 + 1.0
        assert reward == expected_reward

    @pytest.mark.asyncio
    async def test_toy_rewarder_with_zero_state(self):
        """Test ToyRewarder with zero state value."""
        proc = await local_proc_mesh(gpus=1)
        rewarder = await proc.spawn("rewarder", ToyRewarder)

        state = ToyState(step=0, data=torch.tensor([0.0]))
        action = ToyAction(data=torch.tensor([0.0]))
        next_state = ToyState(step=1, data=torch.tensor([0.0]))

        reward = await rewarder.compute_reward.choose(state, action, next_state)

        expected_reward = 0.0 + 1.0
        assert reward == expected_reward


class TestToyPolicy:
    """Test the ToyPolicy component."""

    @pytest.mark.asyncio
    async def test_toy_policy_generate_action(self):
        """Test that ToyPolicy generates actions within specified range."""
        proc = await local_proc_mesh(gpus=1)
        policy = await proc.spawn("policy", ToyPolicy, action_range=(-2.0, 2.0))

        observation = ToyObservation(
            step=1, data=torch.tensor([1.5]), text="Step 1, Value: 1.5"
        )

        action = await policy.generate.choose(observation)

        assert isinstance(action, ToyAction)
        assert action.data is not None
        assert len(action.data) == 1

        # Action should be within the specified range
        action_value = float(action.data[0])
        assert -2.0 <= action_value <= 2.0

    @pytest.mark.asyncio
    async def test_toy_policy_multiple_generations(self):
        """Test that ToyPolicy generates different actions (due to randomness)."""
        proc = await local_proc_mesh(gpus=1)
        policy = await proc.spawn("policy", ToyPolicy, action_range=(-1.0, 1.0))

        observation = ToyObservation(
            step=1, data=torch.tensor([0.0]), text="Step 1, Value: 0.0"
        )

        actions = []
        for _ in range(10):
            action = await policy.generate.choose(observation)
            actions.append(float(action.data[0]))

        # All actions should be in range
        for action_value in actions:
            assert -1.0 <= action_value <= 1.0

        # With high probability, not all actions should be identical
        # (this could theoretically fail due to randomness, but very unlikely)
        assert len(set(actions)) > 1


class TestReplayBuffer:
    """Test the ReplayBuffer component."""

    @pytest.mark.asyncio
    async def test_replay_buffer_initialization(self):
        """Test that ReplayBuffer initializes correctly."""
        proc = await local_proc_mesh(gpus=1)
        buffer = await proc.spawn("buffer", ReplayBuffer)

        length = await buffer.len.choose()
        is_empty = await buffer.is_empty.choose()

        assert length == 0
        assert is_empty is True

    @pytest.mark.asyncio
    async def test_replay_buffer_extend_and_sample(self):
        """Test adding and sampling trajectories."""
        proc = await local_proc_mesh(gpus=1)
        buffer = await proc.spawn("buffer", ReplayBuffer)

        # Create a test trajectory
        trajectory = Trajectory()
        trajectory.states = [
            ToyObservation(step=0, data=torch.tensor([0.0]), text="Step 0, Value: 0.0")
        ]
        trajectory.actions = [ToyAction(data=torch.tensor([1.0]))]
        trajectory.rewards = [2.0]
        trajectory.dones = [False]
        trajectory.infos = [{"test": "info"}]

        # Add trajectory to buffer
        await buffer.extend.choose(trajectory)

        # Check buffer state
        length = await buffer.len.choose()
        is_empty = await buffer.is_empty.choose()

        assert length == 1
        assert is_empty is False

        # Sample from buffer
        sampled_trajectory = await buffer.sample.choose()

        assert sampled_trajectory is not None
        assert len(sampled_trajectory.states) == 1
        assert len(sampled_trajectory.actions) == 1
        assert len(sampled_trajectory.rewards) == 1
        assert sampled_trajectory.rewards[0] == 2.0

    @pytest.mark.asyncio
    async def test_replay_buffer_multiple_trajectories(self):
        """Test buffer with multiple trajectories."""
        proc = await local_proc_mesh(gpus=1)
        buffer = await proc.spawn("buffer", ReplayBuffer)

        # Add multiple trajectories
        for i in range(5):
            trajectory = Trajectory()
            trajectory.states = [
                ToyObservation(
                    step=i,
                    data=torch.tensor([float(i)]),
                    text=f"Step {i}, Value: {float(i)}",
                )
            ]
            trajectory.actions = [ToyAction(data=torch.tensor([float(i + 1)]))]
            trajectory.rewards = [float(i * 2)]
            trajectory.dones = [i == 4]
            trajectory.infos = [{"trajectory_id": i}]

            await buffer.extend.choose(trajectory)

        length = await buffer.len.choose()
        assert length == 5

        # Sample multiple times
        sampled_rewards = []
        for _ in range(10):  # Sample more than we have to test sampling behavior
            sampled = await buffer.sample.choose()
            if sampled and sampled.rewards:
                sampled_rewards.append(sampled.rewards[0])

        # Should have sampled some trajectories
        assert len(sampled_rewards) > 0
        # All sampled rewards should be from our original set
        expected_rewards = {0.0, 2.0, 4.0, 6.0, 8.0}
        for reward in sampled_rewards:
            assert reward in expected_rewards


class TestCollector:
    """Test the Collector component (integration test)."""

    @pytest.mark.asyncio
    async def test_collector_initialization(self):
        """Test that Collector initializes with proper components."""

        class MockPolicy(Actor):
            @endpoint
            async def generate(self, observation):
                return ToyAction(data=torch.tensor([1.0]))

        class MockReplayBuffer(Actor):
            def __init__(self):
                self.trajectories = []

            @endpoint
            async def extend(self, trajectory):
                self.trajectories.append(trajectory)

        class MockRewarder(Actor):
            @endpoint
            async def compute_reward(self, state, action, next_state):
                return 1.0

        max_collector_steps = 5
        proc = await local_proc_mesh(gpus=1)
        mock_policy = await proc.spawn("policy", MockPolicy)
        mock_replay_buffer = await proc.spawn("replay_buffer", MockReplayBuffer)
        mock_rewarder = await proc.spawn("rewarder", MockRewarder)

        def environment_creator():
            # pyre-ignore[6]: MockRewarder<>ToyRewarder discrepancy
            return ToyEnvironment(name="test", max_steps=5, rewarder=mock_rewarder)

        assert callable(environment_creator)

        # Test environment creation
        env = environment_creator()
        assert isinstance(env, ToyEnvironment)
        assert env.name == "test"

        collector = await proc.spawn(
            "collector",
            Collector,
            max_collector_steps=max_collector_steps,
            policy=mock_policy,
            replay_buffer=mock_replay_buffer,
            environment_creator=environment_creator,
        )

        # Test that collector can run an episode
        trajectory = await collector.run_episode.choose()
        assert trajectory is not None

        # Check that trajectory has expected structure
        assert hasattr(trajectory, "states")
        assert hasattr(trajectory, "actions")
        assert hasattr(trajectory, "rewards")
        assert hasattr(trajectory, "dones")

        # Verify trajectory contains data (should have at least one step)
        assert len(trajectory.states) > 0
        assert len(trajectory.actions) > 0
        assert len(trajectory.rewards) > 0

        # Check that states are ToyObservations and actions are ToyActions
        assert all(isinstance(state, ToyObservation) for state in trajectory.states)
        assert all(isinstance(action, ToyAction) for action in trajectory.actions)

        # Verify rewards are floats
        assert all(isinstance(reward, float) for reward in trajectory.rewards)

        # Test that collector respects max_collector_steps config
        assert len(trajectory.states) <= max_collector_steps + 1  # +1 for initial state


class TestIntegration:
    """Integration tests that combine multiple components."""

    @pytest.mark.asyncio
    async def test_environment_policy_interaction(self):
        """Test that environment and policy can interact correctly."""

        class MockRewarder(Actor):
            @endpoint
            async def compute_reward(self, state, action, next_state):
                return 1.0

        proc = await local_proc_mesh(gpus=1)
        mock_rewarder = await proc.spawn("rewarder", MockRewarder)
        env = ToyEnvironment(
            name="integration_test",
            # pyre-ignore[6]: MockRewarder<>ToyRewarder discrepancy
            rewarder=mock_rewarder,
            max_steps=3,
        )

        initial_obs = env.reset()
        assert initial_obs.step == 0

        action = ToyAction(data=torch.tensor([1.5]))
        next_obs, reward = env.step(action)
        assert next_obs.step == 1
        assert torch.equal(next_obs.data, torch.tensor([1.5]))
        assert reward == 1.0

        # Test another step to ensure continued functionality
        action2 = ToyAction(data=torch.tensor([0.5]))
        next_obs2, reward2 = env.step(action2)
        assert next_obs2.step == 2
        assert torch.equal(next_obs2.data, torch.tensor([2.0]))  # 1.5 + 0.5
        assert reward2 == 1.0

    @pytest.mark.asyncio
    async def test_full_rl_pipeline_simulation(self):
        """Test that simulates the full RL pipeline with concurrent tasks."""
        max_collector_steps = 5
        proc = await local_proc_mesh(gpus=1)
        policy = await proc.spawn("policy", ToyPolicy, action_range=(-2.0, 2.0))
        replay_buffer = await proc.spawn("replay_buffer", ReplayBuffer)
        rewarder = await proc.spawn("rewarder", ToyRewarder)
        collector = await proc.spawn(
            "collector",
            Collector,
            max_collector_steps=max_collector_steps,
            policy=policy,
            replay_buffer=replay_buffer,
            environment_creator=partial(
                ToyEnvironment, name="test_env", max_steps=3, rewarder=rewarder
            ),
        )
        generated_trajectories = []
        sampled_trajectories = []

        async def episode_generator_task():
            """Task that generates 5 trajectories."""
            for i in range(5):
                trajectory = await collector.run_episode.choose()
                generated_trajectories.append(trajectory)
                # Small delay to allow sampling task to run
                await asyncio.sleep(0.1)

        async def replay_buffer_sampler_task():
            """Task that samples from replay buffer."""
            samples_collected = 0
            max_attempts = 20  # Prevent infinite loop
            attempts = 0

            while samples_collected < 3 and attempts < max_attempts:
                attempts += 1
                await asyncio.sleep(0.2)  # Wait a bit for trajectories to be added

                # Check if buffer has data
                buffer_length = await replay_buffer.len.choose()
                if buffer_length > 0:
                    sampled_trajectory = await replay_buffer.sample.choose()
                    if sampled_trajectory is not None:
                        sampled_trajectories.append(sampled_trajectory)
                        samples_collected += 1

        # Run both tasks concurrently (like in rl_main)
        await asyncio.gather(episode_generator_task(), replay_buffer_sampler_task())

        # Verify results
        assert (
            len(generated_trajectories) == 5
        ), f"Expected 5 trajectories, got {len(generated_trajectories)}"
        assert (
            len(sampled_trajectories) >= 1
        ), f"Expected at least 1 sampled trajectory, got {len(sampled_trajectories)}"

        # Verify generated trajectories have correct structure
        for i, trajectory in enumerate(generated_trajectories):
            assert trajectory is not None, f"Trajectory {i} is None"
            assert len(trajectory.states) > 0, f"Trajectory {i} has no states"
            assert len(trajectory.actions) > 0, f"Trajectory {i} has no actions"
            assert len(trajectory.rewards) > 0, f"Trajectory {i} has no rewards"

            # Check data types
            assert all(isinstance(state, ToyObservation) for state in trajectory.states)
            assert all(isinstance(action, ToyAction) for action in trajectory.actions)
            assert all(isinstance(reward, float) for reward in trajectory.rewards)

            # Verify trajectory respects max_collector_steps
            assert len(trajectory.states) <= max_collector_steps + 1

        # Verify sampled trajectories match expected structure
        for i, trajectory in enumerate(sampled_trajectories):
            assert trajectory is not None, f"Sampled trajectory {i} is None"
            assert len(trajectory.states) > 0, f"Sampled trajectory {i} has no states"
            assert len(trajectory.actions) > 0, f"Sampled trajectory {i} has no actions"
            assert len(trajectory.rewards) > 0, f"Sampled trajectory {i} has no rewards"

        # Verify replay buffer contains trajectories
        final_buffer_length = await replay_buffer.len.choose()
        assert (
            final_buffer_length >= 3
        ), f"Expected buffer to have at least 3 trajectories, got {final_buffer_length}"

        # Verify that rewards follow ToyRewarder logic (next_state_value + 1.0)
        for trajectory in generated_trajectories:
            for i, (state, reward) in enumerate(
                zip(trajectory.states[1:], trajectory.rewards)
            ):
                # ToyRewarder adds 1.0 to the state value
                expected_reward = float(state.data[0]) + 1.0
                assert (
                    reward == expected_reward
                ), f"Expected reward {expected_reward}, got {reward}"


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
