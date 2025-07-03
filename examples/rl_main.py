# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""A working example showcasing a practical example of forge with RL.

Run this with:
    python -m examples.rl_main
"""

import asyncio
from dataclasses import dataclass
from functools import partial

from forge.monarch_utils.stack import stack
from forge.rl.config import GeneratorConfig
from forge.rl.environments import ToyEnvironment
from forge.rl.generator import Generator
from forge.rl.policy import ToyPolicy
from forge.rl.replay_buffer import ReplayBuffer
from forge.rl.rewards import ToyRewarder

from monarch.proc_mesh import proc_mesh


@dataclass
class Config:
    pass


async def main():
    print("Starting RL example with toy environment and rewarder...")
    print("ToyRewarder: reward = next_state_value + 1")

    # Process allocation
    policy_procs = await proc_mesh(gpus=2)
    replay_procs = await proc_mesh(gpus=1)

    # note - assuming we'd end up doing like
    browser_procs = await proc_mesh(gpus=2)
    deep_research_procs = await proc_mesh(gpus=4)
    coder_procs = await proc_mesh(gpus=8)

    rewarder_procs = await proc_mesh(gpus=1)

    # Actor instantiation
    replay_buffer = await replay_procs.spawn("replay_buffer", ReplayBuffer)
    policy = await policy_procs.spawn("policy", ToyPolicy, action_range=(-2.0, 2.0))
    # Practically, we should create a single rewarder that's shared across the actors
    # in its associated environment mesh. This provides cleaner decoupling and
    # lets us customize our routing. But ultimately we will want to package this
    # with our environment definition somehow.
    rewarder = await rewarder_procs.spawn("rewarder", ToyRewarder)

    generator_config = GeneratorConfig(
        max_generator_steps=5,
    )

    browser_generators = await browser_procs.spawn(
        "browser",
        Generator,
        config=generator_config,
        policy=policy,
        replay_buffer=replay_buffer,
        environment_creator=partial(
            ToyEnvironment, name="browser", max_steps=5, rewarder=rewarder
        ),
    )
    deep_research_generators = await deep_research_procs.spawn(
        "deep_research",
        Generator,
        config=generator_config,
        policy=policy,
        replay_buffer=replay_buffer,
        environment_creator=partial(
            ToyEnvironment, name="deep_research", max_steps=5, rewarder=rewarder
        ),
    )
    coding_generators = await coder_procs.spawn(
        "coding",
        Generator,
        config=generator_config,
        policy=policy,
        replay_buffer=replay_buffer,
        environment_creator=partial(
            ToyEnvironment, name="coding", max_steps=5, rewarder=rewarder
        ),
    )

    generators = stack(
        browser_generators,
        deep_research_generators,
        coding_generators,
        interface=Generator,
    )

    # Create two async tasks
    async def episode_generator_task():
        """Task that continuously runs episodes to fill the replay buffer."""
        episode_count = 0
        while True:
            try:
                print(f"🎮 Running episode {episode_count + 1}...")
                await generators.run_episode.call()
                episode_count += 1
                print(f"✅ Episode {episode_count} completed!")

                # Wait a bit before next episode
                await asyncio.sleep(2)

            except Exception as e:
                print(f"❌ Error in episode {episode_count + 1}: {e}")
                await asyncio.sleep(1)

    async def replay_buffer_sampler_task():
        """Task that samples from replay buffer and prints trajectories in a pretty way."""
        sample_count = 0
        while True:
            try:
                await asyncio.sleep(3)  # Wait a bit before sampling

                # Check if buffer has data
                buffer_length = await replay_buffer.len.choose()
                if buffer_length == 0:
                    print("📦 Replay buffer is empty, waiting for episodes...")
                    continue

                # Sample a trajectory
                trajectory = await replay_buffer.sample.choose()
                if trajectory is None:
                    continue

                sample_count += 1
                print(
                    f"\n🔍 Sample #{sample_count} from replay buffer (buffer size: {buffer_length}):"
                )
                print("=" * 60)

                if trajectory.states and trajectory.actions and trajectory.rewards:
                    for i, (state, action, reward) in enumerate(
                        zip(trajectory.states, trajectory.actions, trajectory.rewards)
                    ):
                        # Extract values for pretty printing
                        state_value = (
                            float(state.data[0]) if state.data is not None else 0.0
                        )
                        action_value = (
                            float(action.data[0]) if action.data is not None else 0.0
                        )

                        print(
                            f"  Step {i+1:2d}: State={state_value:6.2f} → Action={action_value:6.2f} → Reward={reward:6.2f}"
                        )

                total_reward = sum(trajectory.rewards) if trajectory.rewards else 0
                print(f"  📊 Total Reward: {total_reward:.2f}")
                print("=" * 60)

            except Exception as e:
                print(f"❌ Error sampling from replay buffer: {e}")
                await asyncio.sleep(1)

    print("🚀 Starting continuous episode generation and replay buffer sampling...")
    print("Press Ctrl+C to stop")

    # Run both tasks concurrently
    try:
        await asyncio.gather(episode_generator_task(), replay_buffer_sampler_task())
    except KeyboardInterrupt:
        print("\n🛑 Stopping tasks...")


if __name__ == "__main__":
    asyncio.run(main())
