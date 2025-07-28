# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""A working example showcasing a practical example of forge with RL.

Run this with:
    python -m apps.toy_rl.main
"""

import asyncio
from functools import partial

from apps.toy_rl.components import ToyEnvironment, ToyPolicy

from forge.actors.collector import Collector

from forge.controller.stack import stack
from forge.data.replay_buffer import ReplayBuffer
from monarch.actor import proc_mesh


SAMPLES_PER_BATCH = 4  # How many trajectories to sample at once


async def main():
    print("Starting RL example with toy environment...")

    # Process allocation
    policy_procs = await proc_mesh(gpus=2)
    replay_procs = await proc_mesh(gpus=1)

    # Note - here is where we implement our "mixture" logic.
    browser_procs = await proc_mesh(gpus=2)
    deep_research_procs = await proc_mesh(gpus=4)
    coder_procs = await proc_mesh(gpus=8)

    # Actor instantiation
    replay_buffer = await replay_procs.spawn("replay_buffer", ReplayBuffer)

    # TODO - add in an example of a "vLLM executor" and "vLLM controller"
    # This policy just generates something between -2. and 2.
    policy = await policy_procs.spawn("policy", ToyPolicy, action_range=(-2.0, 2.0))

    browser_collectors = await browser_procs.spawn(
        "browser",
        Collector,
        max_collector_steps=5,
        policy=policy,
        replay_buffer=replay_buffer,
        # here, we use a partial so that the collector itself
        # can create its own environment.
        # We could create the environment and pass it in, but it's slightly more efficient
        # to do it this way.
        environment_creator=partial(ToyEnvironment, name="browser", max_steps=5),
    )
    deep_research_collectors = await deep_research_procs.spawn(
        "deep_research",
        Collector,
        max_collector_steps=5,
        policy=policy,
        replay_buffer=replay_buffer,
        environment_creator=partial(ToyEnvironment, name="deep_research", max_steps=5),
    )
    coding_collectors = await coder_procs.spawn(
        "coding",
        Collector,
        max_collector_steps=5,
        policy=policy,
        replay_buffer=replay_buffer,
        environment_creator=partial(ToyEnvironment, name="coding", max_steps=5),
    )

    # Here's our stack API in action!
    collectors = stack(
        browser_collectors,
        deep_research_collectors,
        coding_collectors,
        interface=Collector,
    )

    # Create two async tasks
    async def episode_collector_task():
        """Task that continuously runs episodes to fill the replay buffer."""
        episode_count = 0
        while True:
            try:
                print(f"🎮 Running episode {episode_count + 1}...")

                # call() is essentially our "map" - every collector runs their own
                # episode loop.
                # What's pretty elegant here is if we wanted to control off policiness, we could
                # easily counter on steps and call policy.update_weights.call() at our desired
                # frequency.
                results = await collectors.run_episode.call()
                num_trajectories = sum([len(r._values) for r in results])
                episode_count += 1
                print(
                    f"✅ Episode {episode_count} completed! Generated {num_trajectories} trajectories."
                )

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

                # Check if buffer has enough data
                buffer_length = await replay_buffer.len.choose()
                if buffer_length < SAMPLES_PER_BATCH:
                    print(
                        f"📦 Replay buffer has {buffer_length} trajectories, waiting for at least {SAMPLES_PER_BATCH}..."
                    )
                    continue

                # Sample multiple trajectories at once
                trajectories = []
                for _ in range(SAMPLES_PER_BATCH):
                    trajectory = await replay_buffer.sample.choose()
                    if trajectory is not None:
                        trajectories += trajectory

                # Most of the rest of this is just boilerplate for pretty printing.
                if not trajectories:
                    continue

                sample_count += 1
                print(
                    f"\n🔍 Sample #{sample_count} from replay buffer (buffer size: {buffer_length}):"
                )
                print("=" * 80)

                for idx, trajectory in enumerate(trajectories, 1):
                    # Extract environment name from the first state text
                    env_name = "unknown"
                    if (
                        trajectory.states
                        and len(trajectory.states) > 0
                        and trajectory.states[0].text
                    ):
                        state_text = trajectory.states[0].text
                        if state_text.startswith("[") and "]" in state_text:
                            env_name = state_text.split("]")[0][
                                1:
                            ]  # Extract name between [ and ]

                    print(f"🏷️  Trajectory {idx} - Environment: {env_name}")
                    print("-" * 40)

                    if trajectory.states and trajectory.actions:
                        for i, (state, action) in enumerate(
                            zip(
                                trajectory.states,
                                trajectory.actions,
                            )
                        ):
                            # Extract values for pretty printing
                            state_value = (
                                float(state.data[0]) if state.data is not None else 0.0
                            )
                            action_value = (
                                float(action.data[0])
                                if action.data is not None
                                else 0.0
                            )

                            print(
                                f"  Step {i+1:2d}: State={state_value:6.2f} → Action={action_value:6.2f}"
                            )

                    if idx < len(trajectories):  # Add spacing between trajectories
                        print()

                print("=" * 80)

            except Exception as e:
                print(f"❌ Error sampling from replay buffer: {e}")
                await asyncio.sleep(1)

    print("🚀 Starting continuous episode generation and replay buffer sampling...")
    print("Press Ctrl+C to stop")

    # Run both tasks concurrently
    try:
        await asyncio.gather(episode_collector_task(), replay_buffer_sampler_task())
    except KeyboardInterrupt:
        print("\n🛑 Stopping tasks...")


if __name__ == "__main__":
    asyncio.run(main())
