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

from forge.actors.rollout import RolloutActor

from forge.controller.stack import stack
from forge.data.replay_buffer import ReplayBuffer
from forge.interfaces import Environment, Policy
from forge.types import Message, State
from monarch.actor import endpoint, proc_mesh

SAMPLES_PER_BATCH = 2  # How many trajectories to sample at once
from itertools import cycle


class ToyEnvironment(Environment):
    """A simple toy environment for testing the RL pipeline."""

    def __init__(self, name: str, dataloader):
        self.name = name
        self._dataloader = cycle(dataloader)
        self.reset()

    def reset(self) -> State:
        """Reset the environment to initial state."""
        obs = next(self._dataloader)
        return State(
            observations=[Message.from_dict(obs)], info={"env_name": self.name}
        )

    def step(self, state: State) -> State:
        """Take a step in the environment."""
        action: Message = state.observations[-1]
        if action.role == "assistant":
            state.terminated = True
        return state


class ToyPolicy(Policy):
    """A simple toy policy for testing."""

    def __init__(self):
        super().__init__()
        self._policy_version = 0

    @endpoint
    async def generate(self, state: State) -> State:
        """Generate a simple random action."""
        # Generate a random action within the specified range
        request = state.observations[-1]
        response = self._generate_rand(request.content)
        state.observations.append(Message(role="assistant", content=response))
        state.policy_version = self._policy_version
        return state

    def _generate_rand(self, request: str) -> str:
        import hashlib
        import random
        import string

        seed = int(hashlib.sha256(request.encode()).hexdigest(), 16)
        rng = random.Random(seed)

        length = rng.randint(1, 20)

        chars = string.ascii_letters + string.digits
        return "".join(rng.choice(chars) for _ in range(length))

    @endpoint
    async def update_weights(self):
        """No-op for toy policy."""
        self._policy_version += 1


async def main():
    print("Starting RL example with toy environment...")

    # Process allocation
    policy_procs = await proc_mesh(gpus=2)
    replay_procs = await proc_mesh(gpus=1)

    # Note - here is where we implement our "mixture" logic.
    math_procs = await proc_mesh(gpus=4)
    coder_procs = await proc_mesh(gpus=8)

    # Actor instantiation
    replay_buffer = await replay_procs.spawn(
        "replay_buffer",
        ReplayBuffer,
        batch_size=SAMPLES_PER_BATCH,
        max_policy_age=1_000_000,
    )

    # TODO - add in an example of a "vLLM executor" and "vLLM controller"
    policy = await policy_procs.spawn("policy", ToyPolicy)

    math_dataset = iter(
        [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "user", "content": "What is an approximation of Pi?"},
            {
                "role": "user",
                "content": "If you have 1 turtle and I have 2 turtles, how many turtles do we have?",
            },
        ]
    )
    math_collectors = await math_procs.spawn(
        "math",
        RolloutActor,
        max_collector_steps=3,
        policy=policy,
        replay_buffer=replay_buffer,
        environment_creator=partial(
            ToyEnvironment, name="math", dataloader=math_dataset
        ),
    )
    coding_dataset = iter(
        [
            {
                "role": "user",
                "content": "Write a Python program to solve for x given a simple equation.",
            },
            {"role": "user", "content": "Fix the bug: ```def add(x, y): x - y```"},
            {
                "role": "user",
                "content": "Add an inline comment explaining the function: ```def sub(x, y): x - y```",
            },
        ]
    )
    coding_collectors = await coder_procs.spawn(
        "coding",
        RolloutActor,
        max_collector_steps=3,
        policy=policy,
        replay_buffer=replay_buffer,
        environment_creator=partial(
            ToyEnvironment, name="coding", dataloader=coding_dataset
        ),
    )

    collectors = stack(
        math_collectors,
        coding_collectors,
        interface=RolloutActor,
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
                results = collectors.run_episode.call()

                # Temporary hack due to Monarch changes - ideally you could just await results
                results = [await r for r in results]
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
        curr_policy_version = 0
        while True:
            try:
                await asyncio.sleep(3)  # Wait a bit before sampling

                # Check if buffer has enough data
                buffer_length = await replay_buffer._numel.choose()
                if buffer_length < SAMPLES_PER_BATCH:
                    print(
                        f"📦 Replay buffer has {buffer_length} trajectories, waiting for at least {SAMPLES_PER_BATCH}..."
                    )
                    continue

                # Sample multiple trajectories at once
                states = []
                for _ in range(SAMPLES_PER_BATCH):
                    state = await replay_buffer.sample.choose(
                        curr_policy_version=curr_policy_version
                    )
                    if state is not None:
                        states += state

                # Most of the rest of this is just boilerplate for pretty printing.
                if not states:
                    continue

                sample_count += 1
                print(
                    f"\n🔍 Sample #{sample_count} from replay buffer (buffer size: {buffer_length}):"
                )
                print("=" * 80)

                for idx, state in enumerate(states):
                    print(
                        f"🏷️  State {idx} - Environment: {state.info['env_name']} - "
                        f"Prompt: {state.observations[0]} - Response: {state.observations[1]}"
                    )
                    print("-" * 40)

                    if idx < len(states):  # Add spacing between trajectories
                        print()

                print("=" * 80)
                curr_policy_version += 1

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
