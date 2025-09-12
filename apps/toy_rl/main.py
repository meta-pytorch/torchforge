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
from dataclasses import dataclass
from functools import partial

import torch
from forge.actors.collector import Collector

from forge.actors.replay_buffer import ReplayBuffer

from forge.data.stores import KVStore
from forge.interfaces import Environment, Policy
from forge.types import Action, Observation, State
from monarch.actor import endpoint, proc_mesh

SAMPLES_PER_BATCH = 4  # How many trajectories to sample at once


@dataclass
class ToyState(State):
    """State for the toy environment."""

    data: torch.Tensor
    step: int

    def __repr__(self) -> str:
        return f"ToyState(step={self.step}, data={self.data})"


@dataclass
class ToyObservation(Observation):
    """Observation for the toy environment."""

    data: torch.Tensor
    step: int
    text: str

    def __repr__(self) -> str:
        return f"ToyObservation(step={self.step}, data={self.data})"


@dataclass
class ToyAction(Action):
    """Action for the toy environment."""

    data: torch.Tensor


class ToyEnvironment(Environment):
    """A simple toy environment for testing the RL pipeline.

    This environment maintains a simple numeric state that gets modified by actions.
    It follows the base Environment abstraction with only reset, step, and state methods.
    """

    def __init__(self, name: str, max_steps: int = 10):
        self.name = name
        self.max_steps = max_steps
        self.reset()

    def reset(self) -> ToyObservation:
        """Reset the environment to initial state."""
        self._state = ToyState(
            step=0,
            data=torch.tensor([0.0]),
        )
        return ToyObservation(
            step=self._state.step,
            data=self._state.data,
            text=f"[{self.name}] Step {self._state.step}, Value: {self._state.data}",
        )

    def step(self, action: ToyAction) -> ToyObservation:
        """Take a step in the environment."""
        next_state = ToyState(
            step=self._state.step + 1,
            data=self._state.data + action.data,
        )

        self._state = next_state

        return ToyObservation(
            step=next_state.step,
            data=next_state.data,
            text=f"[{self.name}] Step {next_state.step}, Value: {next_state.data}",
        )

    @property
    def state(self) -> ToyState:
        """Get the current state of the environment."""
        return self._state


class ToyPolicy(Policy):
    """A simple toy policy for testing."""

    def __init__(self, action_range: tuple[float, float] = (-1.0, 1.0)):
        super().__init__()
        self.action_range = action_range

    @endpoint
    async def generate(self, request: Observation) -> Action:
        """Generate a simple random action."""
        # Generate a random action within the specified range
        action_value = (
            torch.rand(1).item() * (self.action_range[1] - self.action_range[0])
            + self.action_range[0]
        )
        action = ToyAction(
            data=torch.tensor([action_value]),
        )
        return action

    @endpoint
    async def update_weights(self):
        """No-op for toy policy."""
        pass


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
    replay_buffer = await replay_procs.spawn(
        "replay_buffer",
        ReplayBuffer,
        store=KVStore(),
        batch_size=SAMPLES_PER_BATCH,
        max_policy_age=float("inf"),
        dp_size=1,
    )

    # TODO - add in an example of a "vLLM executor" and "vLLM controller"
    # This policy just generates something between -2. and 2.
    policy = await policy_procs.spawn("policy", ToyPolicy, action_range=(-2.0, 2.0))

    # TODO - replace multiple collectors with a service.
    collectors = await browser_procs.spawn(
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

    # Create two async tasks
    async def episode_collector_task():
        """Task that continuously runs episodes to fill the replay buffer."""
        episode_count = 0
        while True:
            try:
                print(f"🎮 Running episode {episode_count + 1}...")
                results = await collectors.run_episode.call()
                num_trajectories = len([r for r in results])
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
                buffer_length = await replay_buffer._numel.choose()
                if buffer_length < SAMPLES_PER_BATCH:
                    print(
                        f"📦 Replay buffer has {buffer_length} trajectories, waiting for at least {SAMPLES_PER_BATCH}..."
                    )
                    continue

                # Sample multiple trajectories at once
                trajectories = []
                for _ in range(SAMPLES_PER_BATCH):
                    trajectory = await replay_buffer.sample.choose(
                        curr_policy_version=float(
                            "inf"
                        )  # Update with true policy version when available
                    )
                    if trajectory is not None:
                        trajectories += trajectory[0]

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
                                f"  Step {i + 1:2d}: State={state_value:6.2f} → Action={action_value:6.2f}"
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
