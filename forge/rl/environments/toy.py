# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

import torch
from monarch.actor_mesh import Actor, ActorMeshRef, endpoint

from forge.rl.environments.base import Action, Environment, Observation, State


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


class ToyRewarder(Actor):
    """A very simple toy rewarder for testing data flow."""

    @endpoint
    async def compute_reward(
        self,
        state: ToyState,
        action: ToyAction,
        next_state: ToyState,
    ) -> float:
        """Simple reward: next_state_value + 1."""

        # Extract the state value from next_state
        if next_state.data is not None and len(next_state.data) > 0:
            state_value = float(next_state.data[0])
            return state_value + 1.0

        # Default reward if no data
        return 1.0


class ToyEnvironment(Environment):
    """A simple toy environment for testing the RL pipeline.

    This environment maintains a simple numeric state that gets modified by actions.
    It follows the base Environment abstraction with only reset, step, and state methods.
    """

    def __init__(
        self, name: str, rewarder: ActorMeshRef[ToyRewarder], max_steps: int = 10
    ):
        self.name = name
        self.max_steps = max_steps
        self.reset()
        self.rewarder = rewarder

    def reset(self) -> ToyObservation:
        """Reset the environment to initial state."""
        self._state = ToyState(
            step=0,
            data=torch.tensor([0.0]),
        )
        return ToyObservation(
            step=self._state.step,
            data=self._state.data,
            text=f"Step {self._state.step}, Value: {self._state.data}",
        )

    def step(self, action: ToyAction) -> tuple[ToyObservation, float]:
        """Take a step in the environment."""
        next_state = ToyState(
            step=self._state.step + 1,
            data=self._state.data + action.data,
        )

        # Simple reward: positive if state_value is positive, negative otherwise
        # Note: For now using simple reward since async rewarder calls need special handling
        reward = self.rewarder.compute_reward.choose(
            self._state, action, next_state
        ).get()

        return (
            ToyObservation(
                step=next_state.step,
                data=next_state.data,
                text=f"Step {next_state.step}, Value: {next_state.data}",
            ),
            reward,
        )

    @property
    def state(self) -> ToyState:
        """Get the current state of the environment."""
        return self._state
