# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""A simple toy environment intended only for testing the RL pipeline."""

from dataclasses import dataclass

import torch

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
