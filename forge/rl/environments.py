# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

from monarch.actor_mesh import ActorMeshRef

from forge.rl.entities import ForgeEnvInfo, ForgeRequest
from forge.rl.rewards import ToyRewarder


# Stand-in contract for environments
class ForgeEnvironment:
    # obs, info
    def reset(self) -> tuple[ForgeRequest, ForgeEnvInfo]:
        raise NotImplementedError("Subclasses must implement reset method")

    # obs, rew, term, truncated, info
    def step(
        self, action: ForgeRequest
    ) -> tuple[ForgeRequest, float, bool, bool, ForgeEnvInfo]:
        raise NotImplementedError("Subclasses must implement step method")


class ToyEnvironment(ForgeEnvironment):
    """A simple toy environment for testing the RL pipeline."""

    def __init__(self, rewarder: ActorMeshRef[ToyRewarder], max_steps: int = 10):
        self.rewarder = rewarder
        self.max_steps = max_steps
        self.current_step = 0
        self.state_value = 0.0

    def reset(self) -> tuple[ForgeRequest, ForgeEnvInfo]:
        """Reset the environment to initial state."""
        self.current_step = 0
        self.state_value = 0.0

        state = ForgeRequest(
            data=torch.tensor([self.state_value]),
            text=f"Step {self.current_step}, Value: {self.state_value}",
            metadata={"step": self.current_step, "value": self.state_value},
        )

        info = ForgeEnvInfo(
            episode_id="toy_episode",
            step_count=self.current_step,
            metadata={"max_steps": self.max_steps},
        )

        return state, info

    def step(
        self, action: ForgeRequest
    ) -> tuple[ForgeRequest, float, bool, bool, ForgeEnvInfo]:
        """Take a step in the environment."""
        # Store previous state for rewarder
        prev_state = ForgeRequest(
            data=torch.tensor([self.state_value]),
            text=f"Step {self.current_step}, Value: {self.state_value}",
            metadata={"step": self.current_step, "value": self.state_value},
        )

        self.current_step += 1

        # Simple action interpretation: if action has data, use it to modify state
        if action.data is not None and len(action.data) > 0:
            action_value = float(action.data[0])
            self.state_value += action_value
        else:
            # Default action: increment by 1
            self.state_value += 1.0

        # Create next state
        next_state = ForgeRequest(
            data=torch.tensor([self.state_value]),
            text=f"Step {self.current_step}, Value: {self.state_value}",
            metadata={"step": self.current_step, "value": self.state_value},
        )

        # Simple reward: positive if state_value is positive, negative otherwise
        # Note: For now using simple reward since async rewarder calls need special handling
        reward = self.rewarder.compute_reward.choose(
            prev_state, action, next_state
        ).get()

        # Termination conditions
        terminated = (
            abs(self.state_value) >= 10.0
        )  # Episode ends if value gets too large
        truncated = self.current_step >= self.max_steps  # Episode ends after max steps

        # Create info
        info = ForgeEnvInfo(
            episode_id="toy_episode",
            step_count=self.current_step,
            metadata={
                "max_steps": self.max_steps,
                "terminated": terminated,
                "truncated": truncated,
                "prev_state_value": (
                    float(prev_state.data[0]) if prev_state.data is not None else 0.0
                ),
                "rewarder_available": self.rewarder is not None,
            },
        )

        return next_state, reward, terminated, truncated, info
