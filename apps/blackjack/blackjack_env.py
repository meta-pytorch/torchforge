# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import re
from dataclasses import dataclass, field
from typing import Any

from envs.openspiel_env import OpenSpielAction, OpenSpielEnv
from forge.observability.metrics import record_metric, Reduce


@dataclass
class EnvStepResult:
    """Result from environment step."""

    observation: dict[str, str]  # Next message: {"role": "user", "content": "..."}
    reward: float  # Reward for this step
    done: bool  # Episode ended?
    metadata: dict[str, Any] = field(default_factory=dict)


class BlackjackEnv:
    """
    Minimal blackjack environment.

    Responsibilities:
    - Manage game state via OpenSpielEnv
    - Parse actions from text
    - Return next observation message
    - Compute rewards

    Does NOT:
    - Hold message history (rollout loop does this)
    - Tokenize (rollout loop does this)
    - Track cumulative tokens (rollout loop does this)
    """

    def __init__(self, server_url: str):
        self.server_url = server_url
        self.client = OpenSpielEnv(base_url=server_url)
        self.client._http.trust_env = False

        # Game state
        self.turn_count = 0
        self.has_invalid_action = False

    def reset(self) -> str:
        """
        Reset game and return initial user message.

        Returns:
            Initial observation text (NOT a dict, just the content string)
        """
        self.turn_count = 0
        self.has_invalid_action = False

        # Reset game
        result = self.client.reset()

        # Build initial observation
        return self._format_observation(result.observation)

    def step(self, action_text: str) -> EnvStepResult:
        """
        Execute action and return next observation.

        Args:
            action_text: The assistant's text response

        Returns:
            EnvStepResult with next observation message, reward, done
        """

        # Parse action
        action_name, error_type = self._parse_action(action_text)

        # Track invalid actions
        is_invalid = action_name == "INVALID"
        if is_invalid:
            self.has_invalid_action = True
            action_name = "STAND"  # Treat invalid as STAND
            record_metric("game/invalid_action_rate", 1, Reduce.MEAN)
            if error_type == "NO_TAGS":
                record_metric("game/missing_answer_tags", 1, Reduce.SUM)
            elif error_type == "INVALID_CONTENT":
                record_metric("game/invalid_answer_content", 1, Reduce.SUM)
        else:
            record_metric("game/invalid_action_rate", 0, Reduce.MEAN)

        # Execute in game
        action_id = 0 if action_name == "HIT" else 1
        result = self.client.step(
            OpenSpielAction(action_id=action_id, game_name="blackjack")
        )

        self.turn_count += 1

        # Compute reward
        if result.done:
            reward = self._compute_reward(
                result.reward, is_invalid=self.has_invalid_action
            )
            # Record game outcome metrics
            record_metric("game/games_played", 1, Reduce.SUM)
            record_metric("game/average_turns", self.turn_count, Reduce.MEAN)
            record_metric("game/win_rate", 1 if result.reward > 0 else 0, Reduce.MEAN)
            record_metric("game/env_reward", result.reward, Reduce.MEAN)
        else:
            reward = 0.0  # No intermediate rewards

        # Build next observation (if game continues)
        if result.done:
            observation = {"role": "user", "content": ""}  # Empty, game ended
        else:
            obs_text = self._format_observation(result.observation)
            observation = {"role": "user", "content": obs_text}

        return EnvStepResult(
            observation=observation,
            reward=reward,
            done=result.done,
            metadata={
                "turn_count": self.turn_count,
                "has_invalid_action": self.has_invalid_action,
                "env_reward": result.reward if result.done else 0.0,
            },
        )

    def _format_observation(self, observation) -> str:
        """Format game observation into text."""
        player_total = observation.metadata.get("player_total", "?")
        dealer_card = observation.metadata.get("dealer_card", "?")
        dealer_str = "Ace" if dealer_card == 1 else str(dealer_card)

        return f"Hand: {player_total}, Dealer: {dealer_str}"

    def _parse_action(self, text: str) -> tuple[str, str]:
        """Parse action from assistant text using <answer> tags.

        Returns:
            (action, error_type): action is "HIT", "STAND", or "INVALID"
                                  error_type is "" for valid, "NO_TAGS" or "INVALID_CONTENT"
        """
        import re

        # Try to extract content from <answer> tags
        match = re.search(
            r"<answer>\s*(.*?)\s*</answer>", text, re.IGNORECASE | re.DOTALL
        )

        if match:
            answer = match.group(1).strip().upper()
            if answer == "HIT":
                return ("HIT", "")
            elif answer == "STAND":
                return ("STAND", "")
            else:
                # Has <answer> tags but invalid content
                return ("INVALID", "INVALID_CONTENT")
        else:
            # No <answer> tags found
            return ("INVALID", "NO_TAGS")

    def _compute_reward(self, env_reward: float, is_invalid: bool) -> float:
        """Compute final reward."""
        if env_reward > 0:  # Win
            rwd = 3.0
        else:  # Loss or push
            rwd = -1.0

        if is_invalid:
            rwd = -10.0  # Penalty for not ending with HIT/STAND
            record_metric("game/invalid_action_penalty", 1, Reduce.SUM)

        return rwd

    def close(self):
        """Clean up."""
        self.client.close()
