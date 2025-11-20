# Refactoring Proposal 07: Extract BlackjackEnv to Separate Module

## Overview
Building on Proposals 01-06, this iteration extracts the BlackjackEnv class and related environment code to a dedicated module, following the pattern from grpo/main.py where environment logic is separate.

## Key Changes

### 1. Create New Module for Blackjack Environment
Create `envs/blackjack_env/blackjack_env.py` to house all blackjack-specific logic.

**New file structure:**
```
envs/
├── openspiel_env/
│   ├── __init__.py
│   ├── server/
│   └── ...
└── blackjack_env/  (NEW)
    ├── __init__.py
    └── blackjack_env.py
```

**In envs/blackjack_env/blackjack_env.py:**
```python
"""Blackjack environment for RL training."""
import re
from dataclasses import dataclass, field
from typing import Any

from envs.openspiel_env import OpenSpielAction, OpenSpielEnv
from forge.observability.metrics import record_metric, Reduce


@dataclass
class EnvStepResult:
    """Result from environment step."""
    observation: dict[str, str]
    reward: float
    done: bool


class BlackjackEnv:
    """Blackjack environment wrapper.

    Responsibilities:
    - Manage game state via OpenSpielEnv
    - Parse actions from text (<answer> tags)
    - Compute rewards
    """

    def __init__(self, server_url: str):
        self.server_url = server_url
        self.client = OpenSpielEnv(base_url=server_url)
        self.client._http.trust_env = False
        self.turn_count = 0
        self.has_invalid_action = False

    def reset(self) -> str:
        """Reset game and return initial observation text."""
        self.turn_count = 0
        self.has_invalid_action = False
        result = self.client.reset()
        return self._format_obs(result.observation)

    def step(self, action_text: str) -> EnvStepResult:
        """Execute action and return next observation."""
        # Parse and execute action
        action = self._parse_action(action_text)
        if action == "INVALID":
            self.has_invalid_action = True
            action = "STAND"
            record_metric("game/invalid_actions", 1, Reduce.SUM)

        action_id = 0 if action == "HIT" else 1
        result = self.client.step(
            OpenSpielAction(action_id=action_id, game_name="blackjack")
        )
        self.turn_count += 1

        # Compute reward
        if result.done:
            reward = self._compute_reward(result.reward, self.has_invalid_action)
            record_metric("game/win_rate", 1 if result.reward > 0 else 0, Reduce.MEAN)
        else:
            reward = 0.0

        obs = {"role": "user", "content": ""} if result.done else {
            "role": "user",
            "content": self._format_obs(result.observation)
        }

        return EnvStepResult(observation=obs, reward=reward, done=result.done)

    def close(self):
        """Clean up."""
        self.client.close()

    def _format_obs(self, obs) -> str:
        """Format game state as text."""
        player = obs.metadata.get("player_total", "?")
        dealer = obs.metadata.get("dealer_card", "?")
        dealer = "Ace" if dealer == 1 else str(dealer)
        return f"Hand: {player}, Dealer: {dealer}"

    def _parse_action(self, text: str) -> str:
        """Extract action from <answer> tags. Returns HIT, STAND, or INVALID."""
        match = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.IGNORECASE | re.DOTALL)
        if match:
            answer = match.group(1).strip().upper()
            return answer if answer in ["HIT", "STAND"] else "INVALID"
        return "INVALID"

    def _compute_reward(self, env_reward: float, has_invalid: bool) -> float:
        """Compute final reward with penalty for invalid actions."""
        base_reward = 3.0 if env_reward > 0 else -1.0
        penalty = -10.0 if has_invalid else 0.0
        return base_reward + penalty
```

**In envs/blackjack_env/__init__.py:**
```python
from .blackjack_env import BlackjackEnv, EnvStepResult

__all__ = ["BlackjackEnv", "EnvStepResult"]
```

**In main_v2.py:**
```python
from envs.blackjack_env import BlackjackEnv, EnvStepResult
```

### 2. Extract System Prompt to Config
The system prompt (lines 1698-1720) should be in the config, not hardcoded.

**In qwen3_1_7b.yaml:**
```yaml
blackjack_env:
  game_name: "blackjack"
  server_port: 8000
  max_seq_len: 2048
  max_turns: 20
  system_prompt: |
    You are an expert Blackjack player.

    GOAL: Get a hand total closer to 21 than the dealer without going over 21 (busting).

    RULES:
    - Card values: Ace=1 or 11, Face cards (J,Q,K)=10, Number cards=face value
    - If you go over 21, you bust and lose immediately
    - The dealer plays after you and must hit until reaching 17+

    ACTIONS:
    - HIT: Take another card (increases your hand total)
    - STAND: Keep your current hand and end your turn

    WIN CONDITIONS:
    - Your hand is closer to 21 than the dealer's final hand
    - Dealer busts (goes over 21) and you don't
    - You get exactly 21

    IMPORTANT: You MUST output your action in the following format:
    <answer>HIT</answer> or <answer>STAND</answer>
```

**In main_v2.py:**
```python
# In continuous_rollouts():
initial_messages = [
    {"role": "system", "content": cfg.blackjack_env.system_prompt}
]
```

### 3. Create Rollout Utilities Module
Extract `do_single_rollout` and `do_group_rollout` to `apps/blackjack/rollout.py`.

**In apps/blackjack/rollout.py:**
```python
"""Rollout utilities for Blackjack GRPO training."""
import uuid
import torch
from envs.blackjack_env import BlackjackEnv
from forge.data.token_accumulator import TokenAccumulator, ValidationMode
from forge.observability.metrics import record_metric, Reduce
from vllm import SamplingParams


async def do_single_rollout(
    env: BlackjackEnv,
    policy,
    tokenizer,
    max_seq_len: int,
    max_turns: int,
    messages: list[dict],
    game_id: str | None = None,
) -> Episode:
    """Play one game and return one Episode."""
    # ... (full implementation)
```

**In main_v2.py:**
```python
from apps.blackjack.rollout import do_single_rollout, do_group_rollout
```

### 4. Simplify Main File Structure
With extractions, main_v2.py should have clear sections:

```python
# main_v2.py structure after extractions:

# Imports
# Episode dataclass
# ComputeAdvantages actor
# Loss function
# Utility functions (drop_weights, etc.)
# Main training loop (main function)
```

## Impact
- **Main file:** ~900 lines → ~400 lines (55% reduction from Proposal 05)
- **Modularity:** Environment, rollout, and token accumulation are separate, testable modules
- **Reusability:** BlackjackEnv can be used in other scripts
- **Configuration:** System prompt is configurable, not hardcoded
- **Code organization:** Much clearer separation of concerns
- **Risk:** Low - pure code movement, clear module boundaries
