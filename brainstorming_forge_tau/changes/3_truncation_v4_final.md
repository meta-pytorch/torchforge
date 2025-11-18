# Truncation V4: Complete Implementation (No Tinker Imports)

**Date:** 2025-01-16
**Purpose:** Complete, concrete implementation of blackjack with proper abstractions. No Tinker imports, all classes defined once.

---

## Architecture Overview

```
continuous_rollouts() (while True loop)
    ↓
do_group_rollout(envs: list[BlackjackEnv], policy)
    ↓
    ├─ do_single_rollout(env[0], policy) → Episode
    ├─ do_single_rollout(env[1], policy) → Episode
    ├─ ...
    └─ do_single_rollout(env[N], policy) → Episode
    ↓
Returns list[Episode]
```

**Key insight:** We create N env instances upfront, then pass `env[i]` to each parallel rollout.

---

## Complete Implementation (Every Class, Start to Finish)

### File 1: `apps/blackjack/types.py` - Core Types

```python
"""
Core types for blackjack RL training.
No external dependencies except dataclasses and torch.
"""

from dataclasses import dataclass, field
from typing import Any
import torch


@dataclass
class Episode:
    """
    Episode data for GRPO training with multi-turn support.

    For blackjack:
        - all_token_ids: [prompt1, resp1, prompt2, resp2, ...]
        - response_mask: [0, 0, ..., 1, 1, ..., 0, 0, ..., 1, 1, ...]
        - reward: Final game outcome (win/loss)

    One episode = one complete game with all turns.
    """

    # ============ Core Identifiers ============
    episode_id: str
    task_name: str = "blackjack"

    # ============ Policy Version ============
    generator_version: int = 0
    is_truncated: bool = False

    # ============ Token Data ============
    all_token_ids: torch.Tensor  # Shape: (seq_len,)
    logprobs: torch.Tensor       # Shape: (seq_len,)
    response_mask: torch.Tensor  # Shape: (seq_len,)
                                 # 1.0 = train on this token (response)
                                 # 0.0 = skip this token (prompt)

    # ============ Rewards & Training ============
    reward: float
    advantage: float | None = None
    ref_logprobs: torch.Tensor | None = None  # Shape: (seq_len,)

    # ============ Metadata ============
    metadata: dict[str, Any] = field(default_factory=dict)
    message_log: list[dict[str, Any]] | None = None


@dataclass
class GameState:
    """Observation from blackjack game."""
    player_total: int
    dealer_card: int
    done: bool
    reward: float


# Type alias for GRPO groups
Group = list[Episode]
```

---

### File 2: `apps/blackjack/env.py` - Environment

```python
"""
BlackjackEnv: Manages game state, prompt building, and reward computation.

This wraps OpenSpielEnv to control the data flow and prompt format.
"""

from __future__ import annotations
import asyncio
from typing import Any

from apps.blackjack.types import GameState
from forge.openenv.clients.openspiel_env import OpenSpielEnv, OpenSpielAction


class BlackjackEnv:
    """
    Blackjack environment for RL training.

    Responsibilities:
    - Manage game state via OpenSpielEnv
    - Build conversation messages (user/assistant)
    - Format prompts using tokenizer.apply_chat_template
    - Parse actions from assistant text
    - Compute rewards
    - Track budget and truncation

    Does NOT handle:
    - Policy generation (caller does this)
    - Reference model computation (caller does this)
    - Advantage computation (caller does this)
    """

    def __init__(
        self,
        server_url: str,
        tokenizer,
        system_prompt: str,
        max_seq_len: int = 2048,
        max_turns: int = 10,
    ):
        """
        Args:
            server_url: OpenSpiel server URL (e.g., "http://localhost:8004")
            tokenizer: HuggingFace tokenizer with apply_chat_template
            system_prompt: System message for the game
            max_seq_len: Maximum total tokens across all turns
            max_turns: Maximum number of game turns
        """
        self.server_url = server_url
        self.tokenizer = tokenizer
        self.system_prompt = system_prompt
        self.max_seq_len = max_seq_len
        self.max_turns = max_turns

        # Game client
        self.client = OpenSpielEnv(base_url=server_url)
        self.client._http.trust_env = False

        # Episode state (reset on each game)
        self.messages: list[dict[str, str]] = []
        self.cumulative_tokens = 0
        self.turn_count = 0
        self.has_invalid_action = False

    def reset(self) -> tuple[str, int]:
        """
        Reset environment for new game.

        Returns:
            prompt: Formatted prompt string
            remaining_tokens: Budget remaining for first generation
        """
        # Reset episode state
        self.messages = []
        self.cumulative_tokens = 0
        self.turn_count = 0
        self.has_invalid_action = False

        # Add system message
        if self.system_prompt:
            self.messages.append({"role": "system", "content": self.system_prompt})

        # Reset game
        result = self.client.reset()

        # Build first user message
        user_message = self._format_game_state(
            player_total=result.observation.metadata.get("player_total", "?"),
            dealer_card=result.observation.metadata.get("dealer_card", "?"),
        )
        self.messages.append({"role": "user", "content": user_message})

        # Format prompt
        prompt = self.tokenizer.apply_chat_template(
            self.messages,
            add_generation_prompt=True,
            tokenize=False
        )

        # Track tokens
        prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
        self.cumulative_tokens = len(prompt_tokens)

        # Calculate remaining budget
        remaining = self.max_seq_len - self.cumulative_tokens

        return prompt, remaining

    def step(
        self,
        response_text: str,
        response_token_ids: list[int],
        response_logprobs: list[float],
    ) -> tuple[GameState | None, str | None, int | None]:
        """
        Execute one turn of the game.

        Args:
            response_text: Assistant's text response
            response_token_ids: Token IDs of response
            response_logprobs: Log probabilities of response tokens

        Returns:
            (game_state, next_prompt, remaining_budget) if continuing
            (game_state, None, None) if game ended
            Where game_state contains: player_total, dealer_card, done, reward
        """
        # Update cumulative tokens
        self.cumulative_tokens += len(response_token_ids)

        # Add assistant message to history
        self.messages.append({"role": "assistant", "content": response_text})

        # Parse action
        action_name = self._parse_action(response_text)
        if action_name == "INVALID":
            self.has_invalid_action = True
            action_name = "STAND"  # Fallback

        # Execute action in game
        action_id = 0 if action_name == "HIT" else 1
        result = self.client.step(
            OpenSpielAction(action_id=action_id, game_name="blackjack")
        )

        self.turn_count += 1

        # Build game state
        game_state = GameState(
            player_total=result.observation.metadata.get("player_total", 0),
            dealer_card=result.observation.metadata.get("dealer_card", 0),
            done=result.done,
            reward=result.reward,
        )

        # Check if game ended
        if result.done:
            return game_state, None, None

        # Check if hit max turns
        if self.turn_count >= self.max_turns:
            game_state.done = True
            return game_state, None, None

        # Game continues - build next prompt
        user_message = self._format_game_state(
            player_total=game_state.player_total,
            dealer_card=game_state.dealer_card,
        )
        self.messages.append({"role": "user", "content": user_message})

        # Format next prompt
        next_prompt = self.tokenizer.apply_chat_template(
            self.messages,
            add_generation_prompt=True,
            tokenize=False
        )

        # Track tokens
        prompt_tokens = self.tokenizer.encode(next_prompt, add_special_tokens=False)
        self.cumulative_tokens = len(prompt_tokens)

        # Calculate remaining budget
        remaining = self.max_seq_len - self.cumulative_tokens

        return game_state, next_prompt, remaining

    def compute_reward(self, game_state: GameState) -> float:
        """
        Compute final reward from game outcome.

        Args:
            game_state: Final game state

        Returns:
            Shaped reward for training
        """
        if game_state.reward > 0:  # Win
            return 3.0
        else:  # Loss or push
            return -1.0

    def get_metadata(self) -> dict[str, Any]:
        """Get episode metadata for logging."""
        return {
            "num_turns": self.turn_count,
            "has_invalid_action": self.has_invalid_action,
            "cumulative_tokens": self.cumulative_tokens,
        }

    def _format_game_state(self, player_total: int, dealer_card: int) -> str:
        """Format game state into user message."""
        dealer_str = "Ace" if dealer_card == 1 else str(dealer_card)

        return (
            f"=== BlackJack Game (Turn {self.turn_count + 1}) ===\n\n"
            f"Current State:\n"
            f"  Your hand total: {player_total}\n"
            f"  Dealer shows: {dealer_str}\n"
            f"  Legal actions: HIT, STAND\n\n"
            f"What do you do? Output only 'HIT' or 'STAND'."
        )

    def _parse_action(self, text: str) -> str:
        """Parse action from assistant text."""
        text_lower = text.lower().strip()
        if text_lower.endswith("hit"):
            return "HIT"
        elif text_lower.endswith("stand"):
            return "STAND"
        else:
            return "INVALID"

    def close(self):
        """Clean up resources."""
        self.client.close()
```

---

### File 3: `apps/blackjack/rollouts.py` - Rollout Functions

```python
"""
Rollout functions for blackjack RL training.

These are generic - they work with any environment that follows the pattern:
    env.reset() → (prompt, remaining_budget)
    env.step(text, tokens, logprobs) → (game_state, next_prompt, remaining_budget)
"""

import asyncio
import uuid
import torch
from typing import Any

from apps.blackjack.types import Episode
from apps.blackjack.env import BlackjackEnv


async def do_single_rollout(
    env: BlackjackEnv,
    policy,
    game_id: str | None = None,
) -> Episode:
    """
    Play one game and return one Episode.

    Args:
        env: BlackjackEnv instance
        policy: Policy with .generate.route() method
        game_id: Optional game ID for logging

    Returns:
        Episode with all turns concatenated
    """
    if game_id is None:
        game_id = str(uuid.uuid4())

    # Accumulators for episode data
    all_tokens: list[int] = []
    all_logprobs: list[float] = []
    response_mask: list[int] = []

    # Truncation tracking
    is_truncated = False
    truncation_reason: str | None = None

    try:
        # ============ Reset environment ============
        prompt, remaining = env.reset()

        # Tokenize initial prompt
        prompt_tokens = env.tokenizer.encode(prompt, add_special_tokens=False)

        # Check if initial prompt exceeds budget (edge case)
        if remaining <= 0:
            is_truncated = True
            truncation_reason = "initial_prompt_exceeds_budget"
            # Return minimal episode
            return Episode(
                episode_id=game_id,
                generator_version=0,
                is_truncated=True,
                all_token_ids=torch.tensor(prompt_tokens[:env.max_seq_len], dtype=torch.long),
                logprobs=torch.zeros(min(len(prompt_tokens), env.max_seq_len)),
                response_mask=torch.zeros(min(len(prompt_tokens), env.max_seq_len)),
                reward=0.0,
                metadata={"truncation_reason": truncation_reason, "num_turns": 0},
            )

        # ============ Multi-turn loop ============
        game_state = None
        turn_num = 0

        while True:
            # Tokenize current prompt
            prompt_tokens = env.tokenizer.encode(prompt, add_special_tokens=False)

            # Check budget before generation
            if remaining <= 0:
                is_truncated = True
                truncation_reason = "max_seq_len"
                break

            # ============ Generate response ============
            responses = await policy.generate.route(
                [prompt],
                sampling_params={"max_tokens": remaining}
            )
            response = responses[0]

            # Check if generation was truncated
            if response.stop_reason == "length":
                is_truncated = True
                truncation_reason = "generation_length"
                # Add tokens but break after this turn
                all_tokens.extend(prompt_tokens)
                all_tokens.extend(response.token_ids)
                response_mask.extend([0] * len(prompt_tokens))
                response_mask.extend([1] * len(response.token_ids))
                all_logprobs.extend([0.0] * len(prompt_tokens))
                all_logprobs.extend(response.logprobs)
                break

            # ============ Accumulate tokens ============
            all_tokens.extend(prompt_tokens)
            all_tokens.extend(response.token_ids)
            response_mask.extend([0] * len(prompt_tokens))  # Don't train on prompts
            response_mask.extend([1] * len(response.token_ids))  # Train on responses
            all_logprobs.extend([0.0] * len(prompt_tokens))
            all_logprobs.extend(response.logprobs)

            # ============ Step environment ============
            game_state, next_prompt, next_remaining = env.step(
                response_text=response.text,
                response_token_ids=response.token_ids,
                response_logprobs=response.logprobs,
            )

            turn_num += 1

            # Check if game ended
            if game_state.done or next_prompt is None:
                break

            # Check if hit max turns
            if turn_num >= env.max_turns:
                is_truncated = True
                truncation_reason = "max_turns"
                break

            # Continue to next turn
            prompt = next_prompt
            remaining = next_remaining

        # ============ Compute final reward ============
        if game_state is not None:
            reward = env.compute_reward(game_state)
        else:
            reward = 0.0  # Truncated before first turn completed

        # ============ Create episode ============
        episode = Episode(
            episode_id=game_id,
            task_name="blackjack",
            generator_version=response.generator_version if 'response' in locals() else 0,
            is_truncated=is_truncated,
            all_token_ids=torch.tensor(all_tokens, dtype=torch.long),
            logprobs=torch.tensor(all_logprobs, dtype=torch.float),
            response_mask=torch.tensor(response_mask, dtype=torch.float),
            reward=reward,
            advantage=None,  # Computed later
            ref_logprobs=None,  # Computed later
            message_log=env.messages.copy(),
            metadata={
                **env.get_metadata(),
                "truncation_reason": truncation_reason,
                "env_reward": game_state.reward if game_state else 0.0,
            }
        )

        return episode

    finally:
        env.close()


async def do_group_rollout(
    envs: list[BlackjackEnv],
    policy,
) -> list[Episode]:
    """
    Rollout multiple games in parallel.

    Args:
        envs: List of BlackjackEnv instances (one per game)
        policy: Policy for generation

    Returns:
        List of Episodes (one per env)
    """
    # Create tasks for parallel execution
    # Each task gets its own env from the list
    tasks = [
        do_single_rollout(
            env=envs[i],
            policy=policy,
            game_id=f"game_{i}_{uuid.uuid4().hex[:8]}",
        )
        for i in range(len(envs))
    ]

    # Execute in parallel
    episodes = await asyncio.gather(*tasks)

    return list(episodes)
```

---

### File 4: `apps/blackjack/main.py` - Main Training Loop (Updated)

```python
"""
Main training loop for blackjack with complete implementation.
"""

import asyncio
import uuid
import torch
import torch.nn.functional as F
from omegaconf import DictConfig

from apps.blackjack.types import Episode, Group
from apps.blackjack.env import BlackjackEnv
from apps.blackjack.rollouts import do_group_rollout
from forge.metrics import record_metric, Reduce


async def continuous_rollouts(
    cfg: DictConfig,
    policy,
    ref_model,
    compute_advantages,
    replay_buffer,
    tokenizer,
    pad_id: int,
):
    """
    Main GRPO rollout loop.

    Flow:
    1. Create N environments
    2. Rollout group in parallel → list[Episode]
    3. Filter groups (constant rewards)
    4. Compute ref_model for valid group
    5. Compute advantages
    6. Episode-level acceptance
    7. Add to replay buffer
    8. Repeat
    """

    # Extract config
    server_url = cfg.blackjack_env.server_url
    max_seq_len = cfg.blackjack_env.max_seq_len
    max_turns = cfg.blackjack_env.max_turns
    group_size = cfg.grpo.group_size
    system_prompt = "You are an expert BlackJack player. Analyze the game state and output only 'HIT' or 'STAND'."

    rollout_count = 0

    # ============ Main loop ============
    while True:  # User asked: why shutdown_event? Answer: Just use while True!

        # ============ Step 1: Create N environments ============
        envs = [
            BlackjackEnv(
                server_url=server_url,
                tokenizer=tokenizer,
                system_prompt=system_prompt,
                max_seq_len=max_seq_len,
                max_turns=max_turns,
            )
            for _ in range(group_size)
        ]

        # ============ Step 2: Rollout group in parallel ============
        episodes = await do_group_rollout(envs, policy)

        # ============ Step 3: Filter groups (constant rewards) ============
        rewards = [e.reward for e in episodes]
        if len(set(rewards)) == 1:
            # All rewards identical - no learning signal
            record_metric("groups/rate_dropped", 1, Reduce.MEAN)
            rollout_count += 1
            continue

        record_metric("groups/rate_dropped", 0, Reduce.MEAN)

        # ============ Step 4: Compute ref_model ============
        # Pad episodes to same length for batching
        max_len = max(len(e.all_token_ids) for e in episodes)
        padded_tokens = []
        for episode in episodes:
            seq_len = len(episode.all_token_ids)
            pad_len = max_len - seq_len
            padded = F.pad(episode.all_token_ids, (0, pad_len), value=pad_id)
            padded_tokens.append(padded)

        input_ids = torch.stack(padded_tokens)  # [group_size, max_len]

        # Get reference logprobs (padded)
        ref_logprobs_padded = await ref_model.forward.route(
            input_ids,
            0,  # No separate prompt length (response_mask handles it)
            return_logprobs=True
        )

        # Assign ref_logprobs to episodes (UNPAD)
        for i, episode in enumerate(episodes):
            seq_len = len(episode.all_token_ids)
            episode.ref_logprobs = ref_logprobs_padded[i, :seq_len]  # Remove padding

        del ref_logprobs_padded, input_ids

        # ============ Step 5: Compute advantages ============
        advantages = await compute_advantages.compute.call_one(episodes)
        for episode, advantage in zip(episodes, advantages):
            episode.advantage = advantage

        # ============ Step 6: Episode-level acceptance ============
        accepted_episodes = []
        for episode in episodes:
            should_accept = True

            # Acceptance criterion: is_truncated
            if episode.is_truncated and not cfg.grpo.get("accept_truncated", True):
                should_accept = False
                record_metric("buffer/rate_rejected_truncated", 1, Reduce.MEAN)
            else:
                record_metric("buffer/rate_rejected_truncated", 0, Reduce.MEAN)

            # Future: Add min_advantage criterion here

            if should_accept:
                accepted_episodes.append(episode)

        # ============ Step 7: Add to replay buffer ============
        # TODO: Add all episodes at once instead of one by one
        for episode in accepted_episodes:
            await replay_buffer.add.call_one(episode)

        # Metrics
        record_metric("buffer/episodes_accepted", len(accepted_episodes), Reduce.SUM)
        record_metric("buffer/episodes_generated", len(episodes), Reduce.SUM)
        record_metric("main/rollout_iterations", 1, Reduce.SUM)

        rollout_count += 1


# ============ Update main() to use new rollout ============

async def main(cfg: DictConfig):
    """Main entry point."""

    # ... existing service initialization ...

    # ============ Get tokenizer ============
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.policy.model)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    # ============ Start rollout tasks ============
    num_rollout_threads = cfg.main.get("num_rollout_threads", 1)

    rollout_tasks = [
        asyncio.create_task(
            continuous_rollouts(
                cfg=cfg,
                policy=policy,
                ref_model=ref_model,
                compute_advantages=compute_advantages,
                replay_buffer=replay_buffer,
                tokenizer=tokenizer,
                pad_id=pad_id,
            )
        )
        for _ in range(num_rollout_threads)
    ]

    # ... rest of main ...
```

---

## Complete Flow Diagram

```
continuous_rollouts():
│
├─ Create N BlackjackEnv instances
│   env[0] = BlackjackEnv(server_url, tokenizer, system_prompt, ...)
│   env[1] = BlackjackEnv(...)
│   ...
│   env[N-1] = BlackjackEnv(...)
│
├─ do_group_rollout(envs, policy)
│   │
│   ├─ Launch parallel tasks:
│   │   ├─ asyncio.create_task(do_single_rollout(env[0], policy))
│   │   ├─ asyncio.create_task(do_single_rollout(env[1], policy))
│   │   └─ ...
│   │
│   └─ await asyncio.gather(*tasks) → list[Episode]
│       │
│       └─ Each do_single_rollout():
│           │
│           ├─ prompt, remaining = env.reset()
│           │   └─ env builds messages: [system, user]
│           │   └─ env.tokenizer.apply_chat_template(messages)
│           │
│           ├─ while True:
│           │   ├─ response = await policy.generate(prompt, max_tokens=remaining)
│           │   ├─ Accumulate: all_tokens, all_logprobs, response_mask
│           │   ├─ game_state, next_prompt, next_remaining = env.step(response)
│           │   │   └─ env parses action from response.text
│           │   │   └─ env calls OpenSpielEnv.step(action)
│           │   │   └─ env builds next user message
│           │   │   └─ env.tokenizer.apply_chat_template(messages)
│           │   └─ if game_state.done: break
│           │
│           └─ return Episode(all_tokens, response_mask, reward, ...)
│
├─ Filter: if len(set(rewards)) == 1: continue
│
├─ Compute ref_model (pad → forward → unpad)
│
├─ Compute advantages
│
├─ Episode-level acceptance (truncated filter)
│
└─ Add accepted episodes to replay buffer
```

---

## How do_group_rollout Works (Step by Step)

**Question:** "How does rollout i have access to env i?"

**Answer:** We pass the entire `envs` list to `do_group_rollout()`, then inside that function we create tasks using `envs[i]`:

```python
async def do_group_rollout(
    envs: list[BlackjackEnv],  # ← List of N envs passed in
    policy,
) -> list[Episode]:

    # Create N tasks, each using envs[i]
    tasks = [
        do_single_rollout(
            env=envs[i],  # ← Task i gets env i
            policy=policy,
            game_id=f"game_{i}_...",
        )
        for i in range(len(envs))
    ]

    # Execute all tasks in parallel
    episodes = await asyncio.gather(*tasks)

    return list(episodes)
```

**Flow:**
1. `continuous_rollouts()` creates list of N envs
2. Passes entire list to `do_group_rollout(envs, policy)`
3. `do_group_rollout()` creates N tasks, each with `envs[i]`
4. `asyncio.gather()` runs all N tasks in parallel
5. Each task calls `do_single_rollout(env[i], policy)`
6. Returns list of N episodes

---

## Why `while True` instead of `while not shutdown_event.is_set()`?

**Answer:** You're right - we should just use `while True`! The shutdown will be handled by task cancellation when the program exits. Updated in the code above.

---

## Config Schema

```yaml
blackjack_env:
  server_url: "http://localhost:8004"
  max_seq_len: 2048
  max_turns: 10

grpo:
  group_size: 16
  accept_truncated: true

policy:
  model: "Qwen/Qwen2.5-1.5B-Instruct"
  engine_args:
    enable_prefix_caching: true
    max_model_len: 4096

main:
  num_rollout_threads: 1
```

---

## Summary of Changes from V3

### Removed
- ❌ All Tinker imports
- ❌ Tinker ABCs (Env, EnvGroupBuilder, etc.)
- ❌ Renderer abstraction (just use `tokenizer.apply_chat_template`)
- ❌ Initial prompt check before while loop
- ❌ `shutdown_event` (use `while True`)
- ❌ Redundant class definitions

### Added
- ✅ Complete `BlackjackEnv` class (defined once)
- ✅ Complete `do_single_rollout()` function
- ✅ Complete `do_group_rollout()` function
- ✅ Complete `continuous_rollouts()` function
- ✅ Clear explanation of how env[i] is passed to rollout i
- ✅ `generator_version` from `response.generator_version`

### Key Design
- **No ABCs** - Just concrete classes (battle test first, abstract later)
- **No Tinker** - Self-contained implementation
- **tokenizer.apply_chat_template** - Instead of Renderer
- **OpenEnv inside BlackjackEnv** - We control the data flow
- **Explicit env list** - Create N envs, pass to do_group_rollout

---

**End of Document**
