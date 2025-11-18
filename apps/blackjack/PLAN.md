# Blackjack Multi-Turn Refactor Plan

## Context

### Initial Requirements
From the user:
> Currently the evaluate_response and playgame are a mess. A lot of places are parsing the output. It doesn't make any sense.
>
> Also, what I am seeing is that we are giving the reward we want, but the reward should come from the env.
>
> We need to clean up the file. I guess in our case we want to change the reward to something like this:
> - We win, then reward is 3
> - We play and lose, then reward is 1
> - We don't have Hit or Stand, then reward is -1
>
> But we need to get this reward per interaction, which leads to the next issue: The way that it's currently implemented is not really multiturn. Multiturn would be:
> ```
> A: Hit,
> tool: 7
> A: Hit,
> tool: 14
> ```
> but we are not ready for it, so don't worry about it. We will get there.

### Architecture Alignment
This plan now aligns with Forge's broader multi-turn tool calling architecture:
- **Message format** (from `1_message_format_for_tool_calling.md`): Dataset returns messages, formatting happens in rollout loop
- **Episode class** (from `2_episode_class.md`): New Episode with response_mask, all_token_ids, logprobs
- **Truncation** (from `3_truncation.md`): Episode-level budget tracking with max_seq_len

---

## The Core Problem

**Current implementation has a fundamental learning bug**: All steps in a game get the SAME final reward.

Example:
```python
# Game: HIT (15→18), HIT (18→20), STAND (20) → WIN (+1)
# Current: All 3 steps get reward +3
# Problem: Can't distinguish good HITs from bad HITs!

# Game: HIT (15→18), HIT (18→23) → BUST (-1)
# Current: All 2 steps get reward -1
# Problem: First HIT was good! Second HIT was bad!
```

**Root cause**: We create ONE episode per step instead of ONE episode per game with all turns concatenated.

**Solution**: Multi-turn episode where:
- ONE episode per game (not per step)
- All turns concatenated into single sequence
- Response mask marks which tokens to train on (critical for future tool calling)
- Single final reward applies to entire sequence

This architecture works for both:
- **Blackjack now**: Multiple game steps (HIT/STAND) in one episode
- **Tool calling later**: Multiple LLM + tool interactions in one episode

---

## Architecture Overview

### Current (Broken)
```python
# play_game() returns multiple step_results
# continuous_rollouts() creates one Episode per step
for step_result in all_step_results:
    episode = Episode(...)  # Same game_id, same final_reward
    episodes.append(episode)
```

### New (Fixed)
```python
# Dataset returns structured messages (not formatted strings)
sample = await dataloader.sample.call_one()
messages = sample["messages"]  # List of message dicts

# play_game() formats messages each turn, returns ONE episode per game
episode = await play_game(
    messages=messages,  # Initial messages from dataset
    tokenizer=tokenizer,  # Passed from main
    max_seq_len=2048,   # Episode-level budget
    ...
)

# Episode contains all turns concatenated
episode = Episode(
    all_token_ids=[prompt1, resp1, prompt2, resp2, ...],
    response_mask=[0, 0, 1, 1, 0, 0, 1, 1, ...],  # 0=prompt, 1=response
    logprobs=[0, 0, logp1, logp2, 0, 0, logp3, ...],
    reward=final_game_reward
)
```

---

## Key Changes from Current Code

### 1. Message Format Changes
**From `1_message_format_for_tool_calling.md`:**

| Component | Current | New |
|-----------|---------|-----|
| **Dataset** | Returns formatted string from `apply_chat_template()` | Returns `{"messages": [...], "target": ...}` |
| **Rollout Loop** | Receives string, passes to generator | Formats messages with `tokenizer.apply_chat_template()` each turn |
| **Generator** | Receives string | Unchanged - still receives string |
| **Tokenizer location** | Not available in rollout | Passed from main → rollout loop → play_game |

**Why**: Need message structure to add game state each turn and prepare for tool calling.

### 2. Episode Class Changes
**From `2_episode_class.md`:**

| Field | Current | New | Why |
|-------|---------|-----|-----|
| `pad_id, request_len, response_len` | ✅ Used | ❌ Removed | Workarounds for missing response_mask |
| `response_mask` | ❌ Missing | ✅ Required | Marks which tokens to train on |
| `all_token_ids` | ❌ Missing | ✅ Required | Concatenated tokens from all turns |
| `logprobs` | ❌ Missing | ✅ Required | Log probabilities for all tokens |
| `completion` | ✅ Stores full object | ❌ Removed | Memory waste, just extract needed fields |
| `generator_version` | From `completion` | ✅ First-class field | Critical for replay buffer eviction |
| `is_truncated` | ❌ Missing | ✅ First-class field | Mark incomplete episodes |
| `message_log` | ❌ Missing | ✅ Optional | Store conversation for debugging |

### 3. Truncation Strategy
**From `3_truncation.md`:**

- **Episode-level budget**: `max_seq_len=2048` (covers all turns)
- **Per-turn checks**: Before each generation, check if `len(prompt_tokens) >= max_seq_len`
- **Dynamic max_tokens**: `max_tokens = max_seq_len - len(prompt_tokens)`
- **Mid-generation truncation**: Stop if `response.stop_reason == "length"`
- **Prefix caching**: Enable for 2-3x speedup on multi-turn prompts

---

## Implementation Steps

### Goals
1. ONE function that parses model output (no scattered parsing)
2. Use environment reward as base with custom penalties for invalid actions
3. Create ONE episode per game with all turns concatenated
4. Add response_mask to prevent training on prompts
5. Format messages in rollout loop (not dataset)
6. Episode-level budget tracking with max_seq_len
7. Collate function handles variable-length episodes

---

### Step 1: Create New Episode Class

**File**: `apps/blackjack/episode.py` (new file)

**Based on `2_episode_class.md`:**

```python
from dataclasses import dataclass, field
from typing import Any
import torch


@dataclass
class Episode:
    """
    Episode data for GRPO training with multi-turn support.

    For blackjack (multi-turn game, single episode):
        - all_token_ids: [prompt1, resp1, prompt2, resp2, ...]
        - response_mask: [0, 0, ..., 1, 1, ..., 0, 0, ..., 1, 1, ...]
                         [  prompt1  ][  resp1  ][  prompt2  ][  resp2  ]
        - reward: Final game outcome (win/loss/push)

    One episode = one complete game with all turns.
    """

    # ============ Core Identifiers ============
    episode_id: str
    task_name: str | None = None  # e.g., "blackjack"

    # ============ Policy Version (for replay buffer eviction) ============
    generator_version: int = 0
    is_truncated: bool = False  # Hit max_seq_len or max_turns

    # ============ Token Data ============
    all_token_ids: torch.Tensor  # Shape: (seq_len,)
    logprobs: torch.Tensor       # Shape: (seq_len,)
    response_mask: torch.Tensor  # Shape: (seq_len,)
                                 # 1.0 = train on this token (response)
                                 # 0.0 = skip this token (prompt)

    # ============ Rewards & Training ============
    reward: float | None = None
    advantage: float | None = None
    ref_logprobs: torch.Tensor | None = None  # Shape: (seq_len,)

    # ============ Metadata ============
    metadata: dict[str, Any] = field(default_factory=dict)
    # Suggested fields:
    #   - num_turns: int
    #   - game_id: str
    #   - env_reward: float (raw from environment)
    #   - has_invalid_action: bool
    #   - truncation_reason: str ("max_seq_len", "max_turns", "generation_length", None)

    # ============ Optional Debugging ============
    message_log: list[dict[str, Any]] | None = None
    # OpenAI-compatible messages for debugging/analysis

# Type alias for GRPO groups
Group = list[Episode]
```

**Key differences from current Episode (main.py:80-122)**:
- ❌ Remove: `pad_id`, `request_len`, `response_len`, `completion`
- ✅ Add: `all_token_ids`, `logprobs`, `response_mask`, `is_truncated`, `message_log`
- ✅ Move: `generator_version` from `completion` to first-class field

---

### Step 2: Create Unified Parser

**File**: `apps/blackjack/main.py`

```python
def parse_action(response_text: str) -> str:
    """
    Parse action from model's text response.

    Returns:
        "HIT", "STAND", or "INVALID"

    Note:
        INVALID actions default to STAND in play_game() but are penalized
        in the reward function (-1 regardless of game outcome).
    """
    text_lower = response_text.lower().strip()

    if text_lower.endswith("hit"):
        return "HIT"
    elif text_lower.endswith("stand"):
        return "STAND"
    else:
        return "INVALID"
```

**Replace**: Current `parse_action()` at main.py:244-256

---

### Step 3: Create Reward Calculation Function

**File**: `apps/blackjack/main.py`

```python
def calculate_reward(
    env_reward: float,
) -> float:
    """
    Reward structure:
        - Win: +3
        - Else: -1

    Args:
        env_reward: Raw environment reward (+1 win, 0 push, -1 loss)

    Returns:
        Final shaped reward for training
    """

    # Custom reward shaping based on game outcome
    if env_reward > 0:  # Win
        return 3.0
    else:  # Loss
        return -1.0
```

**Add metrics**:
```python
record_metric("reward/env_reward", env_reward, Reduce.MEAN)
record_metric("reward/final_reward", reward, Reduce.MEAN)
record_metric("reward/invalid_action_rate", 1 if has_invalid_action else 0, Reduce.MEAN)
```

**Delete**: `BlackJackReward` actor (main.py:258-302)

---

### Step 4: Get Tokenizer in main()

**File**: `apps/blackjack/main.py`

**Add after service initialization** (after line 659):

```python
# Get tokenizer for rollout loop
from vllm.transformers_utils.tokenizer import get_tokenizer
tokenizer = get_tokenizer(cfg.policy.get("model"))
pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
```

**Update continuous_rollouts signature**:
```python
async def continuous_rollouts(tokenizer, pad_id):  # Add parameters
```

**Pass to tasks** (main.py:838-840):
```python
rollout_tasks = [
    asyncio.create_task(continuous_rollouts(tokenizer, pad_id))
    for _ in range(num_rollout_threads)
]
```

---

### Step 5: Refactor play_game() for Multi-Turn

**File**: `apps/blackjack/main.py`

**Replace current play_game()** (main.py:359-557) with:

```python
async def play_game(
    game_idx: int,
    game_id: str,
    server_url: str,
    policy: Generator,
    tokenizer,
    pad_id: int,
    max_seq_len: int = 2048,
    max_turns: int = 10,
    rollout_count: int = 0,
) -> Episode:
    """
    Play a single blackjack game and return ONE episode with all turns.

    Key changes:
    - Formats messages each turn (not once at start)
    - Tracks episode-level budget (max_seq_len)
    - Returns single Episode with concatenated tokens
    - Includes response_mask for training

    Returns:
        Episode with all turns concatenated
    """
    env = OpenSpielEnv(base_url=server_url)
    env._http.trust_env = False

    print(f"\n🎮 GAME {game_idx + 1} (Rollout #{rollout_count + 1}) - ID: {game_id}")

    # Initialize message history
    messages = [
        {"role": "system", "content": "You are an expert BlackJack player. Analyze the game state and output only 'HIT' or 'STAND'."}
    ]

    # Track all tokens and masks across all turns
    all_tokens = []
    all_logprobs = []
    response_mask = []

    # Track for reward calculation and metrics
    has_invalid_action = False
    is_truncated = False
    truncation_reason = None

    try:
        result = env.reset()
        obs = result.observation
        done = False
        turn_num = 0

        while not done and turn_num < max_turns:
            # Add user message with current game state
            player_total = obs.metadata.get("player_total", "?")
            dealer_card = obs.metadata.get("dealer_card", "?")
            dealer_str = "Ace" if dealer_card == 1 else str(dealer_card)

            state_desc = f"=== BlackJack Game (Turn {turn_num + 1}) ===\n\n"
            state_desc += "Current State:\n"
            state_desc += f"  Your hand total: {player_total}\n"
            state_desc += f"  Dealer shows: {dealer_str}\n"
            state_desc += f"  Legal actions: HIT, STAND\n\n"
            state_desc += "What do you do? Output only 'HIT' or 'STAND'."

            messages.append({"role": "user", "content": state_desc})

            # Format prompt from full message history
            prompt_text = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False
            )

            # Encode to check budget
            prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=False)

            # Check if prompt exceeds budget
            if len(prompt_tokens) >= max_seq_len:
                is_truncated = True
                truncation_reason = "max_seq_len"
                record_metric("episode/terminated_budget_exceeded", 1, Reduce.MEAN)
                print(f"  [TRUNCATED] Prompt length {len(prompt_tokens)} >= {max_seq_len}")
                break

            # Calculate remaining budget for this turn
            remaining = max_seq_len - len(prompt_tokens)

            # Generate with remaining budget
            try:
                responses = await asyncio.wait_for(
                    policy.generate.route([prompt_text], sampling_params={"max_tokens": remaining}),
                    timeout=60.0
                )
            except asyncio.TimeoutError:
                print(f"[ERROR] Policy generation timed out for {game_id} at turn {turn_num}")
                raise

            response = responses[0]

            # Check if generation was cut off
            if response.stop_reason == "length":
                is_truncated = True
                truncation_reason = "generation_length"
                record_metric("episode/generation_truncated", 1, Reduce.MEAN)
                print(f"  [TRUNCATED] Generation hit max_tokens={remaining}")
                # Continue to parse and execute, but mark episode as truncated

            # Accumulate tokens and build response mask
            all_tokens.extend(prompt_tokens)
            all_tokens.extend(response.token_ids)
            response_mask.extend([0] * len(prompt_tokens))  # Don't train on prompts
            response_mask.extend([1] * len(response.token_ids))  # Train on responses
            all_logprobs.extend([0.0] * len(prompt_tokens))
            all_logprobs.extend(response.logprobs)

            # Parse action
            action_name = parse_action(response.text)

            # Add assistant response to message history
            messages.append({"role": "assistant", "content": response.text})


            if action_name == "INVALID":
                has_invalid_action = True
                action_name = "STAND"  # Fallback
                action_id = 1
            elif action_name == "HIT":
                action_id = 0
            elif action_name == "STAND":
                action_id = 1

            # Execute action
            result = env.step(
                OpenSpielAction(action_id=action_id, game_name="blackjack")
            )
            obs = result.observation
            done = result.done

            turn_num += 1

        # Check if hit max_turns
        if turn_num >= max_turns and not done:
            is_truncated = True
            truncation_reason = "max_turns"
            record_metric("episode/hit_max_turns", 1, Reduce.MEAN)

        # Get final game outcome
        final_game_reward = result.reward

        outcome_text = (
            "WIN" if final_game_reward > 0
            else ("LOSS" if final_game_reward < 0 else "PUSH")
        )
        print(f"  Result: {outcome_text} (reward={final_game_reward}, turns={turn_num})")

        # Calculate final reward using separate function
        reward = calculate_reward(
            env_reward=final_game_reward,
        )

        # Metrics
        record_metric("reward/env_reward", final_game_reward, Reduce.MEAN)
        record_metric("reward/final_reward", reward, Reduce.MEAN)
        record_metric("reward/invalid_action_rate", int(has_invalid_action), Reduce.MEAN)
        record_metric("game/total_games_played", 1, Reduce.SUM)
        record_metric("game/average_game_length_in_turns", turn_num, Reduce.MEAN)
        record_metric("game/average_reward", final_game_reward, Reduce.MEAN)
        record_metric("game/win_rate", final_game_reward > 0:, Reduce.MEAN)

        # Create episode
        episode = Episode(
            episode_id=str(uuid.uuid4()),
            task_name="blackjack",
            generator_version=0,  # TODO: Get from policy
            is_truncated=is_truncated,
            all_token_ids=torch.tensor(all_tokens, dtype=torch.long),
            logprobs=torch.tensor(all_logprobs, dtype=torch.float),
            response_mask=torch.tensor(response_mask, dtype=torch.float),
            reward=reward,
            advantage=None,  # Computed later
            ref_logprobs=None,  # Computed later
            message_log=messages,
            metadata={
                "num_turns": turn_num,
                "game_id": game_id,
                "env_reward": final_game_reward,
                "has_invalid_action": has_invalid_action,
                "truncation_reason": truncation_reason,
            }
        )

        return episode

    except Exception as e:
        print(f"[ERROR] play_game {game_id} failed with {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        env.close()
```

**Key changes**:
- Takes `tokenizer`, `pad_id`, `max_seq_len`, `max_turns` parameters
- Builds messages list and formats each turn
- Tracks episode-level budget
- Returns single Episode with concatenated tokens
- No longer returns list of step_results

---

### Step 6: Update continuous_rollouts()

**File**: `apps/blackjack/main.py`

**Replace current continuous_rollouts()** (main.py:714-786) with:

```python
async def continuous_rollouts(tokenizer, pad_id):
    rollout_count = 0
    server_url = cfg.blackjack_env.get("server_url", "http://localhost:8004")
    max_seq_len = cfg.blackjack_env.get("max_seq_len", 2048)
    max_turns = cfg.blackjack_env.get("max_turns", 10)

    while not shutdown_event.is_set():
        t = Tracer("main_perf/continuous_rollouts")
        t.start()

        # Play group_size games, each returns ONE episode
        episodes = []
        for game_idx in range(group_size):
            game_id = str(uuid.uuid4())[:8]
            episode = await play_game(
                game_idx=game_idx,
                game_id=game_id,
                server_url=server_url,
                policy=policy,
                tokenizer=tokenizer,
                pad_id=pad_id,
                max_seq_len=max_seq_len,
                max_turns=max_turns,
                rollout_count=rollout_count,
            )
            episodes.append(episode)

        t.step("play_games")

        # Compute reference logprobs for all episodes
        max_len = max(len(e.all_token_ids) for e in episodes)

        # Pad episodes to same length for batching
        padded_tokens = []
        for episode in episodes:
            seq_len = len(episode.all_token_ids)
            pad_len = max_len - seq_len
            padded = F.pad(episode.all_token_ids, (0, pad_len), value=pad_id)
            padded_tokens.append(padded)

        input_ids = torch.stack(padded_tokens)  # [batch, max_len]

        # Get reference logprobs
        ref_logprobs = await ref_model.forward.route(
            input_ids,
            0,  # No separate prompt (mask handles it)
            return_logprobs=True
        )
        t.step("reference_model_calculate_logprobs")

        # Assign ref_logprobs to episodes (unpad)
        for i, episode in enumerate(episodes):
            seq_len = len(episode.all_token_ids)
            episode.ref_logprobs = ref_logprobs[i, :seq_len]  # Unpad

        del ref_logprobs, input_ids

        # Compute advantages
        advantages = await compute_advantages.compute.call_one(episodes)
        for episode, advantage in zip(episodes, advantages):
            episode.advantage = advantage
            await replay_buffer.add.call_one(episode)

        rollout_count += 1
        record_metric("main/continuous_rollouts/count_rollout_iterations", 1, Reduce.SUM)
        t.stop()
```

**Key changes**:
- Takes `tokenizer` and `pad_id` parameters
- Gets `max_seq_len` and `max_turns` from config
- Passes new parameters to `play_game()`
- Handles variable-length episodes from `play_game()`

---

### Step 7: Update collate() Function

**File**: `apps/blackjack/main.py`

**Replace current collate()** (main.py:131-166) with:

```python
def collate(
    batches: list[Group],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Collates episodes into batches with dynamic padding.

    Each episode has variable length (different number of turns).
    """
    inputs = []
    targets = []

    for batch in batches:
        # Find max length in this batch
        max_len = max(len(e.all_token_ids) for e in batch)
        pad_id = 0  # Will be set via F.pad value parameter

        all_token_ids = []
        logprobs_list = []
        ref_logprobs_list = []
        advantages_list = []
        masks = []

        for e in batch:
            seq_len = len(e.all_token_ids)
            pad_len = max_len - seq_len

            # Right-pad tokens
            padded_tokens = F.pad(e.all_token_ids, (0, pad_len), value=pad_id)
            all_token_ids.append(padded_tokens)

            # Right-pad response_mask (0 for padding)
            padded_mask = F.pad(e.response_mask, (0, pad_len), value=0)
            masks.append(padded_mask)

            # Pad logprobs
            padded_logprobs = F.pad(e.logprobs, (0, pad_len), value=0)
            logprobs_list.append(padded_logprobs)

            # Pad ref_logprobs
            padded_ref = F.pad(e.ref_logprobs, (0, pad_len), value=0)
            ref_logprobs_list.append(padded_ref)

            advantages_list.append(e.advantage)

        input = {"tokens": torch.stack(all_token_ids)}
        target = {
            "response": torch.stack(all_token_ids),  # Full sequence
            "ref_logprobs": torch.stack(ref_logprobs_list),
            "advantages": torch.tensor(advantages_list).unsqueeze(-1),
            "padding_mask": torch.stack(masks),  # Combined response + padding mask
        }

        inputs.append(input)
        targets.append(target)

    return inputs, targets
```

**Key changes**:
- Dynamic padding based on max episode length in batch
- Uses `response_mask` instead of computing mask from pad_id
- Works with variable-length episodes

---

### Step 8: Update main() Service Initialization

**File**: `apps/blackjack/main.py`

**Remove `reward_actor` from service initialization** (main.py:640-654):

```python
# DELETE this from asyncio.gather:
# BlackJackReward.options(**cfg.services.reward_actor).as_service(),

# BEFORE:
(
    env_actor,
    policy,
    trainer,
    replay_buffer,
    compute_advantages,
    ref_model,
    reward_actor,  # DELETE THIS
) = await asyncio.gather(...)

# AFTER:
(
    env_actor,
    policy,
    trainer,
    replay_buffer,
    compute_advantages,
    ref_model,
) = await asyncio.gather(
    EnvironmentActor.options(**cfg.actors.blackjack_env).as_actor(**env_actor_config),
    Policy.options(**cfg.services.policy).as_service(**cfg.policy),
    TitanTrainer.options(**cfg.actors.trainer).as_actor(**cfg.trainer, loss=simple_grpo_loss),
    ReplayBuffer.options(**cfg.actors.replay_buffer).as_actor(**cfg.replay_buffer, collate=collate),
    ComputeAdvantages.options(**cfg.actors.compute_advantages).as_actor(),
    ReferenceModel.options(**cfg.services.ref_model).as_service(**cfg.ref_model),
)
```

---

### Step 9: Add Config Parameters

**File**: `apps/blackjack/qwen3_1_7b.yaml` (or similar config file)

**Add to `blackjack_env` section**:

```yaml
blackjack_env:
  server_url: "http://localhost:8004"
  server_port: 8004
  game_name: "blackjack"
  model: "Qwen/Qwen3-1.7B"
  max_seq_len: 2048      # Episode-level budget (all turns)
  max_turns: 10          # Hard limit on turns

policy:
  engine_args:
    enable_prefix_caching: true  # Critical for multi-turn (2-3x speedup)
    # max_model_len defaults to model's context length
```

---

### Step 10: Remove Old Code

**File**: `apps/blackjack/main.py`

**Delete**:
1. Old `Episode` class (lines 80-122)
2. `BlackJackReward` actor (lines 258-302)
3. `format_prompt()` function (lines 189-242) - replaced by inline message building
4. `EnvironmentActor` class (lines 316-340) - no longer needed

**Add import**:
```python
from apps.blackjack.episode import Episode, Group
```

---

## Benefits of This Refactor

1. **Fixes fundamental learning problem**: Model gets single reward for entire action sequence
2. **Multi-turn ready**: Same structure works for tool calling later
3. **Proper masking**: `response_mask` prevents training on prompts (critical for tool calling)
4. **Budget tracking**: Episode-level `max_seq_len` prevents OOM
5. **Simpler code**: No `BlackJackReward` actor, reward calculated inline
6. **Variable length**: Collate handles different game lengths dynamically
7. **Message format**: Ready for tool calling with structured messages
8. **Aligned with docs**: Follows patterns from `1_message_format_for_tool_calling.md`, `2_episode_class.md`, `3_truncation.md`

---

## Open Questions & TODOs

### 1. Generator Version Tracking

**Question**: How to get current policy version from Generator?

**Current**: Hardcoded to 0
```python
generator_version=0  # TODO: Get from policy
```

**Need to investigate**: Does Generator actor expose a `.version` property? Or do we track it in main loop?

---

### 2. Reward Scaling

**Question**: What's the right balance between env reward and custom shaping?

**Current plan**:
```python
Win=3, Push=1, Loss=-1, Invalid=-1
```

**Alternative**: Use pure env reward
```python
Win=1, Push=0, Loss=-1, Invalid=-1
```

**Recommendation**: Start with custom scaling, monitor metrics, adjust once model learns basic strategy.

---

### 3. Dataset Integration (Future)

**From `1_message_format_for_tool_calling.md`:**

For blackjack, we don't have a traditional "dataset" - each game generates fresh data. But the pattern is:
- Dataset should return `{"messages": [...], "target": ..., "task_name": "blackjack"}`
- For blackjack: `messages = [{"role": "system", "content": "..."}]`
- This is currently inline in `play_game()`, could be extracted to a dataset-like function

**TODO**: Investigate how other frameworks structure dataset output schema (TypedDict, dataclass, etc.)

---

### 4. Truncated Episode Handling

**From `3_truncation.md`:**

Should we drop truncated episodes from training?

**Config option**:
```yaml
grpo:
  include_truncated_in_buffer: false  # Drop incomplete episodes
```

**Need to implement** in `continuous_rollouts()`:
```python
if not episode.is_truncated or cfg.grpo.get("include_truncated_in_buffer", True):
    await replay_buffer.add.call_one(episode)
else:
    record_metric("replay_buffer/episodes_dropped_truncated", 1, Reduce.SUM)
```

---

### 5. Prefix Caching Verification

**From `3_truncation.md`:**

Enable prefix caching for 2-3x speedup on multi-turn prompts.

**Config**:
```yaml
policy:
  engine_args:
    enable_prefix_caching: true
```

**TODO**: Verify this is enabled and measure speedup in metrics.

---

## Migration Checklist

- [ ] Create `apps/blackjack/episode.py` with new Episode class
- [ ] Update `parse_action()` to return "HIT", "STAND", "INVALID"
- [ ] Add `calculate_reward()` function
- [ ] Delete `BlackJackReward` actor
- [ ] Get tokenizer in `main()` and pass to rollout loop
- [ ] Refactor `play_game()` to return single Episode
- [ ] Update `continuous_rollouts()` to handle new signature
- [ ] Update `collate()` for variable-length episodes
- [ ] Remove `reward_actor` from service initialization
- [ ] Add `max_seq_len`, `max_turns` to config
- [ ] Enable `prefix_caching` in policy config
- [ ] Delete old Episode class from main.py
- [ ] Delete `format_prompt()` function
- [ ] Delete `EnvironmentActor` class
- [ ] Test with single game
- [ ] Test with group_size > 1
- [ ] Monitor new metrics (truncation_reason, episode length, etc.)
- [ ] Verify model training improves with multi-turn structure

---

**End of Plan**
