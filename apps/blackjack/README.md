# Blackjack GRPO Training

## Overview

This project implements GRPO (Group Relative Policy Optimization) training for teaching an LLM to play Blackjack using the OpenSpiel environment from OpenEnv.

**Key Achievement**: Successfully adapted the single-turn GSM8K GRPO example to work with multi-step game-based RL, where each game produces multiple episodes with shared final rewards.

---

## Quick Start

```bash
# Run training
python -m apps.blackjack.main --config apps/blackjack/qwen3_1_7b.yaml
```

---

## Required OpenEnv Fixes

⚠️ **IMPORTANT**: The following fixes must be applied to `/home/felipemello/OpenEnv` for the blackjack training to work correctly.

### Fix 1: HTTP Server Metadata Stripping

**Problem**: The HTTP server was explicitly removing the `metadata` field before sending observations to clients, causing game state information to be lost.

**File**: `/home/felipemello/OpenEnv/src/core/env_server/http_server.py`

**Line to Remove**: Line 153 (approximately)
```python
obs_dict.pop("metadata", None)  # Remove metadata from observation  ← DELETE THIS LINE
```

**Why**: The client expects metadata to contain game state info like `player_total` and `dealer_card`. Without this fix, all metadata arrives as an empty dict `{}`.

---

### Fix 2: Dealer Card Value Conversion

**Problem**: OpenSpiel's `dealers_visible_card()` returns a card index (0-51) representing which physical card in the deck, not the blackjack value (1-10).

**File**: `/home/felipemello/OpenEnv/src/envs/openspiel_env/server/openspiel_environment.py`

**Location**: Lines 255-276 (approximately, in the observation creation section)

**Replace**:
```python
# Extract game-specific metadata for blackjack
metadata = {}
if self.game_name == "blackjack":
    state = self._ospiel_env.get_state
    if hasattr(state, "get_best_player_total"):
        player_total = state.get_best_player_total(self.agent_player)
        metadata["player_total"] = player_total
    if hasattr(state, "dealers_visible_card"):
        dealer_card = state.dealers_visible_card()
        metadata["dealer_card"] = dealer_card  # ❌ This is 0-51, not 1-10!
```

**With**:
```python
# Extract game-specific metadata for blackjack
metadata = {}
if self.game_name == "blackjack":
    # Get underlying OpenSpiel state to access blackjack-specific methods
    state = self._ospiel_env.get_state  # Property, not method!
    if hasattr(state, "get_best_player_total"):
        player_total = state.get_best_player_total(self.agent_player)
        metadata["player_total"] = player_total
    if hasattr(state, "dealers_visible_card"):
        dealer_card_idx = state.dealers_visible_card()
        # Convert card index (0-51) to blackjack value (1-10)
        # This matches the C++ CardValue() logic in blackjack.cc
        # Cards are indexed from 0 to kDeckSize-1 (52 cards total)
        # Rank = card_idx % 13, where 0=Ace, 1-9=2-10, 10=J, 11=Q, 12=K
        rank = dealer_card_idx % 13
        if rank == 0:
            dealer_value = 1  # Ace
        elif rank <= 9:
            dealer_value = rank + 1  # 2-10
        else:
            dealer_value = 10  # Jack, Queen, King
        metadata["dealer_card"] = dealer_value
```

**Why**: The conversion logic mirrors OpenSpiel's C++ `CardValue()` method which isn't exposed to Python bindings. Without this, you'd see invalid dealer cards like 50, 37, etc. instead of 1-10.

---

## Testing the Fixes

Use `/home/felipemello/forge/dummy.py` to verify:

```python
# Test direct environment (bypasses HTTP)
from envs.openspiel_env.server.openspiel_environment import OpenSpielEnvironment
env = OpenSpielEnvironment(game_name="blackjack", agent_player=0, opponent_policy="random")
obs = env.reset()
print(obs.metadata)
# Expected: {'player_total': <some number>, 'dealer_card': <1-10>}

# Test HTTP client (requires server running)
from envs.openspiel_env import OpenSpielEnv
env = OpenSpielEnv(base_url="http://localhost:9000")
env._http.trust_env = False  # Bypass proxy
obs = env.reset().observation
print(obs.metadata)
# Expected: Same as above if fixes are applied
```

---

## Architecture

### Episode Structure

Each blackjack game produces multiple episodes (one per player action):

```python
@dataclass
class Episode:
    episode_id: str           # Unique ID for this step
    game_id: str             # Which game this belongs to
    step_in_game: int        # Step number within the game
    completion: Completion   # Model's response
    reward: float            # Final game outcome (shared across all steps)
    advantage: float         # Normalized advantage
    # ... other fields
```

### Game Flow

1. **Start game**: Reset OpenSpiel environment
2. **Each step**:
   - Format prompt with current state (player total, dealer card, action history)
   - Generate action from policy ("HIT" or "STAND")
   - Execute action in environment
   - Store step data
3. **Game ends**: Assign final reward to ALL steps in the game
4. **Create episodes**: One episode per step, all sharing the final game reward

### Prompt Format

```
=== BlackJack Game (Step 1) ===

Current State:
  Your hand total: 15
  Dealer shows: 10
  Legal actions: HIT, STAND

What do you do? (Output only 'HIT' or 'STAND')
```

For subsequent steps, action history is included:
```
Previous actions:
  1. HIT (hand became 18)
  2. HIT (hand became 23)
```

This allows the model to track card counting and learn from its action sequence.

---

## Metrics Explanation

### Game Outcome Metrics
- **`game/total_games_played`**: Total number of games completed
- **`game/count_wins`**: Games where player won (+1 reward)
- **`game/count_losses`**: Games where player lost (-1 reward)
- **`game/count_pushes`**: Games that tied (0 reward)

### Win Rate & Performance
- **`game/win_rate`**: Percentage of games won (0.0 to 1.0, where 1.0 = 100%)
  - Example: 0.227 = 22.7% win rate
- **`game/average_reward`**: Mean reward across games (-1.0 to +1.0)
  - Can be negative if more losses than wins
  - Example: -0.454 means losing more than winning

### Game Behavior
- **`game/average_game_length_in_steps`**: How many actions per game
  - Low value (e.g., 1.09) suggests model stands too early
- **`game/bust_rate`**: Percentage of games where player busted (>21)
  - Example: 0.227 = 22.7% bust rate

### Hand Analysis
- **`game/average_player_final_hand`**: Average hand total at game end
- **`game/average_dealer_upcard`**: Average dealer visible card (1-10)
- **`game/average_winning_hand_total`**: Average hand when winning
- **`game/average_losing_hand_total`**: Average hand when losing

**Strategy Insight**: If `average_winning_hand_total` is much lower than `average_losing_hand_total`, the model may be standing too early on good hands and hitting too much on bad hands.

---

## Key Code Locations

### Main Training Script
**File**: `/home/felipemello/forge/apps/blackjack/main.py`

- **`format_prompt()`** (line ~202): Creates text prompts from game state
- **`parse_action()`** (line ~257): Parses "HIT"/"STAND" from model output
- **`play_game()`** (line ~365): Plays one complete blackjack game
- **`continuous_rollouts()`** (line ~694): Manages rollout loop
- **`continuous_training()`** (line ~770): Manages training loop

### Helper Actors
- **`BlackJackReward`** (line ~277): Evaluates game outcomes with reward shaping
- **`ComputeAdvantages`** (line ~310): Normalizes rewards to advantages
- **`EnvironmentActor`** (line ~323): Manages tokenizer and server connection

### Configuration
**File**: `/home/felipemello/forge/apps/blackjack/qwen3_1_7b.yaml`

Key settings:
- `group_size`: Number of games per rollout (default: 4)
- `max_req_tokens`: Max prompt length (default: 512)
- `max_res_tokens`: Max response length (default: 256)
- `server_url`: OpenSpiel server URL (default: http://localhost:8004)
- `server_port`: Port for OpenSpiel server (default: 8004)

---

## Implementation Notes

### Differences from GSM8K Example

1. **Multi-step games**: GSM8K is single prompt→response. Blackjack requires playing full games with multiple steps.

2. **Shared rewards**: All steps in a game get the same final reward (win/loss/push).

3. **No dataset**: Instead of sampling from a dataset, we generate games on-the-fly.

4. **Action parsing**: Model outputs are parsed to extract "HIT" or "STAND" decisions.

5. **Game state tracking**: Prompts include current hand, dealer card, and action history.

### Reward Shaping

**File**: `BlackJackReward.evaluate_response()` (line ~278)

```python
if game_reward > 0:
    reward = 2.0   # Make wins more valuable
elif game_reward == 0:
    reward = 0.5   # Pushes better than losses
else:
    reward = -1.0  # Losses
```

This encourages the model to prefer ties over losses and strongly value wins.

### Server Management

The script automatically:
1. Kills any process using the server port
2. Starts OpenSpiel server in background process
3. Waits for health check (up to 30 seconds)
4. Bypasses corporate proxy for localhost connections
5. Gracefully shuts down server on exit

---

## Common Issues

### "Connection refused" on localhost
- **Cause**: Server hasn't started yet
- **Fix**: Wait for "✓ OpenSpiel server ready" message

### Prompts show `?` for game state
- **Cause**: Missing OpenEnv fixes (see above)
- **Fix**: Apply both required fixes and restart server

### Invalid dealer cards (e.g., 50, 37)
- **Cause**: Missing card value conversion fix
- **Fix**: Apply Fix 2 above

### Empty metadata `{}`
- **Cause**: HTTP server stripping metadata
- **Fix**: Apply Fix 1 above

---

## Future Improvements

1. **Better prompting**: Include basic strategy hints or card counting info
2. **Curriculum learning**: Start with simpler scenarios, gradually increase difficulty
3. **Multi-hand tracking**: Support splitting and doubling down
4. **Opponent modeling**: Learn dealer behavior patterns
5. **Reward shaping**: Experiment with intermediate rewards for good decisions

---

## Reference

- **OpenSpiel Blackjack Source**: [blackjack.cc](https://github.com/google-deepmind/open_spiel/blob/master/open_spiel/games/blackjack/blackjack.cc)
- **OpenEnv Repository**: `/home/felipemello/OpenEnv`
- **Original GSM8K Example**: `/home/felipemello/forge/apps/gsm8k/`
