# Blackjack RL Training

## Setup

```bash
# Clone and install OpenEnv
git clone git@github.com:meta-pytorch/OpenEnv.git
cd OpenEnv
pip install -e .

# Apply blackjack modifications
python ../forge/apps/blackjack/openenv_patch/apply_patch.py

# Run training
cd ../forge
python -m apps.blackjack.main --config apps/blackjack/qwen3_1_7b.yaml
```

## What gets changed in OpenEnv

### 1. Enable metadata passthrough (`src/core/env_server/http_server.py`)

```python
# Before:
obs_dict.pop("metadata", None)  # Remove metadata from observation

# After:
# obs_dict.pop("metadata", None)  # Remove metadata from observation
```

### 2. Extract blackjack game state (`src/envs/openspiel_env/server/openspiel_environment.py`)

```python
# Add this after line 252 (before creating OpenSpielObservation):

# Extract game-specific metadata for blackjack
metadata = {}
if self.game_name == "blackjack" and not time_step.last():
    try:
        state = self._ospiel_env.get_state
        if hasattr(state, "get_best_player_total"):
            metadata["player_total"] = state.get_best_player_total(
                self.agent_player
            )
        if hasattr(state, "dealers_visible_card"):
            dealer_card_idx = state.dealers_visible_card()
            rank = dealer_card_idx % 13
            if rank == 0:
                dealer_value = 1  # Ace
            elif rank <= 9:
                dealer_value = rank + 1  # 2-10
            else:
                dealer_value = 10  # Jack, Queen, King
            metadata["dealer_card"] = dealer_value
    except Exception:
        pass

# Then update OpenSpielObservation creation:
obs = OpenSpielObservation(
    ...,
    metadata=metadata,  # Add this line
)
```

This allows observations like `"Hand: 17, Dealer: Ace"` instead of raw state vectors.
