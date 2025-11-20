# Refactoring Proposal 02: Extract TokenAccumulator

## Overview
Building on Proposal 01, this iteration focuses on moving the large `TokenAccumulator` class (400+ lines) to a separate module. This follows the single-responsibility principle and makes main_v2.py focus on the training loop logic.

## Key Changes

### 1. Move TokenAccumulator to Separate File
Create `src/forge/data/token_accumulator.py` with the full class implementation.

**New File Structure:**
```
src/forge/data/
├── common.py (already exists)
├── token_accumulator.py (NEW)
└── ...
```

**In token_accumulator.py:**
```python
"""Token accumulation for multi-turn RL episodes.

Handles incremental tokenization using delta tokenization against
a stable anchor conversation.
"""
from dataclasses import dataclass
from enum import Enum
import threading
import torch

class ValidationMode(Enum):
    STRICT = "strict"
    WARN = "warn"
    OFF = "off"

class TruncationReason(Enum):
    USER_TOO_LONG = "user_too_long"
    ASSISTANT_TOO_LONG = "assistant_too_long"
    MAX_NUM_TURNS = "max_num_turns"

@dataclass
class EpisodeData:
    """Episode data as tensors, ready for training."""
    token_ids: torch.Tensor
    response_mask: torch.Tensor
    logprobs: torch.Tensor
    is_truncated: bool
    truncation_reason: str | None = None

class TokenAccumulator:
    """Accumulate tokens for multi-turn RL episodes using vLLM tokens directly.

    See module docstring for delta tokenization strategy.
    """
    # ... (full implementation)
```

**In main_v2.py:**
```python
from forge.data.token_accumulator import (
    TokenAccumulator,
    ValidationMode,
    TruncationReason,
    EpisodeData,
)
```

### 2. Simplify TokenAccumulator Docstrings
The current docstring is 60+ lines. Move detailed examples to module-level docstring, keep class docstring concise.

**Before (lines 162-223):** Massive docstring with examples
**After:**
```python
class TokenAccumulator:
    """Accumulate tokens for multi-turn episodes with delta tokenization.

    Uses a stable anchor conversation to extract token deltas, avoiding
    expensive re-tokenization. See module docstring for details.

    Args:
        tokenizer: HF tokenizer with apply_chat_template
        messages: Initial messages (must include system)
        max_len: Maximum sequence length
        eos_id: End-of-sequence token ID
        thinking: Enable <think> tags for Qwen
        validation: Validation strictness
    """
```

### 3. Simplify show_messages Method
Currently has complex colorization logic. Make it simpler for debugging purposes.

**Before:** Grouped token runs, color coding, character limits
**After:**
```python
def show_messages(self, show_tokens: bool = False) -> None:
    """Show accumulated messages and optionally token-level details."""
    print("=" * 80)
    print(f"TokenAccumulator: {len(self._tokens)}/{self.max_len} tokens")
    trainable_count = sum(self._mask)
    print(f"Trainable: {trainable_count}/{len(self._tokens)}")
    print("=" * 80)

    for i, msg in enumerate(self.messages):
        print(f"[{i}] {msg['role']:10s}: {msg['content'][:100]}...")

    if show_tokens:
        # Simple token dump without complex colorization
        for i in range(len(self._tokens)):
            symbol = "✓" if self._mask[i] else "·"
            print(f"{symbol} {self._tokens[i]}")

    print("=" * 80)
```

### 4. Remove Unused Validation
The prefix consistency check is disabled (lines 720-744). Remove it entirely.

### 5. Clean Up BlackjackEnv
Move observation formatting logic to be more concise.

**Before:**
```python
def _format_observation(self, observation) -> str:
    player_total = observation.metadata.get("player_total", "?")
    dealer_card = observation.metadata.get("dealer_card", "?")
    dealer_str = "Ace" if dealer_card == 1 else str(dealer_card)
    return f"Hand: {player_total}, Dealer: {dealer_str}"
```

**After:**
```python
def _format_observation(self, obs) -> str:
    """Format game state as text."""
    player = obs.metadata.get("player_total", "?")
    dealer = obs.metadata.get("dealer_card", "?")
    dealer = "Ace" if dealer == 1 else str(dealer)
    return f"Hand: {player}, Dealer: {dealer}"
```

## Impact
- **File size:** ~1400 lines → ~900 lines (additional 35% reduction)
- **Modularity:** Much better - token accumulation logic is now reusable
- **Testability:** TokenAccumulator can be unit tested independently
- **Readability:** Main file focuses on RL loop, not tokenization details
- **Risk:** Low - pure code movement, no logic changes
