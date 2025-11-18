# TruncationReason Simplification

**Date:** 2025-01-17
**Change:** Simplified TruncationReason from dataclass to simple Enum

---

## Before (Overcomplicated)

```python
@dataclass
class TruncationReason:
    type: str
    details: str = ""

# Usage
self.truncation_reason = TruncationReason(
    type="generation_hit_max_tokens",
    details=f"Response has {len(response_token_ids)} tokens, no EOS"
)

# Checking
if episode.truncation_reason and episode.truncation_reason.type == "generation_hit_max_tokens":
    continue
```

**Problems:**
- Verbose dataclass with type/details split
- Need to access `.type` attribute
- Details string is rarely used
- More complex than needed

---

## After (Simple)

```python
class TruncationReason(Enum):
    """Reason for episode truncation."""

    max_num_turns = "max_num_turns"
    agent_max_length = "agent_max_length"  # Agent generation hit max_tokens (no EOS)
    tool_max_length = "tool_max_length"    # Tool response too long
    user_max_length = "user_max_length"    # User message too long
```

### Usage

```python
# Setting
self.truncation_reason = TruncationReason.agent_max_length

# Checking
if episode.truncation_reason == TruncationReason.agent_max_length:
    continue  # Drop episodes with truncated agent responses
```

**Benefits:**
- ✅ Simple enum values
- ✅ Direct comparison: `==` instead of `.type ==`
- ✅ Clean: `TruncationReason.agent_max_length` instead of complex dataclass
- ✅ Type-safe: IDE autocomplete and type checking work perfectly

---

## Enum Values

| Value | Meaning | When Set |
|-------|---------|----------|
| `max_num_turns` | Hit maximum number of turns | User sets during rollout loop |
| `agent_max_length` | Agent response truncated (no EOS) | vLLM hits max_tokens, response has no EOS token |
| `tool_max_length` | Tool response too long | Tool output exceeds budget |
| `user_max_length` | User message too long | User message + overhead > budget, or initial messages > max_seq_len |

---

## Code Changes

### In TokenAccumulator

**1. Initial messages too long:**
```python
# Before
self.truncation_reason = TruncationReason(
    type="initial_messages_too_long",
    details=f"{len(initial_tokens)} tokens > {max_seq_len} max_seq_len"
)

# After
self.truncation_reason = TruncationReason.user_max_length
```

**2. Agent generation truncated:**
```python
# Before
self.truncation_reason = TruncationReason(
    type="generation_hit_max_tokens",
    details=f"Response has {len(response_token_ids)} tokens, no EOS"
)

# After
self.truncation_reason = TruncationReason.agent_max_length
```

**3. User message truncated:**
```python
# Before
self.truncation_reason = TruncationReason(
    type="user_message_length",
    details=f"User message {len(user_message_tokens)} tokens..."
)

# After
self.truncation_reason = TruncationReason.user_max_length
```

### In Tests

```python
# Before
if acc.truncation_reason.type != "user_message_length":
    print("ERROR")

# After
if acc.truncation_reason != TruncationReason.user_max_length:
    print("ERROR")
```

---

## Example Usage in Training Loop

```python
for episode in episodes:
    # Drop all truncated episodes
    if episode.is_truncated:
        continue

    # Or: Keep some truncations, drop others
    if episode.truncation_reason == TruncationReason.agent_max_length:
        continue  # Drop agent truncations (bad quality)

    if episode.truncation_reason == TruncationReason.user_max_length:
        continue  # Drop user truncations (incomplete context)

    # max_num_turns might be OK to keep (episode completed normally)
    train_on(episode)
```

---

## Migration

**Breaking change:** Code that checks `truncation_reason.type` must be updated:

```python
# Old code (breaks)
if episode.truncation_reason and episode.truncation_reason.type == "generation_hit_max_tokens":
    ...

# New code
if episode.truncation_reason == TruncationReason.agent_max_length:
    ...
```

**Import change:**
```python
from token_accumulator_fn_v3 import TokenAccumulator, TruncationReason

# Now TruncationReason is an Enum, not a dataclass
```

---

## Summary

**Before:** Complex dataclass with type/details split
**After:** Simple enum with clean values

Much cleaner! ✨

---

**End of Document**
