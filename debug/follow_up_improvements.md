# Follow-up Improvements to TokenAccumulator V3

**Date:** 2025-01-17
**Changes:** TruncationReason dataclass, initial message handling, zero budget tests

---

## 1. TruncationReason Dataclass

### Motivation
Allow programmatic filtering of truncated episodes by type (e.g., drop assistant truncations, keep max_turns truncations).

### Implementation
```python
@dataclass
class TruncationReason:
    """Reason for episode truncation."""
    type: str  # "generation_hit_max_tokens", "user_message_length", "initial_messages_too_long", "max_turns"
    details: str = ""  # Optional human-readable details

    def __str__(self) -> str:
        return f"{self.type}: {self.details}" if self.details else self.type
```

### Usage
```python
# Check truncation type
if episode.truncation_reason and episode.truncation_reason.type == "generation_hit_max_tokens":
    # Filter out episodes where assistant was truncated
    continue

# Print details
print(f"Truncated: {episode.truncation_reason}")
# Output: "user_message_length: User message 200 tokens + overhead 6 > budget 50"
```

### Changes
- `self.truncation_reason` type changed from `str | None` to `TruncationReason | None`
- All places that set `truncation_reason` now create `TruncationReason(type="...", details="...")`
- Tests updated to check `.type` attribute

---

## 2. Handle Initial Messages > max_seq_len

### Problem
If initial messages (system prompt) exceed `max_seq_len`, the old code would add them anyway, causing immediate budget overflow.

### Solution
In `__init__`, check if initial_tokens exceed budget and truncate:

```python
# Initialize with initial messages
if len(messages) > 0:
    initial_tokens = tokenizer.apply_chat_template(...)

    # Check if initial messages exceed budget
    if len(initial_tokens) > max_seq_len:
        self.is_truncated = True
        self.truncation_reason = TruncationReason(
            type="initial_messages_too_long",
            details=f"{len(initial_tokens)} tokens > {max_seq_len} max_seq_len",
        )
        # Truncate to fit
        initial_tokens = initial_tokens[:max_seq_len]

    self.all_tokens.extend(initial_tokens)
    # ...
```

### Behavior
- Initial messages truncated to fit `max_seq_len`
- `is_truncated=True`, `truncation_reason.type="initial_messages_too_long"`
- `get_remaining_budget()` returns 0 (or small amount if truncation left room)
- Episode should be dropped in training

### Test
```python
def test_initial_messages_too_long(tokenizer):
    long_system = "You are helpful. " * 100  # Very long
    messages = [{"role": "system", "content": long_system}]

    acc = TokenAccumulator(tokenizer, messages, max_seq_len=50, eos_token_id=...)

    assert acc.is_truncated == True
    assert acc.truncation_reason.type == "initial_messages_too_long"
    assert len(acc.all_tokens) == 50  # Truncated to max_seq_len
    assert acc.get_remaining_budget() == 0
```

---

## 3. Zero Budget Behavior

### Problem
What happens if we try to add messages when budget=0? Need clear, tested behavior.

### Solution for add_user_message
If budget allows zero tokens (budget - overhead <= 0), nothing is added:

```python
# Truncate to fit (if budget allows any tokens)
available = max(0, budget - self.assistant_overhead)
user_message_tokens = user_message_tokens[:available]  # Could be empty!

# Accumulate (only if there are tokens to add)
if len(user_message_tokens) > 0:
    self.all_tokens.extend(user_message_tokens)
    # ...
```

**Behavior:**
- Returns `False` (truncated)
- Sets `is_truncated=True`, `truncation_reason.type="user_message_length"`
- Adds 0 tokens if budget is exhausted
- Message still added to `self.messages` but with 0 tokens

### Solution for add_assistant_response
No special handling needed - it uses delta tokenization and will add whatever fits. The key is not exceeding `max_seq_len`.

**Behavior:**
- If budget is very low, assistant tokens might still be added (role markers + content)
- The important check is `len(all_tokens) <= max_seq_len` in finalize()

### Tests

**Test 6: Zero budget user message**
```python
def test_zero_budget_user_message(tokenizer):
    messages = [{"role": "system", "content": "You are helpful." * 50}]  # Takes all budget
    acc = TokenAccumulator(tokenizer, messages, max_seq_len=100, eos_token_id=...)

    initial_len = len(acc.all_tokens)
    success = acc.add_user_message("Hello")

    # Should fail and not add anything (or add 0-1 tokens if budget allowed)
    assert success == False
    assert len(acc.all_tokens) <= initial_len + 1
```

**Test 7: Zero budget assistant message**
```python
def test_zero_budget_assistant_message(tokenizer):
    messages = [{"role": "system", "content": "You are helpful." * 50}]
    acc = TokenAccumulator(tokenizer, messages, max_seq_len=100, eos_token_id=...)

    response_token_ids = [6151, tokenizer.eos_token_id]  # "hi" + EOS
    success = acc.add_assistant_response("hi", response_token_ids)

    # Key: Don't overflow max_seq_len
    assert len(acc.all_tokens) <= acc.max_seq_len
```

---

## 4. Truncation Type Reference

| Type | When | Action | Training |
|------|------|--------|----------|
| `generation_hit_max_tokens` | vLLM truncates assistant (no EOS) | Episode DROPPED (nothing added) | ✗ Drop |
| `user_message_length` | User message + overhead > budget | Message truncated, episode marked | ✗ Drop |
| `initial_messages_too_long` | System prompt > max_seq_len | Prompt truncated, episode marked | ✗ Drop |
| `max_turns` | Rollout hits max_turns | Episode marked (user sets this) | Depends on use case |

**Filtering example:**
```python
# Drop all truncated episodes
if episode.is_truncated:
    continue

# Or: Drop only assistant truncations, keep others
if episode.truncation_reason and episode.truncation_reason.type == "generation_hit_max_tokens":
    continue
```

---

## Summary of Changes

### Code Changes
1. ✅ Added `TruncationReason` dataclass
2. ✅ Updated `truncation_reason` type to `TruncationReason | None`
3. ✅ All truncation setters now create `TruncationReason(type="...", details="...")`
4. ✅ `__init__` now handles initial messages > max_seq_len
5. ✅ `add_user_message` only accumulates if `len(user_message_tokens) > 0`

### Test Changes
1. ✅ Test 5: Initial messages too long
2. ✅ Test 6: Zero budget user message
3. ✅ Test 7: Zero budget assistant message
4. ✅ Test 4: Updated to check `truncation_reason.type`

### Backward Compatibility
⚠️ **Breaking change:** `truncation_reason` is now a dataclass, not a string
- Old: `if episode.truncation_reason == "user_message_length"`
- New: `if episode.truncation_reason and episode.truncation_reason.type == "user_message_length"`

---

**End of Document**
