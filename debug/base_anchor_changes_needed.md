# Changes Needed for BASE Anchor Approach

**Date:** 2025-01-17
**Goal:** Document what needs to change to fix Qwen thinking tag issues

---

## Current V2 Problems

### Problem 1: Prefix Matching Breaks with Qwen
```python
# Current V2 approach
def add_user_message(self, content: str):
    self.messages.append({"role": "user", "content": content})

    # Re-tokenize FULL conversation
    full_tokens = tokenizer.apply_chat_template(self.messages, ...)

    # Extract new tokens via prefix matching
    new_tokens = full_tokens[len(self.all_tokens):]  # ❌ BREAKS!
```

**Why it breaks:**
- After Turn 1: `self.all_tokens = 175` (WITH thinking tags)
- Turn 2: Qwen removes thinking tags → `full_tokens = 60`
- Slice `full_tokens[175:]` = **EMPTY!**

### Problem 2: No Budget Enforcement
```python
def add_assistant_response(self, response_token_ids, ...):
    # Just blindly adds tokens, no check if it exceeds max_seq_len!
    self.all_tokens.extend(new_tokens)  # ❌ Can overflow!
```

### Problem 3: Can't Validate Against Ground Truth
```python
def finalize(self):
    ground_truth = tokenizer.apply_chat_template(self.messages, ...)
    # ❌ ground_truth != self.all_tokens due to thinking tag removal
```

---

## BASE Anchor Solution (VERL Approach)

### Core Idea
**Never re-tokenize the full conversation!** Instead:
1. Define a **fixed BASE conversation** that never changes
2. Tokenize **only deltas** (one new message at a time)
3. Use **pre-computed offsets** to slice out just the new tokens

### BASE_CHAT_HISTORY Pattern
```python
# Fixed anchor - same system, empty user
BASE_CHAT_HISTORY = [
    {"role": "system", "content": "<actual system prompt>"},
    {"role": "user", "content": ""},  # Empty placeholder
]
```

**Why this works:**
- No assistant messages → Qwen never removes thinking tags
- Always same structure → consistent tokenization
- We only compute deltas relative to this base

---

## Required Changes

### 1. Initialization (`__init__`)

**Current V2:**
```python
def __init__(self, tokenizer, messages, max_seq_len, eos_token_id, ...):
    self.tokenizer = tokenizer
    self.max_seq_len = max_seq_len
    self.eos_token_id = eos_token_id
    self.messages = messages.copy()
    self.all_tokens = []
    # ... rest of init

    # Initialize with initial messages
    if len(messages) > 0:
        initial_tokens = tokenizer.apply_chat_template(messages, ...)
        self.all_tokens.extend(initial_tokens)
```

**Needed for BASE Anchor:**
```python
def __init__(self, tokenizer, messages, max_seq_len, eos_token_id, ...):
    self.tokenizer = tokenizer
    self.max_seq_len = max_seq_len
    self.eos_token_id = eos_token_id
    self.messages = messages.copy()
    self.all_tokens = []

    # ✅ NEW: Extract system message
    system_msg = (
        messages[0] if messages[0]["role"] == "system"
        else {"role": "system", "content": ""}
    )

    # ✅ NEW: Setup BASE anchor
    self.BASE_CHAT_HISTORY = [
        system_msg,
        {"role": "user", "content": ""},  # Empty user
    ]

    # ✅ NEW: Pre-compute base lengths
    base_wo_gen = tokenizer.apply_chat_template(
        self.BASE_CHAT_HISTORY,
        add_generation_prompt=False,
        tokenize=True,
    )
    self.base_wo_gen_len = len(base_wo_gen)

    base_with_gen = tokenizer.apply_chat_template(
        self.BASE_CHAT_HISTORY,
        add_generation_prompt=True,
        tokenize=True,
    )
    self.base_with_gen_len = len(base_with_gen)

    # ✅ NEW: Store system length for user message slicing
    system_tokens = tokenizer.apply_chat_template(
        [system_msg],
        add_generation_prompt=False,
        tokenize=True,
    )
    self.system_len = len(system_tokens)

    # ✅ NEW: Compute assistant overhead from base
    self.assistant_overhead = self.base_with_gen_len - self.base_wo_gen_len

    # Initialize with initial messages (same as before)
    if len(messages) > 0:
        initial_tokens = tokenizer.apply_chat_template(messages, ...)
        self.all_tokens.extend(initial_tokens)
```

**New instance variables:**
- `self.BASE_CHAT_HISTORY`: Fixed [system, empty_user] conversation
- `self.base_wo_gen_len`: Length of base WITHOUT generation prompt
- `self.base_with_gen_len`: Length of base WITH generation prompt
- `self.system_len`: Length of just system message
- `self.assistant_overhead`: Tokens for generation prompt

---

### 2. Budget Tracking (`get_remaining_budget`)

**Current V2:**
```python
def get_remaining_budget(self) -> int:
    estimated_overhead = 10  # ❌ Hardcoded guess
    return max(0, self.max_seq_len - len(self.all_tokens) - estimated_overhead)
```

**Needed for BASE Anchor:**
```python
def get_remaining_budget(self) -> int:
    # ✅ Use pre-computed overhead
    current_with_overhead = len(self.all_tokens) + self.assistant_overhead
    return max(0, self.max_seq_len - current_with_overhead)
```

**Change:** Use actual `self.assistant_overhead` instead of hardcoded estimate.

---

### 3. Adding User Messages (`add_user_message`)

**Current V2 (BROKEN):**
```python
def add_user_message(self, content: str, check_budget: bool = True):
    # Add to messages
    self.messages.append({"role": "user", "content": content})

    # ❌ Re-tokenize FULL conversation
    full_tokens = self.tokenizer.apply_chat_template(
        self.messages,  # ❌ Full conversation!
        add_generation_prompt=False,
        tokenize=True,
    )

    # ❌ Prefix matching (breaks when Qwen removes thinking tags)
    new_tokens = full_tokens[len(self.all_tokens):]

    # Check budget and accumulate
    # ...
```

**Needed for BASE Anchor:**
```python
def add_user_message(self, content: str, check_budget: bool = True):
    # Add to messages
    self.messages.append({"role": "user", "content": content})

    # ✅ Tokenize ONLY [system, user_new] using BASE anchor
    temp_messages = [
        self.BASE_CHAT_HISTORY[0],  # System
        {"role": "user", "content": content},  # New user message
    ]
    full_with_user = self.tokenizer.apply_chat_template(
        temp_messages,
        add_generation_prompt=False,
        tokenize=True,
    )

    # ✅ Extract only the user message tokens (slice from system_len onwards)
    user_message_tokens = full_with_user[self.system_len:]

    # Check budget
    success = True
    if check_budget:
        new_amount = len(user_message_tokens) + self.assistant_overhead
        budget = self.max_seq_len - len(self.all_tokens)

        if new_amount > budget:
            self.is_truncated = True
            self.truncation_reason = "user_message_length"
            success = False
            # Truncate to fit
            user_message_tokens = user_message_tokens[:max(0, budget - self.assistant_overhead)]

    # Accumulate
    self.all_tokens.extend(user_message_tokens)
    self.response_mask.extend([0] * len(user_message_tokens))
    self.logprobs.extend([0.0] * len(user_message_tokens))

    return success
```

**Key changes:**
1. ✅ Tokenize only `[system, user_new]` instead of full conversation
2. ✅ Slice from `system_len` to get just the user tokens
3. ✅ Use actual `assistant_overhead` for budget check
4. ✅ No prefix matching needed!

---

### 4. Adding Assistant Responses (`add_assistant_response`)

**Current V2 (Partially works but has issues):**
```python
def add_assistant_response(self, response_text, response_token_ids, response_logprobs):
    # Check truncation
    is_truncated = (
        len(response_token_ids) > 0
        and response_token_ids[-1] != self.eos_token_id
    )
    if is_truncated:
        self.is_truncated = True
        self.truncation_reason = "generation_hit_max_tokens"
        return False

    # Add message
    self.messages.append({"role": "assistant", "content": response_text})

    # ❌ Re-tokenize FULL conversation
    full_tokens = self.tokenizer.apply_chat_template(
        self.messages,  # ❌ Full conversation!
        add_generation_prompt=False,
        tokenize=True,
    )
    new_tokens = full_tokens[len(self.all_tokens):]  # ❌ Prefix matching

    # Accumulate and map logprobs
    # ...
```

**Needed for BASE Anchor:**
```python
def add_assistant_response(self, response_text, response_token_ids, response_logprobs):
    # Check truncation
    is_truncated = (
        len(response_token_ids) > 0
        and response_token_ids[-1] != self.eos_token_id
    )
    if is_truncated:
        self.is_truncated = True
        self.truncation_reason = "generation_hit_max_tokens"
        return False

    # ✅ OPTIONAL: Check budget before adding
    if len(self.all_tokens) + len(response_token_ids) + overhead > self.max_seq_len:
        # This should never happen if we used get_remaining_budget() correctly
        # But defensive programming is good
        raise ValueError(f"Assistant response would exceed budget!")

    # Add message
    self.messages.append({"role": "assistant", "content": response_text})

    # ✅ Tokenize ONLY [system, empty_user, assistant_new] using BASE anchor
    temp_messages = [
        self.BASE_CHAT_HISTORY[0],  # System
        {"role": "user", "content": ""},  # Empty user from base
        {"role": "assistant", "content": response_text},  # New assistant
    ]
    full_with_assistant = self.tokenizer.apply_chat_template(
        temp_messages,
        add_generation_prompt=False,
        tokenize=True,
    )

    # ✅ Extract only the assistant tokens (slice from base_wo_gen_len onwards)
    assistant_tokens = full_with_assistant[self.base_wo_gen_len:]

    # Accumulate tokens
    self.all_tokens.extend(assistant_tokens)
    self.response_mask.extend([1] * len(assistant_tokens))

    # Map logprobs: find where vLLM's tokens appear in assistant_tokens
    content_start = None
    if response_logprobs is not None and len(response_logprobs) == len(response_token_ids):
        # Search for vLLM's token_ids in assistant_tokens
        for i in range(len(assistant_tokens) - len(response_token_ids) + 1):
            if assistant_tokens[i:i+len(response_token_ids)] == response_token_ids:
                content_start = i
                break

    # Build logprobs
    if content_start is not None:
        logprobs = (
            [0.0] * content_start +  # Role markers before
            response_logprobs +  # Actual logprobs
            [0.0] * (len(assistant_tokens) - content_start - len(response_token_ids))
        )
    else:
        logprobs = [0.0] * len(assistant_tokens)

    self.logprobs.extend(logprobs)

    return True
```

**Key changes:**
1. ✅ Tokenize only `[system, empty_user, assistant_new]` instead of full conversation
2. ✅ Slice from `base_wo_gen_len` to get just the assistant tokens
3. ✅ Optional budget check for safety
4. ✅ Logprobs mapping stays the same (search for vLLM tokens)
5. ✅ No prefix matching needed!

---

### 5. Validation (`finalize`)

**Current V2:**
```python
def finalize(self, strict=None):
    # ...

    # ❌ This breaks with Qwen thinking tag removal
    ground_truth = self.tokenizer.apply_chat_template(
        self.messages,
        add_generation_prompt=False,
        tokenize=True,
    )

    if len(self.all_tokens) != len(ground_truth):
        # Mismatch! (Expected with Qwen)
```

**Options for BASE Anchor:**

**Option A: Disable strict validation**
```python
def finalize(self, strict=None):
    # Just check assertions, skip ground truth comparison
    assert len(self.all_tokens) == len(self.response_mask)
    assert len(self.all_tokens) == len(self.logprobs)

    # ✅ Can't validate against ground truth with Qwen
    # Our accumulated tokens are correct (match what was generated)
    # Ground truth would be different (thinking tags removed)

    return True
```

**Option B: Validate only structure**
```python
def finalize(self, strict=None):
    assert len(self.all_tokens) == len(self.response_mask)
    assert len(self.all_tokens) == len(self.logprobs)

    # ✅ Check structural properties instead
    if len(self.all_tokens) > self.max_seq_len:
        raise ValueError(f"Exceeded max_seq_len: {len(self.all_tokens)} > {self.max_seq_len}")

    if not self.is_truncated:
        # Check that last message is complete
        # Could decode and check for proper endings
        pass

    return True
```

**Option C: Keep ground truth check but downgrade to warning**
```python
def finalize(self, strict=None):
    # ... assertions ...

    ground_truth = self.tokenizer.apply_chat_template(
        self.messages,
        add_generation_prompt=False,
        tokenize=True,
    )

    if len(self.all_tokens) != len(ground_truth):
        # ⚠️ Expected with Qwen due to thinking tag removal
        # Just warn, don't fail
        print(f"⚠️  Token count mismatch (expected with Qwen thinking tags)")
        print(f"   Accumulated: {len(self.all_tokens)}, Ground truth: {len(ground_truth)}")

    return True
```

**Recommendation:** Use Option A or B. Can't rely on ground truth with Qwen.

---

## Summary of Changes

### New Instance Variables (in `__init__`)
```python
self.BASE_CHAT_HISTORY       # [system, empty_user]
self.base_wo_gen_len         # Length of base without gen prompt
self.base_with_gen_len       # Length of base with gen prompt
self.system_len              # Length of system message only
self.assistant_overhead      # base_with_gen_len - base_wo_gen_len
```

### Changed Methods

| Method | Current Approach | BASE Anchor Approach |
|--------|------------------|---------------------|
| `__init__` | Simple initialization | ✅ Add BASE setup + pre-compute lengths |
| `get_remaining_budget` | Hardcoded overhead (10) | ✅ Use `self.assistant_overhead` |
| `add_user_message` | Re-tokenize full conversation | ✅ Tokenize `[system, user_new]`, slice from `system_len` |
| `add_assistant_response` | Re-tokenize full conversation | ✅ Tokenize `[system, empty_user, assistant_new]`, slice from `base_wo_gen_len` |
| `finalize` | Compare vs ground truth | ✅ Disable ground truth check (or downgrade to warning) |

---

## Why This Fixes All Issues

### ✅ Fixes Test 3 (multi-turn conversation)
**Before:** Prefix matching breaks when Qwen removes thinking tags
- `self.all_tokens = 175`, `full_tokens = 60`, `new_tokens = full_tokens[175:] = EMPTY`

**After:** No prefix matching needed
- Tokenize only `[system, "Now say bye"]`
- Slice from `system_len` to get just the user tokens
- Works regardless of thinking tag removal in previous turns

### ✅ Fixes Test 4 (budget overflow)
**Before:** Hardcoded overhead estimate (10 tokens)
- Actual overhead could be more, causing overflow

**After:** Pre-computed actual overhead
- `self.assistant_overhead = base_with_gen_len - base_wo_gen_len`
- Accurate budget tracking

### ✅ Fixes logprobs mapping
**Before:** Same approach (search for vLLM tokens)

**After:** Same approach but with correct tokens
- Still search for `response_token_ids` in `assistant_tokens`
- But now `assistant_tokens` are correctly extracted via BASE anchor

### ✅ Enables proper validation
**Before:** Can't validate because ground truth differs

**After:** Skip ground truth comparison
- We know our accumulation is correct
- It matches what was actually generated
- Ground truth would differ due to Qwen's behavior

---

## Migration Checklist

- [ ] Add BASE_CHAT_HISTORY setup in `__init__`
- [ ] Pre-compute all base lengths in `__init__`
- [ ] Update `get_remaining_budget` to use `self.assistant_overhead`
- [ ] Rewrite `add_user_message` to use delta tokenization
- [ ] Rewrite `add_assistant_response` to use delta tokenization
- [ ] Update `finalize` to disable ground truth check
- [ ] Add budget overflow check in `add_assistant_response` (defensive)
- [ ] Update tests to use `get_remaining_budget()` for max_tokens

---

## Expected Behavior After Changes

**Test 1:** ✅ Still passes (no changes needed to test)

**Test 2:** ✅ Still passes (truncation detection works the same)

**Test 3:** ✅ Now passes!
- User message "Now say bye" gets added correctly
- Total tokens increases to ~190
- No prefix matching, so Qwen's thinking tag removal doesn't break it

**Test 4:** ✅ Now passes!
- Accurate budget tracking prevents overflow
- If test uses `get_remaining_budget()`, generation won't exceed 150 tokens

---

**End of Document**
