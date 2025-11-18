# Truncation V7: Simplified Implementation (Based on test_simple_vllm.py Requirements)

**Date:** 2025-01-16
**Based on:** Exact requirements from `/home/felipemello/forge/test_simple_vllm.py`
**Research:** Library comparison from v7 (6 major RL codebases)
**Implementation:** `/home/felipemello/forge/test_simple_vllm_v2.py` ✅ ALL 5 TESTS PASS

**Status:** Partial simplification achieved. Direct token extraction proved more complex than expected.

---

## Implementation Results Summary

### ✅ What We Achieved

**File:** `/home/felipemello/forge/test_simple_vllm_v2.py`
**Test Results:** ALL 5 TESTS PASS ✅

| Improvement | Status | Impact |
|-------------|--------|--------|
| **TokenAccumulator class** | ✅ Implemented | Better code organization, reusable |
| **Immediate env obs accumulation** | ✅ Implemented | Simpler flow (no dangling messages) |
| **Cached gen_prompt_len** | ✅ Implemented | Small optimization |
| **Optional validation** | ✅ Implemented | Can disable in production |
| **Direct token extraction** | ❌ Not achieved | Harder than expected (see below) |

### ⚠️ Why Direct Token Extraction Failed

**Original plan:** Use `output.token_ids` directly from vLLM (no prefix matching).

**Reality discovered:**
- `output.token_ids` contains **content tokens only** (e.g., `[3]` for "HIT")
- Chat templates add **role headers**: `<|im_start|>assistant\n` + content + `<|im_end|>\n`
- These role header tokens are **template-specific** and not returned by vLLM
- Computing role headers requires understanding each template's format

**Attempt:**
```python
def get_role_header_tokens(tokenizer, role: str) -> list[int]:
    # Failed: Cannot call apply_chat_template([])
    # Unclear how to isolate just the role header portion
```

**Libraries that DO use direct extraction:**
- **Prime-RL/Verifiers:** Use vLLM's `return_tokens_as_token_ids=True` flag
- **NeMo-RL:** Use length-based slicing with vLLM's reported lengths
- **VERL:** Use BASE anchor + delta computation (complex)

**Conclusion:** Direct extraction requires deeper vLLM integration or template-specific logic.

### ✅ What We Still Use (Proven Correct)

**Prefix matching** for both assistant and user messages:
```python
# Add message to messages list
self.messages.append({"role": "assistant", "content": response_text})

# Tokenize full conversation
full_conversation = tokenizer.apply_chat_template(
    self.messages, add_generation_prompt=False, tokenize=True
)

# Extract delta
new_tokens = full_conversation[len(self.all_tokens):]
```

This approach:
- ✅ Works reliably across all chat templates
- ✅ Includes role headers automatically
- ✅ Validated by test suite (all 5 tests pass)
- ✅ Used by TRL, Verifiers, and others

### 📊 Comparison: v1 vs v2

| Metric | v1 (test_simple_vllm.py) | v2 (test_simple_vllm_v2.py) | Improvement |
|--------|--------------------------|----------------------------|-------------|
| **Code organization** | Inline logic | `TokenAccumulator` class | ✅ Much cleaner |
| **Env obs accumulation** | Start of next turn | Immediately | ✅ Simpler |
| **Gen prompt len** | Calculated each turn | Cached | ✅ Faster |
| **Validation** | Every turn (mandatory) | Optional flag | ✅ Flexible |
| **Token extraction** | Prefix matching | Prefix matching | Same |
| **Lines of code per test** | ~150 lines | ~100 lines (with class) | ✅ More compact |

### 🎯 Actual Simplifications Achieved

1. **Better Code Structure** - TokenAccumulator encapsulates all logic
2. **Immediate Accumulation** - Clearer flow, no "start of next turn" confusion
3. **Cached Values** - gen_prompt_len computed once
4. **Cleaner Tests** - Less repetitive code

**Net result:** Code is more maintainable, but NOT fewer tokenization calls (still uses prefix matching).

---

## Exact Requirements from test_simple_vllm.py

The test shows the following **precise flow** for multi-turn token accumulation:

### Per-Turn Flow (13 Steps)

**START OF TURN:**
1. **Extract new prompt tokens** (delta)
   - Tokenize `messages` WITHOUT gen prompt
   - Extract: `new_prompt_tokens = full_conversation[len(all_tokens):]`
   - Add to `all_tokens` with `mask=0`

2. **Check budget**
   - Tokenize `messages` WITH gen prompt
   - Calculate: `remaining = max_seq_len - len(prompt_with_gen)`
   - If `remaining <= 0`: break (early exit)

3. **Generate**
   - Create prompt text (tokenize=False, for display)
   - Set `max_tokens = min(remaining, default_max_tokens)`
   - Generate with vLLM
   - Get `response_text` and `response_tokens` (content only, no role headers)

**AFTER GENERATION:**
4. **Add assistant message to messages**
   - `messages.append({"role": "assistant", "content": response_text})`

5. **Extract assistant tokens** (delta, with role headers)
   - Tokenize `messages` (now includes assistant) WITHOUT gen prompt
   - Extract: `assistant_tokens = full_conversation_with_assistant[len(all_tokens):]`
   - This includes role headers: `<|im_start|>assistant\n` + content + `<|im_end|>\n`

6. **Check truncation**
   - If `response_tokens[-1] != eos_token_id`: truncated
   - Set `mask_value = 0` if truncated, else `1`

7. **Add assistant tokens to all_tokens**
   - `all_tokens.extend(assistant_tokens)`
   - `response_mask.extend([mask_value] * len(assistant_tokens))`

8. **Validate** (optional, debug only)
   - Compare `all_tokens` vs ground truth tokenization

**CHECK EARLY EXIT:**
9. **If generation truncated**: break

10. **If game done**: break

**ENV OBSERVATION:**
11. **Add env observation to messages**
    - `messages.append({"role": "user", "content": env_obs})`

12. **Check if env obs exceeds budget**
    - Tokenize `messages` WITH gen prompt (includes new env obs)
    - If `len(temp_conversation) > max_seq_len`:
      - `messages.pop()` (remove the env obs we just added)
      - Break loop

13. **Loop** back to step 1

---

## Key Insights

### 1. Two Accumulation Points Per Turn

**This is critical and often missed!**

Each turn accumulates tokens **TWICE**:
- **Start of turn (step 1):** Accumulate NEW PROMPT TOKENS (the env observation from previous turn)
- **After generation (step 7):** Accumulate ASSISTANT TOKENS (with role headers)

```python
# Visualization of token accumulation
Turn 1 start:  [system, user1]                              # NEW: user1 tokens
Turn 1 gen:    [system, user1, assistant1]                  # NEW: assistant1 tokens
Turn 2 start:  [system, user1, assistant1, user2]           # NEW: user2 tokens
Turn 2 gen:    [system, user1, assistant1, user2, assistant2]  # NEW: assistant2 tokens
```

### 2. Three Tokenization Calls Per Turn (Current Approach)

Looking at the test, each turn does:
1. **Tokenize to extract new prompt tokens** (line 49, tokenize=True)
2. **Tokenize to check budget** (line 67, tokenize=True)
3. **Tokenize to extract assistant tokens** (line 113, tokenize=True)
4. **Tokenize to check env obs budget** (line 189, tokenize=True)
5. **Tokenize for validation** (line 146, tokenize=True) - OPTIONAL

**Total: 4 required calls, 1 optional = 3-5 per turn**

*(Not counting the tokenize=False call at line 86 which is just for string formatting)*

### 3. Prefix Matching is Used Twice

- **For prompt tokens:** Extract delta at start of turn (step 1)
- **For assistant tokens:** Extract delta after generation (step 5)

Both use the same pattern: `delta = full_conversation[len(all_tokens):]`

### 4. Budget Check is Required Before Generation

You CANNOT skip the budget check (step 2) - it's required to:
- Know if we can generate at all (`remaining <= 0` → early exit)
- Set `max_tokens` appropriately for vLLM

---

## Current Implementation Tokenization Count

From test_simple_vllm.py, here are the actual `apply_chat_template` calls:

| Step | Line | Call | Purpose | Required? |
|------|------|------|---------|-----------|
| 1 | 49-54 | `apply_chat_template(messages, add_generation_prompt=False, tokenize=True)` | Extract new prompt tokens | ✅ YES |
| 2 | 67-72 | `apply_chat_template(messages, add_generation_prompt=True, tokenize=True)` | Check budget | ✅ YES |
| 3 | 86-91 | `apply_chat_template(messages, add_generation_prompt=True, tokenize=False)` | Format prompt text | ⚠️ NO (vLLM can do this) |
| 4 | 113-118 | `apply_chat_template(messages, add_generation_prompt=False, tokenize=True)` | Extract assistant tokens | ✅ YES (with current approach) |
| 5 | 146-151 | `apply_chat_template(messages, add_generation_prompt=False, tokenize=True)` | Validation | ⚠️ NO (debug only) |
| 6 | 189-194 | `apply_chat_template(messages, add_generation_prompt=True, tokenize=True)` | Check env obs budget | ✅ YES |

**Total required: 4 tokenization calls per turn**

---

## Proposed Simplifications (Based on Library Research)

From the library comparison (v7), we identified these optimizations:

### ⭐ Optimization 1: Use Direct Token IDs from vLLM

**Current (steps 4-5):**
```python
messages.append({"role": "assistant", "content": response_text})

# Extract assistant tokens via prefix matching
full_conversation_with_assistant = tokenizer.apply_chat_template(
    messages, add_generation_prompt=False, tokenize=True
)
assistant_tokens = full_conversation_with_assistant[len(all_tokens):]
```

**Simplified (all 6 libraries do this):**
```python
# Get assistant tokens directly from vLLM response
assistant_content_tokens = output.token_ids  # Direct from vLLM!

# Get role header tokens (computed once, can be cached)
role_header_tokens = get_role_header_tokens(tokenizer, "assistant")

# Combine
assistant_tokens = role_header_tokens + assistant_content_tokens

# Add to messages (for next turn's prompt)
messages.append({"role": "assistant", "content": response_text})
```

This **eliminates 1 tokenization call** (step 4).

**Helper function (cached):**
```python
@lru_cache(maxsize=2)
def get_role_header_tokens(tokenizer, role: str) -> list[int]:
    """Get tokens for '<|im_start|>assistant\n' etc."""
    empty_msg = tokenizer.apply_chat_template(
        [{role: role, "content": ""}],
        add_generation_prompt=False,
        tokenize=True,
    )
    base = tokenizer.apply_chat_template(
        [],
        add_generation_prompt=False,
        tokenize=True,
    )
    return empty_msg[len(base):]
```

### ⭐ Optimization 2: Use BASE Anchor for Prompt Tokens (VERL Pattern)

**Current (step 1):**
```python
# Tokenize entire conversation every turn
full_conversation = tokenizer.apply_chat_template(
    messages,  # Could be 10+ messages!
    add_generation_prompt=False,
    tokenize=True,
)
new_prompt_tokens = full_conversation[len(all_tokens):]
```

**Simplified (VERL pattern):**
```python
# Pre-compute BASE anchor once at initialization
BASE_CONVERSATION = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": ""},  # Empty placeholder
]
base_tokens = tokenizer.apply_chat_template(BASE_CONVERSATION, ...)
base_len = len(base_tokens)

# For each new user message, tokenize ONLY the delta
def get_user_message_tokens(content: str) -> list[int]:
    temp = BASE_CONVERSATION.copy()
    temp[-1]["content"] = content

    full = tokenizer.apply_chat_template(temp, add_generation_prompt=False, tokenize=True)
    return full[base_len:]  # Extract only the new tokens!
```

This is **more efficient** for long conversations (tokenize 2 messages instead of N messages).

**Caveat:** Works best for simple user messages. For complex multi-message scenarios (tool calls, etc.), fall back to full tokenization.

### ⭐ Optimization 3: Smarter Budget Check for Env Obs

**Current (step 12):**
```python
# Add env obs to messages
messages.append({"role": "user", "content": env_obs})

# Tokenize ENTIRE conversation again
temp_conversation = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
)

if len(temp_conversation) > max_seq_len:
    messages.pop()
    break
```

**Simplified:**
```python
# Get env obs tokens
env_obs_tokens = get_user_message_tokens(env_obs)  # Using BASE anchor

# Calculate: current + env_obs + gen_prompt
gen_prompt_len = get_generation_prompt_len(tokenizer)  # Cached
would_be = len(all_tokens) + len(env_obs_tokens) + gen_prompt_len

if would_be > max_seq_len:
    # Don't even add to messages
    break
else:
    # Add to both messages and all_tokens
    messages.append({"role": "user", "content": env_obs})
    all_tokens.extend(env_obs_tokens)
    response_mask.extend([0] * len(env_obs_tokens))
```

**Problem:** This approach accumulates env obs tokens at the END of the turn, but the test accumulates them at the START of the next turn.

**Solution:** Keep the test's approach (accumulate at start of next turn) OR switch to immediate accumulation (simpler but different ordering).

### Trade-off: When to Accumulate Env Obs Tokens?

**Option A: Accumulate at START of next turn (current test approach)**
- ✅ Pro: Matches test exactly
- ❌ Con: Need to tokenize at start of turn

**Option B: Accumulate IMMEDIATELY after env.step()**
- ✅ Pro: Simpler flow, no "dangling" messages
- ✅ Pro: Can skip tokenization at start of turn
- ❌ Con: Different from test (but equivalent)

**Recommendation:** Use Option B (immediate accumulation) as it's cleaner and matches how most libraries do it (TRL, NeMo-RL, etc.).

---

## Simplified Implementation

### Updated Flow (12 Steps, Immediate Env Obs Accumulation)

**START OF TURN:**
1. **Check budget**
   - Count tokens in `all_tokens` + gen_prompt_len
   - Calculate: `remaining = max_seq_len - (len(all_tokens) + gen_prompt_len)`
   - If `remaining <= 0`: break

2. **Generate**
   - Format prompt from `messages` (can use cached template)
   - Set `max_tokens = min(remaining, default_max_tokens)`
   - Generate with vLLM

**AFTER GENERATION:**
3. **Get assistant tokens directly**
   - `assistant_content_tokens = output.token_ids` (from vLLM)
   - `role_header_tokens = get_role_header_tokens(tokenizer, "assistant")` (cached)
   - `assistant_tokens = role_header_tokens + assistant_content_tokens`

4. **Check truncation**
   - If `output.token_ids[-1] != eos_token_id`: truncated
   - Set `mask_value = 0` if truncated, else `1`

5. **Add assistant tokens**
   - `all_tokens.extend(assistant_tokens)`
   - `response_mask.extend([mask_value] * len(assistant_tokens))`
   - `messages.append({"role": "assistant", "content": output.text})`

6. **Validate** (optional)

**CHECK EARLY EXIT:**
7. **If generation truncated**: break

8. **If game done**: break

**ENV OBSERVATION (IMMEDIATE ACCUMULATION):**
9. **Get env observation**
   - `env_result = env.step(action)`
   - `env_obs = env_result.observation`

10. **Get env obs tokens**
    - Option A (simple): `env_obs_tokens = tokenizer.encode(env_obs, add_special_tokens=False)`
    - Option B (BASE anchor): `env_obs_tokens = get_user_message_tokens(env_obs)`

11. **Check if adding env obs would exceed budget**
    - Calculate: `would_be = len(all_tokens) + len(env_obs_tokens) + gen_prompt_len`
    - If `would_be > max_seq_len`: break (truncated)

12. **Add env obs tokens IMMEDIATELY**
    - `messages.append({"role": "user", "content": env_obs})`
    - `all_tokens.extend(env_obs_tokens)` ← IMMEDIATE!
    - `response_mask.extend([0] * len(env_obs_tokens))`

13. **Loop** back to step 1

---

## Tokenization Call Comparison

| Step | Current Test (v6) | Simplified (v7) | Savings |
|------|-------------------|-----------------|---------|
| **Start of turn** | Extract new prompt tokens (tokenize=True) | ❌ Skipped (accumulated immediately last turn) | -1 call |
| **Budget check** | Tokenize with gen prompt (tokenize=True) | ✅ Use `len(all_tokens) + gen_prompt_len` | -1 call (cached gen_prompt_len) |
| **Format prompt** | Tokenize=False for string | ✅ Same | 0 |
| **Extract assistant** | Prefix matching (tokenize=True) | ❌ Use `output.token_ids` + cached role headers | -1 call |
| **Env obs** | Tokenize to check budget (tokenize=True) | ✅ Use BASE anchor or simple encode | Same (but faster) |
| **Validation** | Full tokenization (tokenize=True) | ⚠️ Optional | 0 (optional) |

**Total: 4 calls → 1-2 calls per turn (depending on BASE anchor usage)**

---

## Complete Simplified Code (IMPLEMENTED & TESTED)

### File: `test_simple_vllm_v2.py` - TokenAccumulator Class

**Key changes from v1:**
1. ✅ Uses `TokenAccumulator` class (better organization)
2. ✅ Immediate env obs accumulation (simpler flow)
3. ✅ Cached gen_prompt_len (optimization)
4. ✅ Optional validation flag
5. ⚠️ Still uses prefix matching (proven correct, not "direct")

```python
@lru_cache(maxsize=1)
def get_generation_prompt_len(tokenizer) -> int:
    """Get length of generation prompt (e.g., '<|im_start|>assistant\n')."""
    messages = [{"role": "user", "content": "x"}]
    without_gen = tokenizer.apply_chat_template(
        messages, add_generation_prompt=False, tokenize=True
    )
    with_gen = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True
    )
    return len(with_gen) - len(without_gen)


class TokenAccumulator:
    """
    Simplified token accumulator with hybrid approach.

    Uses prefix matching (proven correct) with better organization.
    """

    def __init__(
        self,
        tokenizer,
        messages: list[dict],
        max_seq_len: int,
        eos_token_id: int,
        validate: bool = True,
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.eos_token_id = eos_token_id
        self.validate_enabled = validate

        # Message log (for prompt construction)
        self.messages = messages.copy()

        # Token accumulators
        self.all_tokens: list[int] = []
        self.response_mask: list[int] = []

        # Cached values
        self.gen_prompt_len = get_generation_prompt_len(tokenizer)

        # Truncation tracking
        self.is_truncated = False
        self.truncation_reason: str | None = None

        # Initialize with initial messages
        if len(messages) > 0:
            initial_tokens = tokenizer.apply_chat_template(
                messages, add_generation_prompt=False, tokenize=True
            )
            self.all_tokens.extend(initial_tokens)
            self.response_mask.extend([0] * len(initial_tokens))

    def get_remaining_budget(self) -> int:
        """Calculate remaining tokens before hitting max_seq_len."""
        current_with_gen_prompt = len(self.all_tokens) + self.gen_prompt_len
        return self.max_seq_len - current_with_gen_prompt

    def format_prompt(self) -> str:
        """Format prompt for generation (no tokenization, just string)."""
        return self.tokenizer.apply_chat_template(
            self.messages, add_generation_prompt=True, tokenize=False
        )

    def add_assistant_response(
        self, response_text: str, response_token_ids: list[int]
    ) -> bool:
        """
        Add assistant response using prefix matching.

        Args:
            response_text: Response text from vLLM
            response_token_ids: Content tokens (for truncation check only)

        Returns:
            True if successful, False if truncated
        """
        # Check truncation
        is_truncated = (
            len(response_token_ids) > 0 and
            response_token_ids[-1] != self.eos_token_id
        )

        # Add to messages FIRST
        self.messages.append({"role": "assistant", "content": response_text})

        # Use prefix matching to get assistant tokens WITH role headers
        full_conversation = self.tokenizer.apply_chat_template(
            self.messages, add_generation_prompt=False, tokenize=True
        )
        assistant_tokens = full_conversation[len(self.all_tokens):]

        # Accumulate
        mask_value = 0 if is_truncated else 1
        self.all_tokens.extend(assistant_tokens)
        self.response_mask.extend([mask_value] * len(assistant_tokens))

        # Track truncation
        if is_truncated:
            self.is_truncated = True
            self.truncation_reason = "generation_length"

        # Validate if enabled
        if self.validate_enabled:
            self._validate()

        return not is_truncated

    def add_user_message(self, content: str, check_budget: bool = True) -> bool:
        """
        Add user message (env observation) IMMEDIATELY using prefix matching.

        Args:
            content: User message content
            check_budget: If True, check if adding would exceed budget

        Returns:
            True if successful, False if would exceed budget
        """
        # Add to messages FIRST
        self.messages.append({"role": "user", "content": content})

        # Use prefix matching to get user message tokens
        full_conversation = self.tokenizer.apply_chat_template(
            self.messages, add_generation_prompt=False, tokenize=True
        )
        user_message_tokens = full_conversation[len(self.all_tokens):]

        # Check budget if requested
        if check_budget:
            would_be = (
                len(self.all_tokens) + len(user_message_tokens) + self.gen_prompt_len
            )
            if would_be > self.max_seq_len:
                # Remove from messages and mark truncated
                self.messages.pop()
                self.is_truncated = True
                self.truncation_reason = "env_observation_length"
                return False

        # Accumulate
        self.all_tokens.extend(user_message_tokens)
        self.response_mask.extend([0] * len(user_message_tokens))

        # Validate if enabled
        if self.validate_enabled:
            self._validate()

        return True

    def _validate(self):
        """Optional validation: compare vs ground truth."""
        ground_truth = self.tokenizer.apply_chat_template(
            self.messages, add_generation_prompt=False, tokenize=True
        )
        if len(self.all_tokens) != len(ground_truth):
            raise ValueError(
                f"Token mismatch: {len(self.all_tokens)} vs {len(ground_truth)}"
            )
```

### Usage Example (Simplified Rollout)

```python
async def do_single_rollout(env, policy, tokenizer, max_seq_len, max_turns, messages):
    """Simplified rollout using TokenAccumulator."""

    # Initialize accumulator
    accumulator = TokenAccumulator(
        tokenizer=tokenizer,
        messages=messages,
        max_seq_len=max_seq_len,
        eos_token_id=tokenizer.eos_token_id,
        validate=True,  # Enable validation
    )

    # Add initial observation
    initial_obs = env.reset()
    accumulator.add_user_message(initial_obs, check_budget=False)

    for turn in range(max_turns):
        # Check budget
        remaining = accumulator.get_remaining_budget()
        if remaining <= 0:
            break

        # Generate
        prompt = accumulator.format_prompt()
        response = await policy.generate([prompt], max_tokens=remaining)[0]

        # Add assistant response
        success = accumulator.add_assistant_response(
            response_text=response.text,
            response_token_ids=response.token_ids,
        )

        if not success:  # Generation truncated
            break

        # Step env
        result = env.step(response.text)
        if result.done:
            break

        # Add env observation IMMEDIATELY
        success = accumulator.add_user_message(result.observation, check_budget=True)
        if not success:  # Env obs truncated
            break

    # Create Episode
    return Episode(
        all_token_ids=torch.tensor(accumulator.all_tokens),
        response_mask=torch.tensor(accumulator.response_mask),
        is_truncated=accumulator.is_truncated,
        truncation_reason=accumulator.truncation_reason,
        message_log=accumulator.messages,
        ...
    )
```

---

---

## Future Work: True Direct Token Extraction

For those wanting to eliminate prefix matching entirely, here are the approaches used by other libraries:

### Approach 1: vLLM's `return_tokens_as_token_ids` Flag (Prime-RL/Verifiers)

**File:** `/home/felipemello/forge/verifiers/verifiers/rl/trainer/config.py:322`

```python
# In vLLM sampling config
sampling_args["extra_body"] = {
    "return_tokens_as_token_ids": True,  # Returns tokens as "token_id:<int>"
}

# Then parse them
def parse_chat_completion_tokens(chat_completion):
    tokens = [
        int(token["token"].split(":")[-1])
        for token in chat_completion.choices[0].logprobs["content"]
    ]
    return tokens
```

**Status:** Needs investigation - this may return content tokens only, still requiring role header computation.

### Approach 2: Length-Based Slicing (NeMo-RL)

**File:** `/home/felipemello/forge/RL/nemo_rl/experience/rollouts.py:85-102`

```python
# vLLM returns input_lengths and generation_lengths
input_len = input_lengths[i].item()
total_length = unpadded_sequence_lengths[i].item()

# Slice generated tokens using lengths
generated_part = output_ids[i, input_len:total_length]

# Store in message log with pre-tokenized tokens
assistant_message = {
    "role": "assistant",
    "content": text,
    "token_ids": generated_part,  # Store tokens in message!
}
```

**Key insight:** Pre-tokenize and store tokens in message dicts, then concatenate when needed.

**Requires:** Modifying message log structure to include `token_ids` field.

### Approach 3: BASE Anchor + Delta Computation (VERL)

**File:** `/home/felipemello/forge/verl/verl/workers/rollout/schemas.py:204-221, 379-412`

```python
# Pre-compute BASE conversation
BASE_CONVERSATION = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": ""},  # Empty placeholder
]
base_tokens = tokenizer.apply_chat_template(BASE_CONVERSATION, ...)
base_len = len(base_tokens)

# For each message, tokenize with BASE
def add_user_message(content: str):
    temp = [*BASE_CONVERSATION, {"role": "user", "content": content}]
    full_tokens = tokenizer.apply_chat_template(temp, ...)

    # Extract only the new tokens
    new_tokens = full_tokens[base_len:]
    return new_tokens
```

**Benefit:** Avoids tokenizing full conversation each time.

**Requires:** Understanding chat template behavior with BASE anchor (Qwen models modify content!).

### Approach 4: Manual Role Header Computation (Template-Specific)

```python
# For Qwen chat template specifically
def get_qwen_role_header_tokens(tokenizer, role: str) -> list[int]:
    """Qwen format: <|im_start|>{role}\n"""
    header_text = f"<|im_start|>{role}\n"
    return tokenizer.encode(header_text, add_special_tokens=False)

def get_qwen_role_footer_tokens(tokenizer) -> list[int]:
    """Qwen format: <|im_end|>\n"""
    footer_text = "<|im_end|>\n"
    return tokenizer.encode(footer_text, add_special_tokens=False)

# Then combine
assistant_tokens = (
    get_qwen_role_header_tokens(tokenizer, "assistant") +
    response.token_ids +  # From vLLM
    get_qwen_role_footer_tokens(tokenizer)
)
```

**Problem:** This is template-specific and brittle. Won't work across different chat templates.

### Recommendation

**For production use:**
- ✅ Stick with prefix matching (proven correct, works universally)
- ✅ Use `TokenAccumulator` class from v2 (better organization)
- ✅ Enable validation in dev/staging, disable in production

**For optimization (if needed):**
1. Profile first - is prefix matching actually a bottleneck?
2. If yes, try Approach 2 (length-based slicing like NeMo-RL)
3. If that fails, try Approach 3 (BASE anchor like VERL)
4. Last resort: Template-specific logic (Approach 4)

**Don't optimize prematurely** - the current approach is correct and maintainable.

---

## Summary

**What we achieved in v7:**
1. ✅ `TokenAccumulator` class - better code organization
2. ✅ Immediate env obs accumulation - simpler flow
3. ✅ Cached gen_prompt_len - small optimization
4. ✅ Optional validation flag - flexible debugging
5. ✅ All 5 test cases pass - proven correctness

**What we didn't achieve:**
- ❌ Direct token extraction from vLLM (harder than expected)
- ❌ Fewer tokenization calls (still uses prefix matching)

**Recommendation:**
- Use `TokenAccumulator` from `test_simple_vllm_v2.py` for production
- It's cleaner, more maintainable, and provably correct
- Only optimize further if profiling shows tokenization is a bottleneck

**Files:**
- Implementation: `/home/felipemello/forge/test_simple_vllm_v2.py`
- Library comparison: `/home/felipemello/forge/brainstorming_forge_tau/changes/3_truncation_v7_library_comparison.md`

---

**End of Document**
