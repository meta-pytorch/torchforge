# Truncation V8: Qwen Think Tags Deep Dive

**Date:** 2025-01-17
**Focus:** Debugging multi-turn token accumulation with Qwen's `<think>` tags
**Status:** ⚠️ IN PROGRESS - Duplicate tags issue found

---

## Executive Summary

While investigating budget overflow issues in multi-turn RL rollouts, we discovered:

1. ✅ **Budget calculation bug fixed:** Using `assistant_overhead` instead of `gen_prompt_len`
2. ❌ **Duplicate `<think>` tags:** Qwen's chat template auto-wraps content, causing duplicates
3. 🔍 **Root cause:** BASE_CHAT_HISTORY anchor includes empty `<think>` wrapper
4. 📚 **VERL comparison:** Industry uses direct token extraction, we use delta tokenization

---

## Table of Contents

1. [Initial Bug Discovery](#initial-bug-discovery)
2. [Budget Calculation Fix (v1)](#budget-calculation-fix-v1)
3. [VERL Investigation](#verl-investigation)
4. [Qwen's enable_thinking Parameter](#qwens-enable_thinking-parameter)
5. [Duplicate Think Tags Issue](#duplicate-think-tags-issue)
6. [Current Status](#current-status)

---

## Initial Bug Discovery

### Symptom

```
[do_single_rollout] Turn 1
  Remaining budget: 404
  Current tokens: 1641
  Max seq len: 2048
  Calling vLLM with max_tokens=404

  vLLM returned 404 tokens
[TokenAccumulator.add_assistant_response]
  vLLM content tokens: 404
  Assistant tokens (with headers): 413
  Role header overhead: 9
  After: all_tokens=2054, is_truncated=True
  ❌ EXCEEDED max_seq_len by 6 tokens!
```

**Math:**
- We calculated: `remaining = 2048 - 1641 - 3 = 404`
- vLLM generated: 404 tokens
- Added to accumulator: 404 + 9 = 413 tokens
- Total: 1641 + 413 = 2054 > 2048 ❌

### Question Asked

"Why does this work in `test_simple_vllm_v2.py` but not in `main_v2.py`?"

**Answer:** Both were broken! The test used Llama-3.1-8B where the overhead happened to be 4 tokens for both `gen_prompt_len` and actual overhead. When we switched to Qwen3, the mismatch became visible.

---

## Budget Calculation Fix (v1)

### Root Cause

The old `get_generation_prompt_len()` calculated **prompt-side overhead only**:

```python
# OLD (WRONG)
def get_generation_prompt_len(tokenizer) -> int:
    messages = [{"role": "user", "content": "x"}]
    without_gen = tokenize(messages, add_generation_prompt=False)
    # Result: [user_tokens]

    with_gen = tokenize(messages, add_generation_prompt=True)
    # Result: [user_tokens, <|im_start|>assistant\n]

    return len(with_gen) - len(without_gen)  # = 3 for Qwen
```

This only captures the **generation prompt** added before vLLM generates, not the full overhead when accumulating the response.

### The Fix

```python
# NEW (CORRECT v1)
def get_assistant_overhead(tokenizer) -> int:
    """Get FULL overhead including role headers + EOS token."""
    base = [
        {"role": "system", "content": ""},
        {"role": "user", "content": ""},
    ]
    base_tokens = tokenizer.apply_chat_template(
        base, add_generation_prompt=False, tokenize=True
    )

    # Empty assistant response
    with_assistant = base + [{"role": "assistant", "content": ""}]
    full_tokens = tokenizer.apply_chat_template(
        with_assistant, add_generation_prompt=False, tokenize=True
    )

    return len(full_tokens) - len(base_tokens)  # = 9 for Qwen3
```

**Comparison:**

| Tokenizer | gen_prompt_len | assistant_overhead | Difference |
|-----------|----------------|-------------------|------------|
| Llama-3.1-8B | 4 | 4 | 0 (accidentally works!) |
| Qwen2.5-3B | 3 | 5 | 2 tokens |
| Qwen3-1.7B | 3 | 9 | 6 tokens |

**Budget calculation:**
```python
# OLD (wrong)
remaining = max_seq_len - current_tokens - gen_prompt_len
# For Qwen3: 2048 - 1641 - 3 = 404
# vLLM generates 404, adds 9 overhead → 1641 + 413 = 2054 > 2048 ❌

# NEW (correct)
remaining = max_seq_len - current_tokens - assistant_overhead
# For Qwen3: 2048 - 1641 - 9 = 398
# vLLM generates 398, adds 9 overhead → 1641 + 407 = 2048 ✅
```

---

## VERL Investigation

### Why Look at VERL?

After finding the duplicate `<think>` tags, we questioned whether our **prefix matching approach** was fundamentally wrong. From the library comparison doc:

> **🔑 CRITICAL INSIGHT: Most libraries use `response.token_ids` DIRECTLY from vLLM, NOT prefix matching!**

This led us to investigate how VERL handles Qwen without bugs.

### VERL's Architecture

**File:** `/home/felipemello/forge/verl/verl/workers/rollout/schemas.py`

```python
# Lines 31-34: BASE conversation anchor
BASE_CHAT_HISTORY = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "I am a user."}
]

# Lines 204-221: Pre-compute offsets at initialization
base_conv_wo_gen_prompt_end_pos = len(tokenizer.apply_chat_template(
    BASE_CHAT_HISTORY,
    add_generation_prompt=False,
    tokenize=True
))

base_conv_with_gen_prompt_end_pos = len(tokenizer.apply_chat_template(
    BASE_CHAT_HISTORY + [{"role": "assistant", "content": ""}],
    add_generation_prompt=False,
    tokenize=True
))
```

### VERL's Token Flow (with `skip_tokenizer_init=True`)

**Step 1: Add user message (delta tokenization)**
```python
# Lines 379-393
def add_user_message(self, processing_class, content: str):
    self.messages.append(Message(role="user", content=content))

    # Tokenize ONLY the new message using BASE anchor
    messages = [*BASE_CHAT_HISTORY, self.messages[-1]]
    content_ids = self._handle_apply_chat_template(
        processing_class,
        messages,
        add_generation_prompt=False,
        tokenize=True
    )[..., self.base_conv_wo_gen_prompt_end_pos:]  # Slice from pre-computed offset!

    self._update_input_ids(processing_class, content_ids, loss_mask=False)
```

**Step 2: Generate**
```python
# Lines 1053-1075: Generate with engine
generation_prompt_ids = _req.get_generation_prompt_ids(self.processing_class)
output = await self._engine.async_generate(
    input_ids=generation_prompt_ids,
    sampling_params=kwargs,
    return_logprob=return_logprob,
)
```

**Step 3: Add assistant response (direct extraction)**
```python
# Lines 910-918
if self.config.skip_tokenizer_init:
    content_ids = output["output_ids"]  # DIRECT from engine!
    content = self.processing_class.decode(content_ids, skip_special_tokens=True)
else:
    content_ids = None  # Will use delta tokenization fallback
    content = output["text"]

# Lines 395-412
def add_assistant_message(self, processing_class, content: str, content_ids: Optional[torch.Tensor] = None):
    self.messages.append(Message(role="assistant", content=content, ...))

    if content_ids is None:  # Fallback if engine doesn't provide token IDs
        messages = [*BASE_CHAT_HISTORY, self.messages[-1]]
        content_ids = self._handle_apply_chat_template(
            processing_class,
            messages,
            add_generation_prompt=False,
            tokenize=True
        )[..., self.base_conv_with_gen_prompt_end_pos:]  # Slice from offset!

    self._update_input_ids(processing_class, content_ids, loss_mask=True)
```

### Key Difference: VERL vs Our Approach

**VERL (Direct Token Extraction):**
```python
# 1. Generate
gen_prompt = tokenize(messages, add_generation_prompt=True)
# = [...system..., ...user..., <|im_start|>assistant\n]

output = engine.generate(gen_prompt)
# output["output_ids"] = [content_tokens..., <|im_end|>]

# 2. Accumulate generation prompt tokens (role headers)
gen_prompt_tokens = gen_prompt[base_with_gen_prompt_end_pos:]
input_ids.extend(gen_prompt_tokens)  # loss_mask=False

# 3. Accumulate output tokens
input_ids.extend(output["output_ids"])  # loss_mask=True

# Final: [...system..., ...user..., <|im_start|>assistant\n, content..., <|im_end|>]
```

**Our Approach (Delta Tokenization):**
```python
# 1. Generate
prompt_text = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=False
)
response = vLLM.generate(prompt_text)
# response.text = "<think>Okay...</think>"

# 2. Re-tokenize full assistant message
temp_messages = [*BASE_CHAT_HISTORY, {"role": "assistant", "content": response.text}]
full_tokens = tokenizer.apply_chat_template(
    temp_messages,
    add_generation_prompt=False,
    tokenize=True
)

# 3. Extract delta
assistant_delta = full_tokens[base_len_wo_gen:]
all_tokens.extend(assistant_delta)

# Final: [...system..., ...user..., <|im_start|>assistant\n<think>...</think>, content..., <|im_end|>]
```

### Why VERL Works and We Don't (Initially)

**VERL:** Splits response into:
- Generation prompt tokens (added before generation)
- Engine output tokens (added after generation)
- These are kept separate and never re-tokenized

**Us:** Re-apply chat template to full response:
- This re-tokenizes the response through the template
- Template has special handling for `<think>` tags
- If we use empty content for overhead calculation, template auto-adds wrappers

### Concrete Example

**User message:** "Hi"

**VERL Flow:**
```python
# Generation prompt
gen_prompt = tokenize([system, user, "Hi"], add_gen_prompt=True)
# = [1,2,3, 100,101, 151644,77091,198]
#    system  "Hi"    <|im_start|>assistant\n

# Engine generates (continues from prompt)
output["output_ids"] = [9906, 151645]  # "Hello<|im_end|>"

# Accumulate
input_ids = [1,2,3, 100,101, 151644,77091,198, 9906,151645]
#            system  "Hi"    role_header      "Hello"<|im_end|>
```

**Our Flow:**
```python
# Generate
response.text = "Hello"

# Re-tokenize [BASE + assistant]
messages = [BASE, {"role": "assistant", "content": "Hello"}]
full_tokens = tokenize(messages, add_gen_prompt=False)
# = [1,2,3, 151644,77091,198, 9906, 151645]
#    system  <|im_start|>assistant\n  "Hello" <|im_end|>

# Extract delta
assistant_delta = full_tokens[len(base):]
# = [151644,77091,198, 9906, 151645]

# Accumulate
all_tokens.extend([100,101])  # "Hi" (added earlier)
all_tokens.extend(assistant_delta)
# Final: [1,2,3, 100,101, 151644,77091,198, 9906, 151645]
#         system  "Hi"    role_header      "Hello"<|im_end|>
```

**Both produce IDENTICAL results!** The difference is:
- VERL never re-tokenizes (more efficient)
- We re-tokenize (handles complex templates correctly)

### Why Our Approach Is Actually Correct for Qwen

From TEST CASE 7 output (lines 430-486 in out5.txt):

```
APPROACH 1: PREFIX MATCHING (OUR CURRENT IMPLEMENTATION)
  Decoded: '<|im_start|>assistant
<think>

</think>

<think>
Okay, let<|im_end|>'

APPROACH 2: DIRECT EXTRACTION (TRL, VERL, PRIME-RL, etc.)
  Decoded: '<|im_start|>assistant
<think>

</think>

<|im_end|>     ← End token in the MIDDLE!
<think>
Okay, let'
```

**Direct extraction produces INVALID output** for Qwen because the template has special `<think>` tag handling. When we concatenate `role_header + content_tokens`, we bypass this handling.

**Conclusion:** Our prefix matching approach is correct for Qwen. The issue is the overhead calculation, not the approach.

---

## Qwen's enable_thinking Parameter

### Discovery

Qwen's tokenizer has an `enable_thinking` parameter that controls `<think>` wrapper behavior:

```bash
python3 -c "
from vllm.transformers_utils.tokenizer import get_tokenizer
tokenizer = get_tokenizer('Qwen/Qwen3-1.7B')

base = [{'role': 'system', 'content': ''}, {'role': 'user', 'content': ''}]

# Test 1: Generation prompt with enable_thinking=True
tokens_gen_on = tokenizer.apply_chat_template(
    base, add_generation_prompt=True, enable_thinking=True, tokenize=True
)
print('Gen prompt (thinking=True):', tokenizer.decode(tokens_gen_on))

# Test 2: Generation prompt with enable_thinking=False
tokens_gen_off = tokenizer.apply_chat_template(
    base, add_generation_prompt=True, enable_thinking=False, tokenize=True
)
print('Gen prompt (thinking=False):', tokenizer.decode(tokens_gen_off))

# Test 3: Accumulation with empty content (thinking=True)
msgs = base + [{'role': 'assistant', 'content': ''}]
tokens_empty_on = tokenizer.apply_chat_template(
    msgs, add_generation_prompt=False, enable_thinking=True, tokenize=True
)
print('Empty assistant (thinking=True):', tokenizer.decode(tokens_empty_on))

# Test 4: Accumulation with empty content (thinking=False)
tokens_empty_off = tokenizer.apply_chat_template(
    msgs, add_generation_prompt=False, enable_thinking=False, tokenize=True
)
print('Empty assistant (thinking=False):', tokenizer.decode(tokens_empty_off))
"
```

**Output:**
```
1. Empty assistant (enable_thinking=True):
   '<|im_start|>system\n<|im_end|>\n<|im_start|>user\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n<|im_end|>\n'

2. Empty assistant (enable_thinking=False):
   '<|im_start|>system\n<|im_end|>\n<|im_start|>user\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n<|im_end|>\n'

3. Assistant with content "Hello" (enable_thinking=True):
   '<|im_start|>assistant\n<think>\n\n</think>\n\nHello<|im_end|>\n'

4. Generation prompt (enable_thinking=True):
   '<|im_start|>assistant\n'

5. Generation prompt (enable_thinking=False):
   '<|im_start|>assistant\n<think>\n\n</think>\n\n'
```

### Key Findings

1. **For accumulation (`add_generation_prompt=False`):** Both `enable_thinking=True/False` produce **identical output** with empty content - both auto-add `<think>\n\n</think>\n\n` wrapper!

2. **For generation prompt (`add_generation_prompt=True`):**
   - `enable_thinking=True`: No wrapper (just `<|im_start|>assistant\n`)
   - `enable_thinking=False`: Adds wrapper

3. **Content preservation:** When content already has `<think>` tags, both settings preserve them correctly:

```bash
python3 -c "
from vllm.transformers_utils.tokenizer import get_tokenizer
tokenizer = get_tokenizer('Qwen/Qwen3-1.7B')

base = [{'role': 'system', 'content': ''}, {'role': 'user', 'content': ''}]
msgs = base + [{'role': 'assistant', 'content': '<think>\nHello\n</think>'}]

tokens = tokenizer.apply_chat_template(msgs, add_generation_prompt=False, enable_thinking=True, tokenize=True)
print(tokenizer.decode(tokens))
"
```

**Output:**
```
'<|im_start|>system\n<|im_end|>\n<|im_start|>user\n<|im_end|>\n<|im_start|>assistant\n<think>\nHello\n</think>\n\n<|im_end|>\n'
```

✅ Preserves the `<think>` tags correctly, no duplicates!

---

## Duplicate Think Tags Issue

### The Problem

From `out5.txt` (lines 88-100):

```
<|im_start|>assistant
<think>          ← Empty wrapper (shouldn't be here!)

</think>

<think>          ← Actual vLLM generation
Okay, let's see. The user has a hand of 15...
```

### Hypothesis 1: Overhead Calculation

**Original approach (v1):**
```python
def get_assistant_overhead(tokenizer) -> int:
    base = [{"role": "system", "content": ""}, {"role": "user", "content": ""}]
    base_tokens = tokenizer.apply_chat_template(base, add_generation_prompt=False, tokenize=True)

    # Empty assistant response
    with_assistant = base + [{"role": "assistant", "content": ""}]
    full_tokens = tokenizer.apply_chat_template(with_assistant, add_generation_prompt=False, tokenize=True)

    return len(full_tokens) - len(base_tokens)  # = 9 for Qwen3
```

**Decoded:**
```
'<|im_start|>assistant\n<think>\n\n</think>\n\n<|im_end|>\n'
```

The overhead (9 tokens) includes the auto-added `<think>\n\n</think>\n\n` wrapper!

**Attempted fix (v2):**
```python
def get_assistant_overhead(tokenizer) -> int:
    base = [{"role": "system", "content": ""}, {"role": "user", "content": ""}]
    base_tokens = tokenizer.apply_chat_template(base, add_generation_prompt=False, tokenize=True)

    # Use content with think tags to avoid auto-wrapper
    with_assistant = base + [{"role": "assistant", "content": "<think>X</think>"}]
    full_tokens = tokenizer.apply_chat_template(with_assistant, add_generation_prompt=False, tokenize=True)

    # Subtract the content tokens
    content_only = tokenizer.encode("<think>X</think>", add_special_tokens=False)
    overhead = len(full_tokens) - len(base_tokens) - len(content_only)

    return overhead  # = 8 for Qwen3
```

**Test result:**
```bash
OLD overhead (empty content): 9
NEW overhead (with think tags): 8
Difference: 1 tokens
```

But from `out5.txt` line 410-411:
```
Total tokens added (with headers): 161
Role header overhead: 9         ← STILL 9 when accumulating!
```

**The issue:** `tokenizer.encode("<think>X</think>")` tokenizes differently than how it appears inside `apply_chat_template()`. Inside the template, it becomes `<think>\nX\n</think>\n\n` (with newlines).

### Hypothesis 2: BASE_CHAT_HISTORY Anchor

Looking at our BASE_CHAT_HISTORY setup:

```python
# In __init__
self.BASE_CHAT_HISTORY = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": ""},  # Empty user message
]

self.base_len_wo_gen = len(tokenizer.apply_chat_template(
    self.BASE_CHAT_HISTORY,
    add_generation_prompt=False,
    tokenize=True,
))
```

When we extract assistant delta:

```python
temp_messages = [*self.BASE_CHAT_HISTORY, {"role": "assistant", "content": response_text}]
full_with_assistant = tokenizer.apply_chat_template(
    temp_messages,
    add_generation_prompt=False,
    tokenize=True,
)
assistant_tokens = full_with_assistant[self.base_len_wo_gen:]
```

**The question:** Does `BASE_CHAT_HISTORY` include the empty `<think>` wrapper when we tokenize it?

**Test:**
```bash
python3 -c "
from vllm.transformers_utils.tokenizer import get_tokenizer
tokenizer = get_tokenizer('Qwen/Qwen3-1.7B')

BASE = [{'role': 'system', 'content': ''}, {'role': 'user', 'content': ''}]
base_tokens = tokenizer.apply_chat_template(BASE, add_generation_prompt=False, tokenize=True)

# With vLLM response
with_resp = BASE + [{'role': 'assistant', 'content': '<think>Hello</think>'}]
full_tokens = tokenizer.apply_chat_template(with_resp, add_generation_prompt=False, tokenize=True)

print(f'BASE length: {len(base_tokens)}')
print(f'BASE decoded: {repr(tokenizer.decode(base_tokens))}')
print(f'Full length: {len(full_tokens)}')
print(f'Full decoded: {repr(tokenizer.decode(full_tokens))}')
print(f'Delta: {full_tokens[len(base_tokens):]}')
print(f'Delta decoded: {repr(tokenizer.decode(full_tokens[len(base_tokens):]))}')
"
```

This will show us if the delta includes unwanted empty wrappers.

---

## Current Status

### What Works
- ✅ Test validation passes (all_tokens matches ground_truth)
- ✅ Budget calculation uses correct overhead value
- ✅ Token accumulation is accurate (no missing tokens)

### What's Broken
- ❌ Duplicate `<think>` tags in decoded output
- ❌ Empty `<think>\n\n</think>\n\n` wrapper appearing before actual content
- ❌ Budget still exceeds by 1 token in TEST CASE 6

### Evidence from out5.txt

**Lines 88-100 (Duplicate tags):**
```
<|im_start|>assistant
<think>

</think>

<think>
Okay, let's see...
```

**Lines 410-421 (Budget overflow):**
```
Assistant overhead: 8
vLLM generated: 152 tokens
Total tokens added: 161
Role header overhead: 9    ← Actual is 9, not 8!
❌ BUDGET EXCEEDED by 1 token
```

**Lines 514-525 (Multi-turn duplicates):**
```
<|im_start|>assistant
<think>
Okay, let<|im_end|>
<|im_start|>user
Hand: 16, Dealer: 10<|im_end|>
<|im_start|>assistant
<think>

</think>

<think>
Okay, let<|im_end|>
```

---

## Next Debugging Steps

1. ✅ Test if `BASE_CHAT_HISTORY` tokenization includes empty wrapper
2. ⚠️ Investigate where the empty `<think></think>` comes from during delta extraction
3. ⚠️ Fix overhead calculation to return 9 instead of 8
4. ⚠️ Decide: Keep prefix matching or switch to direct extraction?

---

## Code Locations

- Test file: `/home/felipemello/forge/test_simple_vllm_v2.py`
- Main training: `/home/felipemello/forge/apps/blackjack/main_v2.py`
- Config: `/home/felipemello/forge/apps/blackjack/qwen3_1_7b.yaml`
- Library comparison: `/home/felipemello/forge/brainstorming_forge_tau/changes/3_truncation_v7_library_comparison.md`
- VERL schemas: `/home/felipemello/forge/verl/verl/workers/rollout/schemas.py`
- VERL rollout: `/home/felipemello/forge/verl/verl/workers/rollout/sglang_rollout/sglang_rollout.py`

---

## Key Learnings

1. **Budget calculation:** Must account for FULL overhead (role headers + EOS), not just generation prompt
2. **Model-specific behavior:** Llama vs Qwen have different overhead values; tests must use production model
3. **Qwen's think tags:** Template auto-wraps empty content in `<think></think>`, causing overhead calculation issues
4. **VERL's approach:** Direct token extraction avoids re-tokenization but requires careful role header handling
5. **Prefix matching trade-offs:** Handles complex templates correctly but requires precise overhead calculation
6. **Test robustness:** Using different models in test vs production masked the bug initially

---

**STATUS:** Investigation ongoing - need to determine source of empty `<think></think>` wrapper in delta extraction.

**Symptom:**
```
[do_single_rollout] Turn 1
  Remaining budget: 404
  vLLM returned 404 tokens

[TokenAccumulator.add_assistant_response]
  vLLM content tokens: 404
  Assistant tokens (with headers): 413
  Role header overhead: 9
  After: all_tokens=2054, is_truncated=True
  ❌ EXCEEDED max_seq_len by 6 tokens!
```

**Root Cause:**

The old `get_generation_prompt_len()` calculated:
```python
# Calculates prompt-side overhead only
messages = [{"role": "user", "content": "x"}]
without_gen = tokenize(messages, add_generation_prompt=False)  # [tokens]
with_gen = tokenize(messages, add_generation_prompt=True)       # [tokens, <|im_start|>assistant\n]
gen_prompt_len = len(with_gen) - len(without_gen)  # = 3 for Qwen
```

This gives **only the prompt-side assistant header** (`<|im_start|>assistant\n`), but not the full overhead when accumulating responses.

**The Fix (v1):**

```python
def get_assistant_overhead(tokenizer) -> int:
    """Get FULL overhead including role headers + EOS token."""
    base = [
        {"role": "system", "content": ""},
        {"role": "user", "content": ""},
    ]
    base_tokens = tokenizer.apply_chat_template(base, add_generation_prompt=False, tokenize=True)

    # Empty assistant response
    with_assistant = base + [{"role": "assistant", "content": ""}]
    full_tokens = tokenizer.apply_chat_template(with_assistant, add_generation_prompt=False, tokenize=True)

    return len(full_tokens) - len(base_tokens)  # = 9 for Qwen
```

**Budget calculation:**
```python
# OLD (wrong)
remaining = max_seq_len - current_tokens - gen_prompt_len  # Uses 3
# Result: 2048 - 1641 - 3 = 404
# vLLM generates 404, adds 9 overhead → 1641 + 413 = 2054 > 2048 ❌

# NEW (correct)
remaining = max_seq_len - current_tokens - assistant_overhead  # Uses 9
# Result: 2048 - 1641 - 9 = 398
# vLLM generates 398, adds 9 overhead → 1641 + 407 = 2048 ✅
```

---

### Issue 2: Qwen's `enable_thinking` Parameter

**Discovery:**

Qwen's tokenizer has an `enable_thinking` parameter that controls `<think>` wrapper behavior:

```python
# Test with generation prompt (add_generation_prompt=True)
tokenize(messages, add_generation_prompt=True, enable_thinking=True)
# → '<|im_start|>assistant\n' (NO wrapper)

tokenize(messages, add_generation_prompt=True, enable_thinking=False)
# → '<|im_start|>assistant\n<think>\n\n</think>\n\n' (ADDS wrapper)

# Test with accumulation (add_generation_prompt=False, empty content)
tokenize([...assistant with ""], add_generation_prompt=False, enable_thinking=True)
# → '<|im_start|>assistant\n<think>\n\n</think>\n\n<|im_end|>\n'

tokenize([...assistant with ""], add_generation_prompt=False, enable_thinking=False)
# → '<|im_start|>assistant\n<think>\n\n</think>\n\n<|im_end|>\n' (SAME!)
```

**Key Insight:**
- For `add_generation_prompt=False` (accumulation), both settings produce the same output with empty content
- The template auto-adds `<think></think>` wrapper for empty assistant messages

**With content that already has think tags:**
```python
tokenize([...assistant with "<think>Hello</think>"], add_generation_prompt=False, enable_thinking=True)
# → '<|im_start|>assistant\n<think>\nHello\n</think>\n\n<|im_end|>\n' (Preserves tags ✅)

tokenize([...assistant with "<think>Hello</think>"], add_generation_prompt=False, enable_thinking=False)
# → '<|im_start|>assistant\n<think>\nHello\n</think>\n\n<|im_end|>\n' (Preserves tags ✅)
```

---

### Issue 3: Duplicate `<think>` Tags (CURRENT ISSUE)

**Symptom:**

From test output (`out5.txt`):
```
<|im_start|>assistant
<think>          ← Empty wrapper (shouldn't be here!)

</think>

<think>          ← Actual vLLM generation
Okay, let's see...
```

**The Problem:**

When computing overhead with **empty content**, the template adds `<think>\n\n</think>\n\n`:

```python
# Old approach
with_assistant = base + [{"role": "assistant", "content": ""}]
full_tokens = tokenize(with_assistant, add_generation_prompt=False)
# Result: [..., <|im_start|>assistant\n<think>\n\n</think>\n\n<|im_end|>\n]
overhead = len(full_tokens) - len(base_tokens)  # = 9 tokens
```

This overhead (9 tokens) includes the auto-added `<think>\n\n</think>\n\n` wrapper, which shouldn't be counted as overhead!

**The Fix (v2 - attempted):**

```python
def get_assistant_overhead(tokenizer) -> int:
    """Compute overhead WITHOUT the think wrapper."""
    base = [...]
    base_tokens = tokenizer.apply_chat_template(base, add_generation_prompt=False, tokenize=True)

    # Use content with think tags to avoid auto-wrapper
    with_assistant = base + [{"role": "assistant", "content": "<think>X</think>"}]
    full_tokens = tokenizer.apply_chat_template(with_assistant, add_generation_prompt=False, tokenize=True)

    # Subtract the content tokens
    content_only = tokenizer.encode("<think>X</think>", add_special_tokens=False)
    overhead = len(full_tokens) - len(base_tokens) - len(content_only)

    return overhead  # = 8 tokens (was 9)
```

**Result:**
```
OLD overhead (empty content): 9
NEW overhead (with think tags): 8
Difference: 1 token
```

But the decoded output still shows duplicates! This means the issue is elsewhere.

---

## Current Hypothesis: Generation Prompt Issue

The problem might be in `format_prompt()`:

```python
def format_prompt(self) -> str:
    """Format prompt for generation."""
    return self.tokenizer.apply_chat_template(
        self.messages,
        add_generation_prompt=True,
        tokenize=False,
        # ⚠️ Missing: enable_thinking parameter!
    )
```

**Hypothesis:**
1. If default `enable_thinking=True` → generation prompt = `<|im_start|>assistant\n` (no wrapper)
2. vLLM generates: `<think>Okay...</think>`
3. Accumulation extracts the full response including headers
4. But somewhere an empty `<think></think>` is being added

**Need to investigate:**
1. What is the actual generation prompt sent to vLLM?
2. What does vLLM's `output.text` contain? (raw response)
3. How does `add_assistant_response()` process it?

---

## Token Flow Comparison: VERL vs Our Approach

### VERL (Direct Token Extraction)

```python
# Step 1: Generate
gen_prompt = tokenize(messages, add_generation_prompt=True)
# = [..., <|im_start|>assistant\n]

output = engine.generate(gen_prompt)
# output["output_ids"] = [content_tokens..., <|im_end|>]

# Step 2: Accumulate generation prompt tokens
gen_prompt_tokens = gen_prompt[base_len:]  # Role headers
input_ids.extend(gen_prompt_tokens)  # loss_mask=False

# Step 3: Accumulate output
input_ids.extend(output["output_ids"])  # loss_mask=True
```

**Key:** They split the response into (role headers from prompt) + (content from engine).

### Our Approach (Delta Tokenization)

```python
# Step 1: Generate
prompt_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
response = vLLM.generate(prompt_text)
# response.text = "<think>Okay...</think>"
# response.token_ids = [content_tokens] (vLLM removes special tokens by default)

# Step 2: Re-tokenize full assistant message
temp_messages = [*BASE_CHAT_HISTORY, {"role": "assistant", "content": response.text}]
full_tokens = tokenizer.apply_chat_template(temp_messages, add_generation_prompt=False, tokenize=True)

# Step 3: Extract delta
assistant_delta = full_tokens[base_len_wo_gen:]
all_tokens.extend(assistant_delta)
```

**Key:** We re-apply chat template to get the full assistant message with proper formatting.

---

## Debugging Steps

### 1. Check what vLLM actually returns

```python
response = vLLM.generate(prompt_text)
print(f"response.text: {repr(response.text)}")
print(f"response.token_ids: {response.token_ids}")
```

### 2. Check the generation prompt

```python
prompt_text = accumulator.format_prompt()
print(f"Generation prompt:\n{prompt_text}")

# Also tokenize it to see the tokens
prompt_tokens = tokenizer.apply_chat_template(
    accumulator.messages,
    add_generation_prompt=True,
    tokenize=True,
)
print(f"Last 20 tokens: {prompt_tokens[-20:]}")
print(f"Decoded last part: {tokenizer.decode(prompt_tokens[-20:])}")
```

### 3. Check the delta extraction

```python
# In add_assistant_response
temp_messages = [*self.BASE_CHAT_HISTORY, {"role": "assistant", "content": response_text}]
full_with_assistant = tokenizer.apply_chat_template(temp_messages, add_generation_prompt=False, tokenize=True)

print(f"BASE_CHAT_HISTORY: {self.BASE_CHAT_HISTORY}")
print(f"base_len_wo_gen: {self.base_len_wo_gen}")
print(f"response_text: {repr(response_text)}")
print(f"full_with_assistant: {full_with_assistant}")
print(f"Decoded: {tokenizer.decode(full_with_assistant)}")
print(f"assistant_delta: {full_with_assistant[self.base_len_wo_gen:]}")
```

---

## Next Steps

1. ✅ Add debug logging to `format_prompt()` and `add_assistant_response()`
2. ✅ Test with explicit `enable_thinking=True` in `format_prompt()`
3. ✅ Verify that vLLM's response doesn't include the empty wrapper
4. ⚠️ Find where the duplicate `<think></think>` is coming from

---

## Code Locations

- Test file: `/home/felipemello/forge/test_simple_vllm_v2.py`
- Main training: `/home/felipemello/forge/apps/blackjack/main_v2.py`
- Config: `/home/felipemello/forge/apps/blackjack/qwen3_1_7b.yaml`
- Library comparison: `/home/felipemello/forge/brainstorming_forge_tau/changes/3_truncation_v7_library_comparison.md`

---

## Key Learnings

1. **Budget calculation:** Must account for FULL overhead (role headers + EOS), not just generation prompt
2. **Qwen's think tags:** Template auto-wraps empty content, causing issues with overhead calculation
3. **Prefix matching is correct:** For complex templates like Qwen, we NEED to re-apply chat template to handle special tokens
4. **VERL uses direct extraction:** Works for simpler templates but requires careful handling of role headers

---

**STATUS:** Investigation ongoing - duplicate `<think>` tags still appearing despite overhead fix.


----

appendix

python3 -c "
from vllm.transformers_utils.tokenizer import get_tokenizer
tokenizer = get_tokenizer('Qwen/Qwen3-1.7B')

BASE = [{'role': 'system', 'content': ''}, {'role': 'user', 'content': ''}]
base_tokens = tokenizer.apply_chat_template(BASE, add_generation_prompt=False, tokenize=True)

print('='*80)
print('TEST 1: Complete think tags (closing tag present)')
print('='*80)
with_complete = BASE + [{'role': 'assistant', 'content': '<think>\nHello\n</think>'}]
full = tokenizer.apply_chat_template(with_complete, add_generation_prompt=False, tokenize=True)
delta = full[len(base_tokens):]
print(f'Content: <think>\\nHello\\n</think>')
print(f'Delta decoded:\n{repr(tokenizer.decode(delta))}')

print('\n' + '='*80)
print('TEST 2: Incomplete think tags (NO closing tag - TRUNCATED)')
print('='*80)
with_incomplete = BASE + [{'role': 'assistant', 'content': '<think>\nHello'}]
full = tokenizer.apply_chat_template(with_incomplete, add_generation_prompt=False, tokenize=True)
delta = full[len(base_tokens):]
print(f'Content: <think>\\nHello (no closing tag)')
print(f'Delta decoded:\n{repr(tokenizer.decode(delta))}')

print('\n' + '='*80)
print('TEST 3: No think tags at all')
print('='*80)
with_none = BASE + [{'role': 'assistant', 'content': 'Hello'}]
full = tokenizer.apply_chat_template(with_none, add_generation_prompt=False, tokenize=True)
delta = full[len(base_tokens):]
print(f'Content: Hello')
print(f'Delta decoded:\n{repr(tokenizer.decode(delta))}')
"

EST 1: Complete think tags (closing tag present)
================================================================================
Content: <think>
Hello
</think>
Delta decoded:
'<|im_start|>assistant\n<think>\nHello\n</think>\n\n<|im_end|>\n'

================================================================================
TEST 2: Incomplete think tags (NO closing tag - TRUNCATED)
================================================================================
Content: <think>
Hello (no closing tag)
Delta decoded:
'<|im_start|>assistant\n<think>\n\n</think>\n\n<think>\nHello<|im_end|>\n'

================================================================================
TEST 3: No think tags at all
================================================================================
Content: Hello
Delta decoded:
'<|im_start|>assistant\n<think>\n\n</think>\n\nHello<|im_end|>\n'



  What's Happening

  TEST 1 (Complete <think> tags):
  Content: '<think>\nHello\n</think>'
  Delta: '<|im_start|>assistant\n<think>\nHello\n</think>\n\n<|im_end|>\n'
  ✅ CLEAN! No duplicates - template preserves complete tags correctly.

  TEST 2 (Incomplete <think> tags - TRUNCATED):
  Content: '<think>\nHello' (no closing tag!)
  Delta: '<|im_start|>assistant\n<think>\n\n</think>\n\n<think>\nHello<|im_end|>\n'
  ❌ DUPLICATE! Template adds empty wrapper <think>\n\n</think>\n\n BEFORE the incomplete tag!

  TEST 3 (No <think> tags):
  Content: 'Hello'
  Delta: '<|im_start|>assistant\n<think>\n\n</think>\n\nHello<|im_end|>\n'
  ✅ Expected - template wraps plain content in <think> tags.

  ---
  The Real Problem

  When vLLM truncates mid-generation, the response has:
  - Opening tag: <think>\n
  - Content: Okay, let's see...
  - NO closing tag (truncated before </think>)

  The chat template's logic:
  1. Sees incomplete <think> tag in content
  2. Thinks: "This is malformed, I need to add proper wrappers!"
  3. Adds: <think>\n\n</think>\n\n (empty wrapper) + keeps the original incomplete <think>\nOkay...
  4. Result: Duplicate tags!

  ---
  Why This Happens in Your Tests

  Looking at out5.txt:
  - Line 61-69: Response: '<think>\nOkay...' + ⚠️ Generation TRUNCATED
  - Lines 92-100: Duplicate tags appear

  The generation was truncated at 50 tokens (max_tokens limit), so the <think> tag was never closed!

 TEST CASE 6 (lines 389-421) shows a different problem:
  - NOT truncated (generation completed normally)
  - Budget overflow: assistant_overhead=8 but actual overhead is 9
  - This is the overhead calculation bug (v2 fix didn't work correctly)

  The duplicate tags issue only affects truncated responses, which are already being filtered out by your mask logic. The real bug to fix is getting the
  overhead calculation to return 9 instead of 8.
