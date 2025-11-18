# Truncation V9: Core Issue and Fix

**Date:** 2025-01-17
**Status:** Root cause identified, simple fix available

---

## The Problem

Your decoded conversations show duplicate `<think>` tags:

```
<|im_start|>assistant
<think>             ← AUTO-ADDED BY TEMPLATE
</think>

<think>             ← FROM VLLM
Okay, let's see...
```

**Root cause:** Your current implementation re-applies `chat_template` to get role headers, which triggers Qwen's auto-wrapping behavior on incomplete `<think>` tags.

---

## How VeRL Does It

**VeRL's approach:**

```python
# 1. Generate with engine
output = engine.generate(prompt)

# 2. Get FULL token sequence directly from engine (including role headers)
if skip_tokenizer_init:
    assistant_tokens = output["output_ids"]  # Contains: role_header + content + eos
else:
    # Fallback: re-tokenize via BASE anchor
    assistant_tokens = tokenize(BASE + [{"role": "assistant", "content": output["text"]}])[base_len:]
```

**Key:** VeRL's engine (SGLang) returns `output_ids` with role headers already included.

---

## Why You Can't Do the Exact Same

**VeRL's engine vs your vLLM:**

| What | VeRL (SGLang with skip_tokenizer_init) | Your vLLM |
|------|----------------------------------------|-----------|
| Returns | `[role_start, assistant, newline, content..., eos]` | `[content...]` only |
| Role headers | ✅ Included | ❌ Missing |
| Can use directly | ✅ Yes | ❌ No, need to add headers |

**Example:**
```python
# VeRL's engine returns:
[151644, 77091, 198, 151667, 271, 151668, 271, 151667, 198, 32313, 11, 1077, 151645]
# ^role  ^asst  ^nl  ^think ^nl  ^/think^nl  ^think ^nl  ^content...   ^eos

# Your vLLM returns:
[151667, 198, 32313, 11, 1077]
# ^think ^nl  ^content...
```

**You must add role headers separately.**

---

## Your Current Approach (Why It Creates Duplicates)

```python
# Current code (main_v2.py:261-298)
def add_assistant_response(response_text, response_token_ids):
    # 1. Add message to list
    self.messages.append({"role": "assistant", "content": response_text})

    # 2. Re-tokenize via chat template to get role headers
    temp_messages = [*BASE_CHAT_HISTORY, {"role": "assistant", "content": response_text}]
    full_with_assistant = tokenizer.apply_chat_template(temp_messages, tokenize=True)
    assistant_tokens = full_with_assistant[base_len:]  # Extract delta
```

**What happens when response_text = `"<think>\nOkay..."`** (incomplete, no closing tag):

1. Chat template sees incomplete `<think>` tag
2. Qwen's template logic: "malformed think tag, I'll add proper wrappers"
3. Outputs: `<think>\n\n</think>\n\n` + `<think>\nOkay...`
4. Result: **duplicate tags**

**Evidence from v8 appendix (lines 1010-1017):**
```
Content: '<think>\nHello' (no closing tag!)
Delta decoded:
'<|im_start|>assistant\n<think>\n\n</think>\n\n<think>\nHello<|im_end|>\n'
❌ DUPLICATE!
```

---

## The Simple Fix

**Use vLLM's `output.token_ids` directly + pre-computed role headers.**

### Step 1: Pre-compute role headers (one-time, at init)

```python
@lru_cache(maxsize=1)
def get_role_header_and_footer(tokenizer):
    """Get role header and footer tokens for assistant."""
    # Tokenize conversation with COMPLETE think tags (avoids auto-wrapper)
    base = [
        {"role": "system", "content": ""},
        {"role": "user", "content": ""},
    ]
    with_assistant = base + [{"role": "assistant", "content": "<think>X</think>"}]

    # Get full sequence
    full_tokens = tokenizer.apply_chat_template(with_assistant, tokenize=True)

    # Get base length
    base_len = len(tokenizer.apply_chat_template(base, tokenize=True))

    # Get content-only tokens
    content_tokens = tokenizer.encode("<think>X</think>", add_special_tokens=False)

    # Extract role tokens: full - base - content
    assistant_full = full_tokens[base_len:]

    # Find where content starts and ends
    # Role header = everything before content
    # Footer = everything after content (typically just eos)

    # Simple approach: header is first N tokens, footer is last M tokens
    # For Qwen: header ≈ 8 tokens, footer ≈ 1 token (eos)

    # More robust: search for content_tokens in assistant_full
    import numpy as np
    content_arr = np.array(content_tokens)
    assistant_arr = np.array(assistant_full)

    # Find content position
    for i in range(len(assistant_arr) - len(content_arr) + 1):
        if np.array_equal(assistant_arr[i:i+len(content_arr)], content_arr):
            header = assistant_full[:i].tolist()
            footer = assistant_full[i+len(content_arr):].tolist()
            return header, footer

    raise ValueError("Could not find content in assistant tokens")
```

### Step 2: Use direct tokens + headers

```python
def add_assistant_response(response_text, response_token_ids, response_logprobs):
    """
    Add assistant response using DIRECT token IDs from vLLM.

    This avoids re-applying chat template, which prevents Qwen's
    think-tag auto-wrapping behavior.
    """
    # Get pre-computed role headers
    role_header, role_footer = get_role_header_and_footer(self.tokenizer)

    # Combine: header + content (from vLLM) + footer
    assistant_tokens = role_header + response_token_ids + role_footer

    # Create logprobs: zeros for headers, actual for content
    assistant_logprobs = (
        [0.0] * len(role_header) +
        response_logprobs +
        [0.0] * len(role_footer)
    )

    # Check truncation (last content token != eos)
    is_truncated = (response_token_ids[-1] != self.eos_token_id)
    mask_value = 0 if is_truncated else 1

    # Accumulate
    self.all_tokens.extend(assistant_tokens)
    self.response_mask.extend([mask_value] * len(assistant_tokens))
    self.logprobs.extend(assistant_logprobs)

    # Add to messages (for next turn's prompt)
    self.messages.append({"role": "assistant", "content": response_text})

    return not is_truncated
```

---

## Why This Works

**Old approach:**
```
vLLM returns: [<think>, Okay]
↓ re-apply chat template
Chat template sees: "<think>\nOkay" (incomplete)
↓ auto-wraps
Result: [role_start, <think>, </think>, <think>, Okay, eos]
```

**New approach:**
```
vLLM returns: [<think>, Okay]
↓ prepend pre-computed header, append footer
Result: [role_start, <think>, Okay, eos]
No template re-application = no auto-wrapping
```

**Key insight:** By using vLLM's tokens directly and only adding static role headers, we never re-apply the chat template on vLLM's content, so Qwen's think-tag logic never triggers.

---

## Implementation

### Change 1: Update `get_assistant_overhead`

```python
# main_v2.py lines 134-167

@lru_cache(maxsize=1)
def get_assistant_overhead(tokenizer) -> tuple[int, list[int], list[int]]:
    """
    Get role header and footer tokens for assistant responses.

    Returns:
        (overhead_count, header_tokens, footer_tokens)
    """
    base = [{"role": "system", "content": ""}, {"role": "user", "content": ""}]
    base_tokens = tokenizer.apply_chat_template(base, tokenize=True)

    # Use complete think tags to avoid auto-wrapper
    with_assistant = base + [{"role": "assistant", "content": "<think>X</think>"}]
    full_tokens = tokenizer.apply_chat_template(with_assistant, tokenize=True)

    # Get content-only tokens
    content_tokens = tokenizer.encode("<think>X</think>", add_special_tokens=False)

    # Extract assistant portion
    assistant_full = full_tokens[len(base_tokens):]

    # Find content position
    import numpy as np
    for i in range(len(assistant_full) - len(content_tokens) + 1):
        if assistant_full[i:i+len(content_tokens)] == content_tokens:
            header = assistant_full[:i]
            footer = assistant_full[i+len(content_tokens):]
            overhead = len(header) + len(footer)
            return overhead, header, footer

    # Fallback: assume eos is footer, rest is header
    header = assistant_full[:-1]
    footer = assistant_full[-1:]
    overhead = len(assistant_full) - len(content_tokens)
    return overhead, header, footer
```

### Change 2: Update TokenAccumulator.__init__

```python
# main_v2.py lines 185-206

def __init__(self, tokenizer, messages, max_seq_len, eos_token_id, ...):
    self.tokenizer = tokenizer
    self.max_seq_len = max_seq_len
    self.eos_token_id = eos_token_id

    # Get role headers/footers
    overhead, self.role_header, self.role_footer = get_assistant_overhead(tokenizer)
    self.assistant_overhead = overhead

    # Rest of init...
```

### Change 3: Update add_assistant_response

```python
# main_v2.py lines 261-329

def add_assistant_response(self, response_text, response_token_ids, response_logprobs=None):
    """Add assistant response using DIRECT tokens from vLLM."""

    # Check truncation
    is_truncated = (len(response_token_ids) > 0 and
                   response_token_ids[-1] != self.eos_token_id)

    # Combine: header + vLLM content + footer
    assistant_tokens = self.role_header + response_token_ids + self.role_footer

    # Create logprobs
    num_content = len(response_token_ids)
    assistant_logprobs = [0.0] * len(self.role_header)
    if response_logprobs:
        assistant_logprobs.extend(response_logprobs)
    else:
        assistant_logprobs.extend([0.0] * num_content)
    assistant_logprobs.extend([0.0] * len(self.role_footer))

    # Accumulate
    mask_value = 0 if is_truncated else 1
    self.all_tokens.extend(assistant_tokens)
    self.response_mask.extend([mask_value] * len(assistant_tokens))
    self.logprobs.extend(assistant_logprobs)

    # Add to messages for next prompt
    self.messages.append({"role": "assistant", "content": response_text})

    if is_truncated:
        self.is_truncated = True
        self.truncation_reason = "generation_length"

    return not is_truncated
```

---

## Comparison: Old vs New

| Aspect | Old (Prefix Matching) | New (Direct Tokens) |
|--------|-----------------------|---------------------|
| Tokenization | Re-applies chat template every turn | Uses vLLM tokens + static headers |
| Think tag handling | ❌ Triggers auto-wrapper | ✅ No template re-application |
| Complexity | Medium (BASE anchor slicing) | Low (simple concatenation) |
| Matches VeRL | Partially (uses BASE anchor) | Yes (direct tokens + headers) |
| Token count | Exact (via finalize check) | Exact (pre-computed headers) |

---

## What About User Messages?

**User messages still use prefix matching** (unchanged):

```python
def add_user_message(self, content, check_budget=True):
    """Add user message using BASE anchor (unchanged)."""
    self.messages.append({"role": "user", "content": content})

    # Tokenize system + user to get delta
    temp_messages = [self.BASE_CHAT_HISTORY[0], {"role": "user", "content": content}]
    full_with_user = self.tokenizer.apply_chat_template(temp_messages, tokenize=True)
    user_message_tokens = full_with_user[self.system_len:]

    # ... budget check and accumulation
```

**Why this is fine:**
- User messages don't have think tags (no auto-wrapper issue)
- Content is under our control (from environment)
- Prefix matching is reliable here

---

## Summary

**How VeRL does it:** Direct token IDs from engine (which includes role headers).

**Why you can't do the exact same:** vLLM only returns content tokens, not role headers.

**The fix:** Use vLLM's content tokens directly + pre-computed static role headers.

**Why this fixes think tags:** No re-application of chat template = no auto-wrapping logic triggered.

**Code changes:** 3 small changes to `get_assistant_overhead`, `__init__`, and `add_assistant_response`.

---

**End of Document**
