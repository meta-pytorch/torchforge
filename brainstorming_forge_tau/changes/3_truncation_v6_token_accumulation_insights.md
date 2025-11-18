# Token Accumulation Insights - How to Fix V5

**Date:** 2025-01-16
**Context:** Understanding how to correctly accumulate tokens incrementally in multi-turn episodes

---

## The Critical Question

**When adding environment/tool responses to the conversation, should we:**
1. Tokenize just the content string: `tokenizer.encode(obs_text)`?
2. Use chat template on the new message: `tokenizer.apply_chat_template([new_message])`?
3. Re-tokenize the full conversation and extract the delta (prefix matching)?
4. Get token IDs from the generation engine response?

**Answer: It depends on the library, but there are THREE distinct patterns.**

---

## Pattern 1: Get Token IDs from Generation Response (TRL)

**Used by:** TRL, VERL SGLang Rollout (preferred mode)

**How it works:**
- The generation engine (vLLM) returns token IDs along with the text
- No need to tokenize again - just use what the engine provides
- **Most efficient** and **guaranteed to match** what the model saw

### TRL Example

**File:** `trl/examples/scripts/openenv/wordle.py:342-381`

```python
# Build prompt text
prompt_text = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=False,  # Get text, not tokens
)

# Call vLLM
vllm_result = request_vllm_completion(prompt_text, args, ...)

# Get token IDs from vLLM response
prompt_ids.extend(vllm_result["prompt_ids"])      # Prompt tokens
completion_ids.extend(vllm_result["completion_ids"])  # Response tokens
logprobs.extend(vllm_result["logprobs"])
```

### VERL SGLang Rollout Example

**File:** `verl/workers/rollout/sglang_rollout/sglang_rollout.py:910-915`

```python
if self.config.skip_tokenizer_init:
    # Use token IDs directly from engine
    content_ids = output["output_ids"]
    content = self.processing_class.decode(content_ids, skip_special_tokens=True)
else:
    # Fallback to prefix matching
    content = output["text"]
    content_ids = None  # Will trigger prefix matching
```

**Key advantage:** Zero tokenization overhead, perfect alignment with model.

**When to use:**
- During rollout with vLLM/SGLang server
- When engine returns token IDs
- For maximum efficiency

---

## Pattern 2: Prefix Matching with apply_chat_template (VERL, Verifiers)

**Used by:** VERL Tool Agent Loop, Verifiers

**How it works:**
- Re-tokenize the full conversation with `apply_chat_template`
- Compare with previous tokenization to extract only new tokens
- Relies on the **prefix property**: `tokenize([A, B])` starts with same tokens as `tokenize([A])`

### Verifiers Example

**File:** `verifiers/utils/processing_utils.py:129-145`

```python
# Tokenize conversation UP TO last completed turn
token_prefix = processing_class.apply_chat_template(
    conversation=messages_consumed,
    add_generation_prompt=False,
    tools=oai_tools,
)

# Tokenize WITH new messages added
token_prefix_with_turn = processing_class.apply_chat_template(
    conversation=messages_consumed + consecutive_messages,
    add_generation_prompt=True,
    tools=oai_tools,
)

# Assert prefix property holds
assert token_prefix_with_turn[:len(token_prefix)] == token_prefix

# Extract ONLY the new tokens
completion_turn_ids = token_prefix_with_turn[len(token_prefix):]
```

### VERL Tool Agent Loop Example

**File:** `verl/experimental/agent_loop/tool_agent_loop.py:355-375`

```python
# Tokenize tool response messages
response_ids = await self.loop.run_in_executor(
    None,
    lambda: self.tokenizer.apply_chat_template(
        add_messages,  # New tool/env messages
        add_generation_prompt=True,
        tokenize=True
    ),
)

# Strip the system prompt prefix
response_ids = response_ids[len(self.system_prompt):]

# Accumulate
agent_data.prompt_ids += response_ids
agent_data.response_mask += [0] * len(response_ids)  # Mark as observation
```

**Key advantage:** Guaranteed correctness - tokens match what `apply_chat_template` produces.

**When to use:**
- Offline processing / data preparation
- When you don't have access to engine token IDs
- When you need perfect chat template formatting

**Gotchas:**
- Prefix property can fail if tokenizer behavior is context-dependent
- Must keep `add_generation_prompt` consistent
- O(n²) complexity (re-tokenize growing conversation each turn)

---

## Pattern 3: Tokenize Each Message Independently (NeMo-RL)

**Used by:** NeMo-RL

**How it works:**
- Each message is tokenized separately and stores its own `token_ids`
- At training time, concatenate all `token_ids` from message log
- **Does NOT use `apply_chat_template` for environment responses**

### NeMo-RL Example

**File:** `RL/nemo_rl/experience/rollouts.py:446-477`

```python
# Get environment observation text
env_obs_content = env_output.observations[i]["content"]

# Tokenize the raw content (NO chat template!)
# TODO @sahilj: handle if we want these subsequent messages to have a chat template
tokenized_obs = tokenizer(
    env_obs_content,
    return_tensors="pt",
    add_special_tokens=False  # No special tokens
).input_ids[0]

# Store in message log
tokenized_env_obs_message = {
    "role": "environment",
    "content": env_obs_content,
    "token_ids": tokenized_obs,  # Raw tokens stored
}
current_batch["message_log"][global_idx].append(tokenized_env_obs_message)
```

**At training time** (`RL/nemo_rl/data/llm_message_utils.py:36-123`):

```python
def message_log_to_flat_messages(message_log):
    """Concatenate token_ids from all messages."""
    result = {"token_ids": []}

    for message in message_log:
        result["token_ids"].append(message["token_ids"])

    # Concatenate all token_ids tensors
    concat["token_ids"] = torch.cat(result["token_ids"])
    return concat
```

**Key insight:** Environment responses are tokenized as **raw text WITHOUT chat template formatting** (no role headers, turn separators, etc.)

**When to use:**
- When you want simplicity
- When environment responses don't need chat template formatting
- When you're okay with potentially missing special tokens between turns

**Gotchas:**
- Tokens may NOT match what `apply_chat_template` would produce for the full conversation
- Missing role markers and special tokens between turns
- There's even a TODO comment acknowledging this limitation

---

## The Critical Difference: `encode()` vs `apply_chat_template()`

### Example with Llama-3

```python
message = {"role": "user", "content": "Hand: 15, Dealer: 10"}

# Method 1: Encode content only
tokens_content = tokenizer.encode("Hand: 15, Dealer: 10", add_special_tokens=False)
# Result: [2367, 25, 220, 868, 11, 79289, 25, 220, 605]
#         [Hand :   1   5  ,   Dealer :   1   0 ]

# Method 2: Apply chat template
tokens_chat = tokenizer.apply_chat_template(
    [message],
    add_generation_prompt=False,
    tokenize=True
)
# Result: [128000, 128006, 882, 128007, 271, 2367, 25, 220, 868, 11, 79289, 25, 220, 605, 128009]
#         [BOS   ][start_header][user][end_header][nl][Hand: 15, Dealer: 10    ][eot_id]

# Method 3: Apply chat template with generation prompt
tokens_chat_gen = tokenizer.apply_chat_template(
    [message],
    add_generation_prompt=True,
    tokenize=True
)
# Result: [128000, 128006, 882, 128007, 271, 2367, 25, 220, 868, 11, 79289, 25, 220, 605, 128009, 128006, 78191, 128007, 271]
#         [BOS   ][start_header][user][end_header][nl][Hand: 15, Dealer: 10    ][eot_id][start_header][assistant][end_header][nl]
```

**Key differences:**
1. **BOS token** (`128000`) - only in chat template
2. **Role headers** (`<|start_header_id|>user<|end_header_id|>`) - only in chat template
3. **End-of-turn token** (`128009`) - only in chat template
4. **Generation prompt** (`<|start_header_id|>assistant<|end_header_id|>`) - only when `add_generation_prompt=True`

**This means:** If you tokenize just the content, you're missing 4-6 special tokens PER MESSAGE!

---

## What V5 Is Doing Wrong

Looking at `3_truncation_v5_simplified_env.md:349-360`:

```python
# After env.step(), tokenize and potentially truncate observation
if not result.done:
    messages.append(result.observation.message)

    # Tokenize and add to all_tokens
    obs_text = result.observation.message["content"]
    obs_tokens = tokenizer.encode(obs_text, add_special_tokens=False)

    # TODO: Add truncation for long observations if needed
    all_tokens.extend(obs_tokens)
    all_logprobs.extend([0.0] * len(obs_tokens))
    response_mask.extend([0] * len(obs_tokens))  # Don't train on env observations
```

**Problems:**
1. ❌ Tokenizes only the content string, not the full message with chat template
2. ❌ Missing role headers, turn separators, and special tokens
3. ❌ `all_tokens` won't match what the model actually sees next turn
4. ❌ Budget calculation will be WRONG (underestimating actual token count)

**Example of the mismatch:**

```python
# V5 current approach (WRONG):
obs_tokens = tokenizer.encode("Hand: 18, Dealer: Ace", add_special_tokens=False)
# [2367, 25, 220, 972, 11, 79289, 25, 42964]  (8 tokens)

# What the model ACTUALLY sees next turn when we call apply_chat_template:
prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
# Includes: [eot_id, start_header, user, end_header, nl, content, eot_id, start_header, assistant, end_header, nl]
# Total: 8 content tokens + 6 special tokens = 14 tokens!
```

**Impact:**
- Budget tracking is off by ~40% (missing 6 tokens per turn)
- Episode may exceed `max_seq_len` without detecting it
- Training data tokens don't match what model saw during generation

---

## How to Fix V5: Three Options

### Option A: Use vLLM Token IDs (RECOMMENDED - Most Efficient)

**Pattern:** Like TRL/VERL SGLang

**Change 1:** Get prompt token IDs from generation response

```python
# In do_single_rollout(), after generate
responses = await policy.generate.route(
    [prompt],
    sampling_params={"max_tokens": remaining}
)
response = responses[0]

# Get prompt tokens from response (if available)
if hasattr(response, 'prompt_token_ids'):
    prompt_tokens = response.prompt_token_ids
else:
    # Fallback: encode
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)

# Accumulate prompt + response
all_tokens.extend(prompt_tokens)
all_tokens.extend(response.token_ids)
response_mask.extend([0] * len(prompt_tokens))
response_mask.extend([1] * len(response.token_ids))
all_logprobs.extend([0.0] * len(prompt_tokens))
all_logprobs.extend(response.logprobs)
```

**Change 2:** For environment observations, use prefix matching

```python
# After env.step()
if not result.done:
    # Add observation to messages
    messages.append(result.observation.message)

    # Tokenize full conversation to get correct token count
    full_prompt = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        enable_thinking=False,
    )

    # Extract only the NEW tokens (env observation + special tokens)
    obs_tokens = full_prompt[len(all_tokens):]

    # Accumulate
    all_tokens.extend(obs_tokens)
    all_logprobs.extend([0.0] * len(obs_tokens))
    response_mask.extend([0] * len(obs_tokens))
```

**Pros:**
- ✅ Guaranteed correctness - tokens match what model sees
- ✅ Efficient - vLLM already computed prompt tokens
- ✅ Handles all special tokens automatically

**Cons:**
- Requires vLLM response to include `prompt_token_ids`
- Slightly more complex logic

---

### Option B: Full Prefix Matching (Most Correct)

**Pattern:** Like Verifiers

**Implementation:**

```python
# Track cumulative token count
cumulative_tokens = 0

for turn in range(max_turns):
    # Build prompt from messages
    prompt_text = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=False,
    )

    # Tokenize full conversation
    full_tokens = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        enable_thinking=False,
    )

    # Extract NEW tokens since last turn (prefix matching)
    new_prompt_tokens = full_tokens[cumulative_tokens:]
    cumulative_tokens = len(full_tokens)

    # Check budget BEFORE generating
    if cumulative_tokens >= max_seq_len:
        truncation_reason = "max_seq_len"
        break

    remaining = max_seq_len - cumulative_tokens

    # Generate
    responses = await policy.generate.route(
        [prompt_text],
        sampling_params={"max_tokens": remaining}
    )
    response = responses[0]

    # Accumulate prompt tokens (the delta)
    all_tokens.extend(new_prompt_tokens)
    response_mask.extend([0] * len(new_prompt_tokens))
    all_logprobs.extend([0.0] * len(new_prompt_tokens))

    # Accumulate response tokens
    all_tokens.extend(response.token_ids)
    response_mask.extend([1] * len(response.token_ids))
    all_logprobs.extend(response.logprobs)
    cumulative_tokens += len(response.token_ids)

    # Add assistant response to messages
    messages.append({"role": "assistant", "content": response.text})

    # Step environment
    result = env.step(action_text=response.text)

    if not result.done:
        # Add env observation to messages
        messages.append(result.observation.message)
        # (Tokens will be extracted at top of next loop via prefix matching)
```

**Pros:**
- ✅ Most correct - perfect alignment with chat template
- ✅ Handles all edge cases automatically
- ✅ Clear separation of concerns

**Cons:**
- Re-tokenizes full conversation each turn (O(n²) complexity)
- More expensive computationally

---

### Option C: Simplified NeMo-RL Pattern (Simplest)

**Pattern:** Like NeMo-RL, but acknowledge the limitations

**Implementation:**

```python
# Accept that we tokenize messages independently
# This means we DON'T get the exact chat template formatting

for turn in range(max_turns):
    # Build prompt text
    prompt_text = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )

    # Encode prompt to check budget (approximate)
    prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=False)

    # Generate
    responses = await policy.generate.route([prompt_text], ...)
    response = responses[0]

    # Accumulate prompt + response tokens
    all_tokens.extend(prompt_tokens)
    all_tokens.extend(response.token_ids)
    response_mask.extend([0] * len(prompt_tokens))
    response_mask.extend([1] * len(response.token_ids))

    # Step environment
    result = env.step(...)

    if not result.done:
        # Tokenize observation content only (like NeMo-RL)
        obs_text = result.observation.message["content"]
        obs_tokens = tokenizer.encode(obs_text, add_special_tokens=False)

        all_tokens.extend(obs_tokens)
        response_mask.extend([0] * len(obs_tokens))

        messages.append(result.observation.message)
```

**Pros:**
- ✅ Simplest implementation
- ✅ Works for simple cases

**Cons:**
- ❌ Tokens don't perfectly match chat template
- ❌ Budget tracking is approximate
- ❌ May break with complex chat templates or tool calling

---

## Recommendation: Option A (vLLM Token IDs + Prefix Matching)

**Why:**
1. **Efficient**: Uses vLLM's already-computed tokens when available
2. **Correct**: Falls back to prefix matching for environment observations
3. **Future-proof**: Works with tool calling, complex templates
4. **Clear**: Separates response tokens (from engine) vs observation tokens (prefix matching)

**Implementation sketch:**

```python
async def do_single_rollout(...) -> Episode:
    messages = messages.copy()
    all_tokens = []
    all_logprobs = []
    response_mask = []

    # Reset environment
    initial_obs = env.reset()
    messages.append({"role": "user", "content": initial_obs})

    for turn_num in range(max_turns):
        # Format prompt
        prompt = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )

        # Tokenize to check budget and get prompt tokens
        prompt_tokens_for_budget = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
        )

        # Extract NEW prompt tokens since last turn (prefix matching)
        new_prompt_tokens = prompt_tokens_for_budget[len(all_tokens):]

        # Check budget
        if len(all_tokens) + len(new_prompt_tokens) >= max_seq_len:
            truncation_reason = "max_seq_len"
            break

        remaining = max_seq_len - (len(all_tokens) + len(new_prompt_tokens))

        # Generate
        responses = await policy.generate.route(
            [prompt],
            sampling_params={"max_tokens": remaining}
        )
        response = responses[0]

        # Accumulate NEW prompt tokens
        all_tokens.extend(new_prompt_tokens)
        all_logprobs.extend([0.0] * len(new_prompt_tokens))
        response_mask.extend([0] * len(new_prompt_tokens))

        # Accumulate response tokens
        all_tokens.extend(response.token_ids)
        all_logprobs.extend(response.logprobs)
        response_mask.extend([1] * len(response.token_ids))

        # Add to messages
        messages.append({"role": "assistant", "content": response.text})

        # Step environment
        result = env.step(action_text=response.text)

        if not result.done:
            # Add observation to messages
            messages.append(result.observation.message)
            # Tokens will be extracted at next iteration via prefix matching
        else:
            break

    return Episode(
        all_token_ids=torch.tensor(all_tokens, dtype=torch.long),
        logprobs=torch.tensor(all_logprobs, dtype=torch.float),
        response_mask=torch.tensor(response_mask, dtype=torch.float),
        ...
    )
```

**Key points:**
1. Use `apply_chat_template(tokenize=True)` to get the FULL token sequence
2. Extract delta via `new_tokens = full_tokens[len(all_tokens):]` (prefix matching)
3. This captures ALL special tokens, role markers, etc.
4. Budget calculation is exact
5. Works for environment observations, tool responses, everything

---

## Summary Table

| Pattern | Libraries | Efficiency | Correctness | Complexity | Use When |
|---------|-----------|------------|-------------|------------|----------|
| **vLLM Token IDs** | TRL, VERL SGLang | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | Online rollout with vLLM |
| **Prefix Matching** | VERL Agent Loop, Verifiers | ⭐ | ⭐⭐⭐ | ⭐⭐⭐ | Offline processing, guaranteed correctness |
| **Independent Messages** | NeMo-RL | ⭐⭐ | ⭐ | ⭐ | Simple cases, no complex templates |
| **Hybrid (RECOMMENDED)** | - | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | Best of both worlds |

---

## Action Items for V5

1. ✅ **Change environment observation tokenization** from `tokenizer.encode(content)` to prefix matching
2. ✅ **Track cumulative tokens** correctly including all special tokens
3. ✅ **Update budget checks** to use the correct token count
4. ✅ **Add assertions** to verify prefix property holds (optional, for debugging)
5. ✅ **Test** that `all_token_ids` matches what model sees when we call `apply_chat_template`

---

## Testing the Fix

Add this validation to ensure correctness:

```python
# At the end of do_single_rollout()
# Verify that all_tokens matches full conversation tokenization
full_tokens_check = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=False,  # No gen prompt at end
    tokenize=True,
)

# They should match (or be very close, accounting for final generation prompt)
if len(all_tokens) != len(full_tokens_check):
    logger.warning(
        f"Token count mismatch: all_tokens={len(all_tokens)}, "
        f"full_recompute={len(full_tokens_check)}"
    )
```

---

**End of Document**
