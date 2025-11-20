# Prime-RL Multi-Turn Conversation Masking Research

## Executive Summary

Prime-RL uses a different approach to multi-turn conversation masking than Forge. Key differences:

1. **NO suffix stripping after EOS** - Prime-RL does NOT check or strip tokens after EOS in responses
2. **Incremental tokenization** - Uses incremental chat template application to build masks
3. **Delegation to verifiers library** - RL masking logic is in the external `verifiers` library, not prime-rl itself
4. **SFT ensures EOS presence** - SFT training always ensures EOS token is present in target_ids

---

## 1. SFT Loss Mask Creation (Multi-Turn)

### Location
**File**: `/home/felipemello/forge/prime-rl/src/prime_rl/trainer/sft/data.py`
**Function**: `build_loss_mask()` (lines 226-255)

### How It Works

Prime-RL uses **incremental tokenization** with `apply_chat_template()` to build loss masks:

```python
def build_loss_mask(prompt, completion, tokenizer, loss_mask_config: LossMaskConfig) -> list[bool]:
    messages = prompt + completion
    loss_mask: list[bool] = []
    prev_ids, prev_len = [], 0
    for i, message in enumerate(messages):
        # Tokenize conversation up to current message (incremental)
        cur_ids = tokenizer.apply_chat_template(
            messages[: i + 1],
            tools=tools,
            add_generation_prompt=True if (
                message["role"] in ["user", "tool"]
                and i + 1 < len(messages)
                and messages[i + 1]["role"] == "assistant"
            ) else False,
        )
        # Verify incremental consistency
        assert prev_ids == cur_ids[:prev_len]

        # Extend mask for new tokens with role-based masking
        loss_mask.extend([should_mask(message, loss_mask_config)] * (len(cur_ids) - prev_len))
        prev_ids, prev_len = cur_ids, len(cur_ids)

    return loss_mask
```

**Key Points:**
- Incremental tokenization: tokenize `messages[:i+1]` at each step
- Verifies prefix consistency: `prev_ids == cur_ids[:prev_len]`
- Uses `add_generation_prompt=True` after user/tool messages to mask assistant header tokens
- Role-based masking controlled by `LossMaskConfig` (system, user, assistant, tool)

### Loss Mask Configuration

**File**: `/home/felipemello/forge/prime-rl/src/prime_rl/trainer/sft/config.py` (lines 36-42)

```python
class LossMaskConfig(BaseModel):
    system: bool = False      # Don't train on system messages
    user: bool = False        # Don't train on user messages
    assistant: bool = True    # DO train on assistant messages
    tool: bool = False        # Don't train on tool messages
```

**Default behavior**: Only train on assistant messages, mask everything else.

---

## 2. EOS Token Handling in SFT

### Location
**File**: `/home/felipemello/forge/prime-rl/src/prime_rl/trainer/sft/data.py`
**Function**: `_process()` (lines 270-293)

### EOS Handling Logic

```python
# Build input_ids using chat template
input_ids = self.tokenizer.apply_chat_template(
    prompt + completion,
    tools=tools,
)

# Build loss_mask
loss_mask = build_loss_mask(prompt, completion, self.tokenizer, self.loss_mask_config)

# If EOS token is not found, manually append it
if not self.tokenizer.eos_token_id in input_ids:
    self.logger.warning(
        f"Did not find EOS token ID {self.tokenizer.eos_token_id} in input_ids. "
        "Is something wrong with the chat template? Manually appending EOS token..."
    )
    input_ids.append(cast(int, self.tokenizer.eos_token_id))
    loss_mask.append(True)

# Prepare inputs (shift for next-token prediction)
target_ids = input_ids.copy()[1:]
loss_mask = loss_mask[1:]
input_ids = input_ids[:-1]

# Assertions
assert sum(loss_mask) > 0, "There are no tokens in this sample that contribute to the loss"
assert self.tokenizer.eos_token_id in target_ids, "EOS token ID must be present in target_ids"
```

**Critical Findings:**
1. ✅ **EOS is REQUIRED** in target_ids (assertion on line 293)
2. ✅ **Manually appends EOS** if chat template doesn't include it
3. ❌ **NO suffix stripping** - Does NOT check for or remove tokens after EOS
4. ✅ **Trains on EOS** - The EOS token has `loss_mask=True`

---

## 3. RL Loss Mask Creation (Multi-Turn)

### Architecture

RL mask creation is **delegated to the verifiers library**:

```
prime-rl/orchestrator/scheduler.py
  └─> env.process_env_results_vllm()
      └─> verifiers/envs/environment.py::process_env_results_vllm()
          └─> verifiers/utils/processing_utils.py::process_chat_format_vllm()
```

### Main Entry Point

**File**: `/home/felipemello/forge/prime-rl/src/prime_rl/orchestrator/scheduler.py` (lines 71-85)

```python
def process_generate_outputs(self, generate_outputs: GenerateOutputs) -> list[Rollout]:
    processed_outputs: ProcessedOutputs = self.env.process_env_results_vllm(
        prompts=generate_outputs.prompt,
        completions=generate_outputs.completion,
        states=generate_outputs.state,
        rewards=generate_outputs.reward,
        processing_class=self.tokenizer,
        max_seq_len=self.seq_len,
        mask_env_responses=self.config.mask_env_responses,
        zero_truncated_completions=self.config.zero_truncated_completions,
        mask_truncated_completions=self.config.mask_truncated_completions,
    )
    # Returns: prompt_ids, prompt_mask, completion_ids, completion_mask, completion_logprobs
```

### Verifiers Library Processing

**File**: `/home/felipemello/forge/verifiers/verifiers/utils/processing_utils.py`
**Function**: `process_chat_format_vllm()` (lines 72-162)

#### Chat Format Processing

```python
def process_chat_format_vllm(
    prompt: list[ChatMessage],
    completion: list[ChatMessage],
    state: State,
    processing_class: "PreTrainedTokenizerBase",
    mask_env_responses: bool = False,
) -> tuple[list[int], list[int], list[int], list[int], list[float]]:
    """
    Process chat format conversations using incremental prefixes.
    """
    responses = state["responses"]  # vLLM response objects

    # Match completion messages with vLLM responses
    zipped = []
    for turn in completion:
        if turn["role"] == "assistant":
            zipped.append((turn, responses[responses_idx]))
            responses_idx += 1
        else:
            zipped.append((turn, None))

    # Tokenize prompt
    prompt_ids = processing_class.apply_chat_template(
        conversation=prompt,
        add_generation_prompt=True,
        tools=oai_tools,
    )
    prompt_mask = [0] * len(prompt_ids)  # Don't train on prompt

    # Process completion turns incrementally
    completion_ids = []
    completion_mask = []
    completion_logprobs = []

    i = 0
    while i < len(zipped):
        message, response = zipped[i]

        if message["role"] == "assistant":
            # Use vLLM response tokens and logprobs
            completion_turn_ids = parse_chat_completion_tokens(response)
            completion_turn_mask = [1] * len(completion_turn_ids)
            completion_turn_logprobs = parse_chat_completion_logprobs(response)

            completion_ids.extend(completion_turn_ids)
            completion_mask.extend(completion_turn_mask)
            completion_logprobs.extend(completion_turn_logprobs)
            messages_consumed.append(message)
            i += 1

        else:  # user/tool case
            # Collect consecutive non-assistant messages
            consecutive_messages = [message]
            j = i + 1
            while j < len(zipped) and zipped[j][0]["role"] != "assistant":
                consecutive_messages.append(zipped[j][0])
                j += 1

            # Tokenize prefix (up to last assistant)
            token_prefix = processing_class.apply_chat_template(
                conversation=messages_consumed,
                add_generation_prompt=False,
                tools=oai_tools,
            )

            # Tokenize with new user/tool + assistant header
            token_prefix_with_turn = processing_class.apply_chat_template(
                conversation=messages_consumed + consecutive_messages,
                add_generation_prompt=True,  # Includes assistant header
                tools=oai_tools,
            )

            # Extract new tokens (user message + assistant header)
            completion_turn_ids = token_prefix_with_turn[len(token_prefix):]

            if mask_env_responses:
                completion_turn_mask = [0] * len(completion_turn_ids)  # Mask env responses
            else:
                completion_turn_mask = [1] * len(completion_turn_ids)  # Train on env responses

            completion_turn_logprobs = [0.0] * len(completion_turn_ids)  # No logprobs for env

            completion_ids.extend(completion_turn_ids)
            completion_mask.extend(completion_turn_mask)
            completion_logprobs.extend(completion_turn_logprobs)
            messages_consumed.extend(consecutive_messages)
            i = j

    return (prompt_ids, prompt_mask, completion_ids, completion_mask, completion_logprobs)
```

**Key Points:**
1. Uses **vLLM response objects** stored in `state["responses"]` to get actual generated tokens/logprobs
2. **Incremental tokenization** similar to SFT (verifies prefix consistency)
3. **mask_env_responses flag**: controls whether environment responses (user/tool) are trained on
4. Assistant messages use **actual vLLM tokens**, env responses use **tokenizer**

### Tokens from vLLM Responses

**File**: `/home/felipemello/forge/verifiers/verifiers/utils/processing_utils.py` (lines 38-52)

```python
def parse_chat_completion_tokens(chat_completion: ChatCompletion) -> list[int]:
    """Parses the output token ids from vLLM chat completion."""
    tokens = [
        # tokens are token_id:<int> because we request `return_tokens_as_token_ids` from vllm
        int(token.token.split(":")[-1])
        for token in chat_completion.choices[0].logprobs.content
    ]
    return tokens
```

**Critical**: Uses **vLLM's exact generated tokens**, which are in `choices[0].logprobs.content`.

---

## 4. How Tokens After EOS Are Handled

### The KEY Finding

**Prime-RL does NOT check or strip tokens after EOS in responses.**

Let me trace through what happens:

#### In RL (verifiers library):

1. **vLLM generates response** with tokens (may include EOS)
2. **parse_chat_completion_tokens()** extracts ALL tokens from `logprobs.content`
   - This includes the EOS token if generated
   - **NO filtering or stripping** of tokens after EOS
3. **completion_mask** is set to `[1] * len(completion_turn_ids)` for assistant messages
   - ALL assistant tokens (including and after EOS) have mask=1
4. These tokens are added to `completion_ids` and `completion_mask`

#### In SFT:

1. **apply_chat_template()** returns full token sequence
2. **Manually appends EOS** if not present
3. **NO suffix stripping** - No code checks for or removes tokens after EOS
4. **loss_mask[EOS] = True** - EOS token is trained on
5. Assertion ensures EOS is in target_ids, but doesn't check uniqueness or position

### What This Means

**If vLLM generates tokens after EOS** (e.g., padding, extra tokens):
- ✅ Those tokens ARE included in `completion_ids`
- ✅ Those tokens ARE included in `completion_mask` with value `1`
- ✅ Those tokens WILL contribute to the loss
- ❌ There is NO check or warning about suffix length
- ❌ There is NO stripping of post-EOS tokens

**This is fundamentally different from Forge's approach**, which:
- Checks for tokens after EOS
- Strips suffix tokens after EOS
- Validates suffix length

---

## 5. Multi-Turn Conversation Example

Let's trace a 2-turn conversation:

### Messages
```python
prompt = [
    {"role": "user", "content": "Hello"}
]
completion = [
    {"role": "assistant", "content": "Hi there!"},
    {"role": "user", "content": "How are you?"},
    {"role": "assistant", "content": "I'm good!"}
]
```

### SFT Processing

**Step 1**: Tokenize `[user: "Hello"]`
- Tokens: `[<|im_start|>user\nHello<|im_end|><|im_start|>assistant\n]`
- Mask: `[False, False, False, ..., False]` (all user + assistant header)

**Step 2**: Tokenize `[user: "Hello", assistant: "Hi there!"]`
- New tokens: `[Hi, there, !, <|im_end|>]`
- Mask extends: `[True, True, True, True]` (assistant message)

**Step 3**: Tokenize `[..., user: "How are you?"]`
- New tokens: `[<|im_start|>user\nHow, are, you, ?, <|im_end|><|im_start|>assistant\n]`
- Mask extends: `[False, False, ..., False]` (user + assistant header)

**Step 4**: Tokenize `[..., assistant: "I'm good!"]`
- New tokens: `[I, 'm, good, !, <|im_end|>]`
- Mask extends: `[True, True, True, True, True]` (assistant message)

**Final**:
- `input_ids`: All tokens except last
- `target_ids`: All tokens except first
- `loss_mask`: Only True for assistant content (not headers, not user)

### RL Processing (verifiers)

**Prompt tokenization**:
```python
prompt_ids = tokenizer.apply_chat_template(
    [{"role": "user", "content": "Hello"}],
    add_generation_prompt=True  # Adds assistant header
)
prompt_mask = [0] * len(prompt_ids)
```

**Turn 1** (assistant):
```python
# Use vLLM response object
response = state["responses"][0]
completion_ids = parse_chat_completion_tokens(response)  # [Hi, there, !, <|im_end|>]
completion_mask = [1, 1, 1, 1]
completion_logprobs = parse_chat_completion_logprobs(response)
```

**Turn 2** (user):
```python
# Incremental tokenization
prefix = tokenizer.apply_chat_template(
    [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi there!"}],
    add_generation_prompt=False
)
prefix_with_turn = tokenizer.apply_chat_template(
    [..., {"role": "user", "content": "How are you?"}],
    add_generation_prompt=True  # Adds next assistant header
)
new_tokens = prefix_with_turn[len(prefix):]  # User message + assistant header
completion_ids.extend(new_tokens)
completion_mask.extend([1] * len(new_tokens))  # or [0] if mask_env_responses=True
completion_logprobs.extend([0.0] * len(new_tokens))
```

**Turn 3** (assistant):
```python
response = state["responses"][1]
completion_ids.extend(parse_chat_completion_tokens(response))  # [I, 'm, good, !, <|im_end|>]
completion_mask.extend([1, 1, 1, 1, 1])
completion_logprobs.extend(parse_chat_completion_logprobs(response))
```

---

## 6. Comparison with Forge

| Aspect | Prime-RL | Forge |
|--------|----------|-------|
| **Mask Creation** | Incremental tokenization with chat template | Base anchor + response mask |
| **EOS Handling** | Ensures EOS present, NO suffix stripping | Checks and strips tokens after EOS |
| **Suffix Validation** | None | Validates suffix_len <= max_suffix_len |
| **Multi-turn** | Native support via incremental tokenization | Handles via base anchors |
| **RL vs SFT** | Different codepaths (verifiers vs trainer) | Same masking logic |
| **vLLM Integration** | Uses vLLM response tokens directly | Tokenizes text responses |
| **Env Response Masking** | Configurable via `mask_env_responses` | Not directly supported |
| **Library Separation** | Mask logic in external `verifiers` lib | All in forge.data.common |

---

## 7. Configuration Options

### SFT Configuration

```python
# In SFTDataConfig
loss_mask: LossMaskConfig = LossMaskConfig(
    system=False,     # Don't train on system messages
    user=False,       # Don't train on user messages
    assistant=True,   # Train on assistant messages
    tool=False        # Don't train on tool messages
)
```

### RL Configuration

```python
# In OrchestratorConfig (via process_env_results_vllm)
mask_env_responses: bool = False              # Whether to mask env responses (user/tool)
zero_truncated_completions: bool = False      # Zero reward for truncated completions
mask_truncated_completions: bool = False      # Mask loss for truncated completions
```

---

## 8. Key Files Reference

### Prime-RL

| File | Lines | Purpose |
|------|-------|---------|
| `/home/felipemello/forge/prime-rl/src/prime_rl/trainer/sft/data.py` | 226-255 | SFT loss mask creation (build_loss_mask) |
| `/home/felipemello/forge/prime-rl/src/prime_rl/trainer/sft/data.py` | 270-293 | EOS token handling in SFT |
| `/home/felipemello/forge/prime-rl/src/prime_rl/trainer/sft/config.py` | 36-42 | LossMaskConfig definition |
| `/home/felipemello/forge/prime-rl/src/prime_rl/orchestrator/scheduler.py` | 71-85 | RL entry point for processing |
| `/home/felipemello/forge/prime-rl/src/prime_rl/orchestrator/batch.py` | 21-64 | Rollout to training batch conversion |
| `/home/felipemello/forge/prime-rl/src/prime_rl/trainer/rl/data.py` | 13-23 | RL MicroBatch type definition |

### Verifiers Library

| File | Lines | Purpose |
|------|-------|---------|
| `/home/felipemello/forge/verifiers/verifiers/envs/environment.py` | 913-1007 | process_env_results_vllm main logic |
| `/home/felipemello/forge/verifiers/verifiers/utils/processing_utils.py` | 72-162 | process_chat_format_vllm (mask creation) |
| `/home/felipemello/forge/verifiers/verifiers/utils/processing_utils.py` | 38-69 | Token/logprob parsing from vLLM |
| `/home/felipemello/forge/verifiers/verifiers/types.py` | 135-147 | Rollout TypedDict definition |

---

## 9. Critical Code Snippets

### Incremental Tokenization Pattern (SFT)

```python
# From prime-rl/src/prime_rl/trainer/sft/data.py:226-253
messages = prompt + completion
loss_mask: list[bool] = []
prev_ids, prev_len = [], 0

for i, message in enumerate(messages):
    # Incrementally tokenize up to current message
    cur_ids = tokenizer.apply_chat_template(
        messages[: i + 1],
        tools=tools,
        add_generation_prompt=True if (
            message["role"] in ["user", "tool"]
            and i + 1 < len(messages)
            and messages[i + 1]["role"] == "assistant"
        ) else False,
    )

    # Verify incremental consistency
    assert prev_ids == cur_ids[:prev_len], "Incremental tokenization mismatch"

    # Extend mask based on message role
    loss_mask.extend([should_mask(message, loss_mask_config)] * (len(cur_ids) - prev_len))
    prev_ids, prev_len = cur_ids, len(cur_ids)

return loss_mask
```

### vLLM Token Extraction (RL)

```python
# From verifiers/verifiers/utils/processing_utils.py:38-52
def parse_chat_completion_tokens(chat_completion: ChatCompletion) -> list[int]:
    """Parses the output token ids from vLLM chat completion."""
    tokens = [
        int(token.token.split(":")[-1])  # Parse "token_id:123" -> 123
        for token in chat_completion.choices[0].logprobs.content
    ]
    return tokens
```

### Env Response Masking (RL)

```python
# From verifiers/verifiers/utils/processing_utils.py:120-155
else:  # user/tool case
    # Collect consecutive non-assistant messages
    consecutive_messages = [message]
    j = i + 1
    while j < len(zipped) and zipped[j][0]["role"] != "assistant":
        consecutive_messages.append(zipped[j][0])
        j += 1

    # Get tokens for user/tool messages + assistant header
    token_prefix = processing_class.apply_chat_template(
        conversation=messages_consumed,
        add_generation_prompt=False,
    )
    token_prefix_with_turn = processing_class.apply_chat_template(
        conversation=messages_consumed + consecutive_messages,
        add_generation_prompt=True,  # Include assistant header for next turn
    )

    completion_turn_ids = token_prefix_with_turn[len(token_prefix):]

    # Apply masking based on config
    if mask_env_responses:
        completion_turn_mask = [0] * len(completion_turn_ids)
    else:
        completion_turn_mask = [1] * len(completion_turn_ids)

    completion_turn_logprobs = [0.0] * len(completion_turn_ids)
```

---

## 10. Recommendations for Forge

Based on this research, here are key differences to consider:

### 1. EOS Token Handling
**Prime-RL**: Does NOT strip tokens after EOS
**Recommendation**: Forge's approach (stripping post-EOS tokens) is safer and more correct

### 2. Incremental Tokenization
**Prime-RL**: Uses incremental chat template application with verification
**Recommendation**: Consider adopting this pattern for better multi-turn support

### 3. Environment Response Masking
**Prime-RL**: Has explicit `mask_env_responses` flag
**Recommendation**: Useful feature to prevent training on environment feedback

### 4. Separation of Concerns
**Prime-RL**: RL masking in separate `verifiers` library
**Recommendation**: Forge's unified approach in `forge.data.common` is simpler

### 5. vLLM Integration
**Prime-RL**: Uses actual vLLM token IDs from responses
**Recommendation**: More accurate than re-tokenizing text, but requires vLLM

### 6. Truncation Handling
**Prime-RL**: Has flags for `zero_truncated_completions` and `mask_truncated_completions`
**Recommendation**: Good pattern for handling incomplete generations

---

## 11. Testing Evidence

From `/home/felipemello/forge/prime-rl/tests/unit/train/sft/test_sft_dataset.py`:

```python
def test_multiturn_loss_mask():
    dataset = Dataset.from_list([
        {
            "prompt": [
                {"role": "system", "content": "System 0"},
                {"role": "user", "content": "Prompt 0"}
            ],
            "completion": [
                {"role": "assistant", "content": "Completion 0"},
                {"role": "user", "content": "Prompt 1"},
                {"role": "assistant", "content": "Completion 1"},
            ],
        },
    ])
    tokenizer = AutoTokenizer.from_pretrained("PrimeIntellect/Qwen3-0.6B")
    dataset = SFTDataset(dataset, tokenizer=tokenizer, max_examples=1)
    sample = next(iter(dataset))
    print_sample(sample["input_ids"], sample["loss_mask"], tokenizer)
```

This test validates the multi-turn masking but does NOT test suffix handling.

---

## Conclusion

Prime-RL's approach to multi-turn masking is solid but **does NOT handle tokens after EOS**. This is a significant difference from Forge's approach and could lead to training on garbage tokens if vLLM generates extra tokens after EOS.

The incremental tokenization pattern is elegant and robust for multi-turn conversations, but the lack of suffix validation is a potential issue.
