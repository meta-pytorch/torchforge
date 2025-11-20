# RL Library Multi-Turn Conversation Masking Research

## Executive Summary

The NVIDIA NeMo-RL library (located at `/home/felipemello/forge/RL/`) provides a comprehensive approach to handling multi-turn conversation masking for RL training. The library **does NOT perform explicit suffix stripping after EOS tokens** - instead, it relies on the chat template to handle EOS tokens correctly and creates loss masks based on message roles.

## Key Findings

### 1. Loss Mask Creation (`token_loss_mask`)

The primary function for creating loss masks is `add_loss_mask_to_message_log()` located in:
- **File**: `/home/felipemello/forge/RL/nemo_rl/data/llm_message_utils.py`
- **Lines**: 141-176

**Code snippet:**
```python
def add_loss_mask_to_message_log(
    batch_message_log: list[LLMMessageLogType],
    roles_to_train_on: list[str] = ["assistant"],
    only_unmask_final: bool = False,
) -> None:
    """Add token-level loss masks to each message in a message log.

    Args:
        message_log (LLMMessageLogType): List of message dictionaries containing token IDs and metadata
        roles_to_train_on (list[str]): List of strings indicating which speakers to unmask. Default: ["assistant"]
        only_unmask_final (bool): If True, only unmask the final message in the log. Default: False
    """
    for i, role in enumerate(roles_to_train_on):
        roles_to_train_on[i] = role.lower()

    for message_log in batch_message_log:
        for i, message in enumerate(message_log):
            if only_unmask_final:
                if i == len(message_log) - 1:
                    message["token_loss_mask"] = torch.ones_like(
                        cast(Tensor, message["token_ids"])
                    )
                else:
                    message["token_loss_mask"] = torch.zeros_like(
                        cast(Tensor, message["token_ids"])
                    )
            else:
                if message["role"] in roles_to_train_on:
                    message["token_loss_mask"] = torch.ones_like(
                        cast(Tensor, message["token_ids"])
                    )
                else:
                    message["token_loss_mask"] = torch.zeros_like(
                        cast(Tensor, message["token_ids"])
                    )
```

**Key behavior:**
- Creates a `token_loss_mask` tensor that is `torch.ones_like(token_ids)` for assistant messages
- Creates a `token_loss_mask` tensor that is `torch.zeros_like(token_ids)` for non-assistant messages
- **ALL tokens in assistant messages are masked in (value=1), including any EOS tokens**
- No special handling for tokens after EOS

**Usage locations:**
- SFT: `/home/felipemello/forge/RL/nemo_rl/algorithms/sft.py:265`
- DPO: `/home/felipemello/forge/RL/nemo_rl/algorithms/dpo.py:176` (with `add_loss_mask=True`)
- GRPO: `/home/felipemello/forge/RL/nemo_rl/algorithms/grpo.py:1080-1086`
- Distillation: `/home/felipemello/forge/RL/nemo_rl/algorithms/distillation.py:659-663`

### 2. EOS Token Handling

The library handles EOS tokens at the **chat template level** during tokenization, not during masking.

**File**: `/home/felipemello/forge/RL/nemo_rl/data/llm_message_utils.py`
**Function**: `get_formatted_message_log()`
**Lines**: 443-659

**Key EOS handling code (lines 588-606):**
```python
if i == len(message_log_strs) - 1:
    r"""
    This is an attempt to robustly append the eos token. The origin is Qwen
    chat templates always append <eos>\n and some models like gemma do not
    use the <eos> at all in the chat template. Adding a <eos> if the <eos> is
    already at the end, is likely a user error, and since we know Qwen likes to
    have <eos>\n we'll check for that case.

    This makes the logic slightly more robust to the model family's chat template
    so users don't need to know whether they need to add add_eos or not.
    """
    stripped_message_chunk = message_chunk.rstrip("\n")
    if add_eos_token:
        if tokenizer.eos_token is None:
            warnings.warn(
                "add_eos_token is True but the tokenizer does not have an EOS token. Skipping EOS token addition."
            )
        elif not stripped_message_chunk.endswith(tokenizer.eos_token):
            message_chunk += tokenizer.eos_token
```

**Behavior:**
- EOS token is added to the **last message** in the conversation
- The code strips trailing newlines before checking if EOS is already present
- If the stripped message doesn't end with EOS, it appends `tokenizer.eos_token`
- This ensures EOS is present exactly once at the end

### 3. Multi-Turn Generation: Handling Tokens After EOS

**File**: `/home/felipemello/forge/RL/nemo_rl/models/generation/vllm/vllm_worker_async.py`
**Function**: `_replace_prefix_tokens()`
**Lines**: 40-121

This is the most sophisticated EOS handling in the codebase. It deals with multi-turn generation where previous turns may have EOS tokens.

**Code snippet (lines 97-121):**
```python
eos_token_id = tokenizer.eos_token_id
assert eos_token_id is not None, "Your tokenizer must have an EOS token ID!"

model_cut_end = len(model_prefix_token_ids)
if model_prefix_token_ids:
    # We are not always guaranteed that the model outputs an EOS token as the stop criteria of the previous model call e.g. when the model reaches max_tokens.
    # And since chat templates will always add one for us, we just cut the model input to right before the EOS token ID (if applicable)
    if model_prefix_token_ids[-1] == eos_token_id:
        model_cut_end -= 1

# We take everything starting with the EOS token ID.
template_cut_start = -1
for pos in reversed(range(len(template_prefix_token_ids))):
    if template_token_ids[pos] == eos_token_id:
        template_cut_start = pos
        break

# This should never be the case, but
assert template_cut_start >= 0, (
    "No EOS token ID found in the chat-templated messages!"
)

return (
    model_prefix_token_ids[:model_cut_end] + template_token_ids[template_cut_start:]
)
```

**Key behavior:**
- When continuing multi-turn generation, it finds the last EOS in the template
- If the model's previous output ended with EOS, it **cuts before that EOS** (`model_cut_end -= 1`)
- Then it appends everything from the template starting at the EOS position
- This ensures proper token alignment when the chat template re-tokenizes text differently

**Test validation** (lines 1283-1301 in `/home/felipemello/forge/RL/tests/unit/models/generation/test_vllm_generation.py`):
```python
model_prefix_token_ids = og_model_token_ids[:-16]
assert model_prefix_token_ids[-1] == eos_token_id
# newline after EOS
template_prefix_token_ids = template_token_ids[:-15]
assert template_prefix_token_ids[-2] == eos_token_id
assert template_prefix_token_ids[-1] != eos_token_id
result = _replace_prefix_tokens(
    tokenizer=tokenizer,
    model_prefix_token_ids=model_prefix_token_ids,
    template_prefix_token_ids=template_prefix_token_ids,
    template_token_ids=template_token_ids,
)
assert result == og_model_token_ids
```

This test shows they handle the case where template has **newline after EOS**.

### 4. No Suffix Stripping After EOS

**Finding**: The library **does NOT strip or validate suffix length after EOS tokens**.

Evidence:
1. No grep results for patterns like "strip.*suffix", "suffix.*strip", "after.*eos" in data processing code
2. Loss masks are created based solely on role, not on EOS position
3. The `token_loss_mask` is created with `torch.ones_like(token_ids)` for entire assistant messages

**Implication**: If a chat template generates tokens after EOS (e.g., `<eos>\n`), those tokens would be:
- **Included in the token_ids**
- **Included in the loss mask (masked in with value=1)**
- **Used for training loss computation**

The library relies on:
1. Chat templates being well-formed (not generating extra tokens after EOS)
2. EOS token handling at generation time (via `_replace_prefix_tokens`)
3. Proper tokenizer configuration

### 5. Chat Template Usage

**File**: `/home/felipemello/forge/RL/nemo_rl/data/llm_message_utils.py`
**Lines**: 541-543

```python
formatted_message: str = tokenizer.apply_chat_template(  # type: ignore
    message_log_strs[: i + 1], **template_kwargs
)
```

The library uses `tokenizer.apply_chat_template()` extensively:
- Each message turn is formatted incrementally
- Difference between consecutive formatted strings gives the current message chunk
- This approach handles model-specific formatting (Llama, Qwen, Gemma, etc.)

**Configurable chat templates** (`/home/felipemello/forge/RL/nemo_rl/models/policy/__init__.py:137`):
```python
# Arguments to pass to tokenizer.apply_chat_template(...). This can be used to pass kwargs like enable_thinking=true
```

Users can pass custom kwargs to `apply_chat_template` (e.g., `enable_thinking=True` for Qwen3).

### 6. Test Validation of EOS Handling

**File**: `/home/felipemello/forge/RL/tests/unit/data/test_llm_message_utils.py`
**Function**: Test parameterization
**Lines**: 420-498

Test expectations documented (lines 420-434):
```python
"""
Expectations:
- Require an EOS token for well-defined end-of-turn comparison.
- When add_generation_prompt is False, the concatenated contents must match
  the tokenizer's apply_chat_template output; if the tokenizer omits a final
  EOS, accept the actual with EOS by appending EOS to the expected before
  comparison.
- When add_generation_prompt is True and the last turn is an assistant
  message, accept either:
    (1) prefix built with add_generation_prompt=True followed by the raw
        assistant content plus EOS; or
    (2) the tokenizer's full non-generation template output plus EOS.
  This avoids hard-coding model-specific headers or delimiters while still
  verifying semantic equivalence.
- Only normalization performed is trimming a trailing newline after EOS.
"""
```

**Normalization function (lines 449-453):**
```python
def normalize(s: str) -> str:
    # Normalize EOS+newline quirk to EOS only
    if s.endswith(eos + "\n"):
        return s[:-1]
    return s
```

**Key insight**: The test normalizes `<eos>\n` → `<eos>` for comparison, acknowledging that some templates (like Qwen) add newlines after EOS. This is **purely for test validation**, not for actual training data processing.

### 7. Collate Function Integration

**File**: `/home/felipemello/forge/RL/nemo_rl/data/collate_fn.py`
**Function**: `preference_collate_fn()`
**Lines**: 127-197

```python
def preference_collate_fn(
    data_batch: list[DPODatumSpec],
    tokenizer: TokenizerType,
    make_sequence_length_divisible_by: int,
    add_loss_mask: bool,
) -> BatchedDataDict[Any]:
    # ... batching logic ...

    if add_loss_mask:
        add_loss_mask_to_message_log(
            batch["message_log"],
            only_unmask_final=True,  # For DPO, only train on final response
        )

    cat_and_padded, input_lengths = batched_message_log_to_flat_message(
        batch["message_log"],
        pad_value_dict={"token_ids": tokenizer.pad_token_id},
        make_sequence_length_divisible_by=make_sequence_length_divisible_by,
    )

    data: BatchedDataDict[Any] = BatchedDataDict(
        {
            "input_ids": cat_and_padded["token_ids"],
            "input_lengths": input_lengths,
            "sample_mask": batch["loss_multiplier"],
        }
    )
    if add_loss_mask:
        data["token_mask"] = cat_and_padded["token_loss_mask"]

    return data
```

The `token_mask` from `token_loss_mask` is used directly for loss computation.

## Summary: Design Philosophy

The NeMo-RL library's approach:

1. **Trust the chat template**: Assumes `tokenizer.apply_chat_template()` produces well-formed sequences
2. **Role-based masking**: Masks are created based on message role, not token content
3. **EOS at generation time**: Handles EOS tokens during generation (multi-turn) with `_replace_prefix_tokens()`
4. **No post-EOS stripping**: Does not validate or strip tokens after EOS
5. **Test normalization only**: Tests normalize `<eos>\n` but training data keeps it as-is

## Comparison to Other Approaches

**What NeMo-RL does NOT do:**
- ❌ Check if tokens exist after EOS
- ❌ Strip suffix after EOS
- ❌ Validate suffix length is 0 after EOS
- ❌ Create masks based on EOS position

**What NeMo-RL DOES do:**
- ✅ Add EOS token if missing from chat template
- ✅ Handle EOS during multi-turn generation continuations
- ✅ Create loss masks based on role (assistant vs user)
- ✅ Normalize `<eos>\n` → `<eos>` in tests only

## Relevant File Paths

1. **Core masking logic**: `/home/felipemello/forge/RL/nemo_rl/data/llm_message_utils.py`
   - `add_loss_mask_to_message_log()` (lines 141-176)
   - `get_formatted_message_log()` (lines 443-659)

2. **EOS handling for generation**: `/home/felipemello/forge/RL/nemo_rl/models/generation/vllm/vllm_worker_async.py`
   - `_replace_prefix_tokens()` (lines 40-121)

3. **Collate functions**: `/home/felipemello/forge/RL/nemo_rl/data/collate_fn.py`
   - `preference_collate_fn()` (lines 127-197)

4. **Algorithm usage**:
   - SFT: `/home/felipemello/forge/RL/nemo_rl/algorithms/sft.py:265`
   - DPO: `/home/felipemello/forge/RL/nemo_rl/algorithms/dpo.py:176`
   - GRPO: `/home/felipemello/forge/RL/nemo_rl/algorithms/grpo.py:1080-1086`
   - Distillation: `/home/felipemello/forge/RL/nemo_rl/algorithms/distillation.py:659-663`

5. **Tests**: `/home/felipemello/forge/RL/tests/unit/data/test_llm_message_utils.py`
   - EOS normalization tests (lines 420-498)
   - Loss mask tests (lines 567-614)

6. **Generation tests**: `/home/felipemello/forge/RL/tests/unit/models/generation/test_vllm_generation.py`
   - `test_VllmAsyncGenerationWorker_replace_prefix_tokens()` (lines 1235-1329)

## Recommendation

If you need to handle tokens after EOS in your implementation:

1. **For training data**: You may want to add validation/stripping logic before `add_loss_mask_to_message_log()` is called
2. **For generation**: Use NeMo-RL's `_replace_prefix_tokens()` approach for multi-turn handling
3. **For chat templates**: Ensure your templates don't generate tokens after EOS, or strip them explicitly

The NeMo-RL approach assumes clean chat templates. If your chat template generates `<eos>\n`, you would need to:
- Either modify the chat template to not generate the newline
- Or add a post-processing step to strip tokens after EOS before creating masks
