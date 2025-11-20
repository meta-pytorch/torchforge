# TRL Multi-Turn Conversation Masking Research

## Executive Summary

TRL (Transformers Reinforcement Learning) library handles multi-turn conversation masking in the following key ways:

1. **EOS Token Masking**: Automatically masks tokens AFTER the first EOS token in completions
2. **Assistant-Only Masking**: Uses `assistant_masks` from tokenizer's chat template for multi-turn conversations
3. **Completion Masking**: Uses `completion_mask` to distinguish prompt from completion in prompt-completion datasets
4. **No Suffix Length Checking**: Does NOT explicitly check or strip tokens after EOS beyond basic masking
5. **Chat Template Integration**: Relies on tokenizer's `apply_chat_template` with `return_assistant_tokens_mask=True`

---

## 1. Completion Mask Creation for Multi-Turn Conversations

### GRPO Trainer (grpo_trainer.py)

**File**: `/home/felipemello/forge/trl/trl/trainer/grpo_trainer.py`

#### Initial Completion Mask Creation (Lines 1470-1473)
```python
# After generation, create initial mask based on actual completion lengths
completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids_list]
completion_mask = [torch.ones_like(ids, dtype=torch.long) for ids in completion_ids]
completion_ids = pad(completion_ids, padding_value=self.pad_token_id, padding_side="right")
completion_mask = pad(completion_mask, padding_value=0, padding_side="right")
```

**Key Points**:
- Creates a mask with 1s for all actual completion tokens
- Pads with 0s for padding tokens
- Does NOT differentiate between assistant/user tokens at this stage

#### Truncated Completion Masking (Lines 1480-1484)
```python
# If mask_truncated_completions is enabled, zero out truncated completions in completion_mask
if self.mask_truncated_completions:
    eos_and_pad = [self.eos_token_id, self.pad_token_id]
    is_truncated = torch.tensor([ids[-1] not in eos_and_pad for ids in completion_ids_list], device=device)
    completion_mask = completion_mask * (~is_truncated).unsqueeze(1).int()
```

**Key Points**:
- Optional masking of entire truncated completions
- Checks if last token is EOS or PAD
- If not, masks the ENTIRE completion (sets all to 0)
- This is sequence-level masking, not token-level

---

## 2. How TRL Handles Tokens AFTER EOS in Completions

### EOS Token Masking During Generation (Lines 1383-1390)

**File**: `/home/felipemello/forge/trl/trl/trainer/grpo_trainer.py`

```python
# Mask everything after the first EOS token
is_eos = completion_ids == self.eos_token_id
eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()
prompt_ids = [p[m].tolist() for p, m in zip(prompt_ids, prompt_mask.bool(), strict=True)]
completion_ids = [c[m].tolist() for c, m in zip(completion_ids, completion_mask.bool(), strict=True)]
```

**Key Points**:
- Finds FIRST EOS token using `argmax`
- Creates mask that includes tokens up to and including the first EOS
- Tokens AFTER first EOS are excluded from completion_ids entirely
- This happens during generation with transformers (non-vLLM path)

**Behavior**: Tokens after the first EOS are **stripped out** of the completion_ids list, not just masked.

### RLOO Trainer - Same Pattern

**File**: `/home/felipemello/forge/trl/trl/trainer/rloo_trainer.py` (Lines 1176-1183)

```python
# Mask everything after the first EOS token
is_eos = completion_ids == self.eos_token_id
eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()
prompt_ids = [p[m].tolist() for p, m in zip(prompt_ids, prompt_mask.bool(), strict=True)]
completion_ids = [c[m].tolist() for c, m in zip(completion_ids, completion_mask.bool(), strict=True)]
```

**Identical behavior to GRPO.**

---

## 3. Suffix Length Checking After EOS

### Answer: NO explicit suffix length checking

TRL does NOT check or validate suffix length after EOS. Instead:

1. **During generation (transformers path)**: Tokens after first EOS are stripped (see above)
2. **For vLLM/rollout_func paths**: vLLM handles this internally
3. **For truncation detection**: Only checks if last token is EOS/PAD (Lines 1421-1424)

```python
# Identify sequences that terminated with EOS and log their lengths
eos_and_pad = [self.eos_token_id, self.pad_token_id]
is_truncated = torch.tensor([ids[-1] not in eos_and_pad for ids in completion_ids], device=device)
agg_is_truncated = self.accelerator.gather(is_truncated)
```

**Key Points**:
- No validation of "how many tokens after EOS"
- No error/warning if there are extra tokens after EOS
- Relies on masking to exclude them from loss computation

---

## 4. Chat Template Handling for Multi-Turn Conversations

### SFT Trainer - Assistant Masks

**File**: `/home/felipemello/forge/trl/trl/trainer/sft_trainer.py`

#### Tokenization with Assistant Masking (Lines 969-985)

```python
prompt_completion_processed = processing_class.apply_chat_template(
    prompt + completion,
    return_dict=True,
    tokenize=True,
    return_assistant_tokens_mask=assistant_only_loss,
    tools=example.get("tools"),
    **example.get("chat_template_kwargs", {}),
)
# Fix transformers inconsistency: for VLMs, apply_chat_template returns lists of lists
# even for single examples, while for LLMs it returns lists of ints.
prompt_completion_processed = {
    k: v[0] if isinstance(v[0], list) else v
    for k, v in prompt_completion_processed.items()
}
prompt_completion_ids = prompt_completion_processed["input_ids"]
if "assistant_masks" in prompt_completion_processed:
    output["assistant_masks"] = prompt_completion_processed["assistant_masks"]
```

#### For Language Modeling (Lines 1011-1022)

```python
processed = processing_class.apply_chat_template(
    messages,
    return_dict=True,
    tokenize=True,
    return_assistant_tokens_mask=assistant_only_loss,
    tools=example.get("tools"),
    **example.get("chat_template_kwargs", {}),
)
# Fix transformers inconsistency: for VLMs, apply_chat_template returns lists of lists
# even for single examples, while for LLMs it returns lists of ints.
processed = {k: v[0] if isinstance(v[0], list) else v for k, v in processed.items()}
output = {k: processed[k] for k in ("input_ids", "assistant_masks") if k in processed}
```

**Key Points**:
- Uses `return_assistant_tokens_mask=True` when `assistant_only_loss=True`
- The tokenizer's chat template must support this feature
- Requires `{% generation %}` keyword in the chat template

#### Assistant Mask Validation (Lines 1026-1032)

```python
if "assistant_masks" in output and 1 not in output["assistant_masks"]:
    raise RuntimeError(
        "You're using `assistant_only_loss=True`, but at least one example has no assistant "
        "tokens. This usually means the tokenizer's chat template doesn't generate assistant "
        "masks — it may be missing the `{% generation %}` keyword. Please check the template and "
        "ensure it's correctly configured to support assistant masking."
    )
```

### Data Collator - Applying Assistant Masks

**File**: `/home/felipemello/forge/trl/trl/trainer/sft_trainer.py` (Lines 177-222)

```python
if "assistant_masks" in examples[0]:
    assistant_masks = [torch.tensor(example["assistant_masks"]) for example in examples]

# ... (padding logic) ...

if "assistant_masks" in examples[0]:
    assistant_masks = pad(
        assistant_masks, padding_value=0, padding_side="right", pad_to_multiple_of=self.pad_to_multiple_of
    )
    output["labels"][assistant_masks == 0] = -100
```

**Key Points**:
- `assistant_masks` are binary: 1 for assistant tokens, 0 for everything else
- Setting `labels[assistant_masks == 0] = -100` excludes non-assistant tokens from loss
- This handles multi-turn: only assistant responses contribute to loss

### Chat Template Integration

TRL relies on Transformers' tokenizer `apply_chat_template` method:

1. **Input**: List of messages with roles (`user`, `assistant`, `system`)
2. **Output**:
   - `input_ids`: Tokenized conversation
   - `assistant_masks` (optional): Binary mask for assistant tokens
3. **Template Requirement**: Chat template must include `{% generation %}` tags

---

## 5. Complete Masking Flow for Multi-Turn Conversations

### For GRPO/RLOO (Online RL)

1. **Generation Phase** (Lines 1383-1390):
   - Generate completions
   - Find first EOS token
   - Strip tokens after first EOS from completion_ids

2. **Scoring Phase** (Lines 1470-1473):
   - Create completion_mask with 1s for all completion tokens
   - Pad with 0s

3. **Optional Truncation Masking** (Lines 1480-1484):
   - If `mask_truncated_completions=True`
   - Check if last token is EOS/PAD
   - If not, zero out ENTIRE completion

4. **Loss Computation**:
   - `completion_mask` multiplied element-wise with per-token losses
   - Example (Line 1856): `loss = ((per_token_loss * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)).mean()`

### For SFT (Supervised Fine-Tuning)

1. **Tokenization Phase** (Lines 969-1033):
   - Apply chat template with `return_assistant_tokens_mask=True`
   - Get `assistant_masks` for multi-turn conversations
   - OR get `completion_mask` for prompt-completion format

2. **Collation Phase** (Lines 177-222):
   - Convert masks to tensors
   - Pad masks
   - Apply to labels: `labels[mask == 0] = -100`

3. **Loss Computation**:
   - Standard cross-entropy loss
   - Tokens with `label == -100` are automatically ignored

---

## 6. Key Differences from Other Approaches

### What TRL Does:

1. ✅ **Masks tokens after first EOS** (strips them during generation)
2. ✅ **Uses chat template for assistant masking** in multi-turn
3. ✅ **Provides optional truncation masking** (entire sequence)
4. ✅ **Handles both prompt-completion and conversational formats**

### What TRL Does NOT Do:

1. ❌ **No suffix length validation** after EOS
2. ❌ **No explicit checking** of how many tokens exist after EOS
3. ❌ **No warnings/errors** if suffix after EOS is non-zero
4. ❌ **No token-level truncation masking** (only sequence-level)

---

## 7. Code Examples

### Example 1: Creating Completion Mask in GRPO

**Location**: `/home/felipemello/forge/trl/trl/trainer/grpo_trainer.py:1470-1473`

```python
# Convert lists of token IDs to padded tensors
prompt_ids = [torch.tensor(ids, device=device) for ids in prompt_ids_list]
prompt_mask = [torch.ones_like(ids, dtype=torch.long) for ids in prompt_ids]
prompt_ids = pad(prompt_ids, padding_value=self.pad_token_id, padding_side="left")
prompt_mask = pad(prompt_mask, padding_value=0, padding_side="left")
completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids_list]
completion_mask = [torch.ones_like(ids, dtype=torch.long) for ids in completion_ids]
completion_ids = pad(completion_ids, padding_value=self.pad_token_id, padding_side="right")
completion_mask = pad(completion_mask, padding_value=0, padding_side="right")
```

### Example 2: Stripping Tokens After EOS

**Location**: `/home/felipemello/forge/trl/trl/trainer/grpo_trainer.py:1383-1390`

```python
# Mask everything after the first EOS token
is_eos = completion_ids == self.eos_token_id
# Initialize eos_idx to sequence length (no EOS found)
eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
# For sequences with EOS, find the first occurrence
eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
# Create sequence indices [0, 1, 2, ..., seq_len-1]
sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
# Mask includes tokens up to and including first EOS
completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()
# Extract only the masked tokens
prompt_ids = [p[m].tolist() for p, m in zip(prompt_ids, prompt_mask.bool(), strict=True)]
completion_ids = [c[m].tolist() for c, m in zip(completion_ids, completion_mask.bool(), strict=True)]
```

### Example 3: Assistant Mask Application in SFT

**Location**: `/home/felipemello/forge/trl/trl/trainer/sft_trainer.py:218-222`

```python
if "assistant_masks" in examples[0]:
    assistant_masks = pad(
        assistant_masks, padding_value=0, padding_side="right", pad_to_multiple_of=self.pad_to_multiple_of
    )
    output["labels"][assistant_masks == 0] = -100
```

### Example 4: Completion Mask in Loss Computation

**Location**: `/home/felipemello/forge/trl/trl/trainer/grpo_trainer.py:1856`

```python
if self.loss_type == "grpo":
    loss = ((per_token_loss * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)).mean()
    loss = loss / self.current_gradient_accumulation_steps
```

---

## 8. Relevant Configuration Options

### GRPO Configuration

- `mask_truncated_completions` (bool): Whether to mask entire truncated completions
- `max_completion_length` (int): Maximum length for completions
- `completion_only_loss` (bool): Whether to compute loss only on completions

### SFT Configuration

- `assistant_only_loss` (bool): Whether to compute loss only on assistant tokens
- `completion_only_loss` (bool): Whether to compute loss only on completion (for prompt-completion format)
- `max_length` (int): Maximum sequence length

---

## 9. File Reference Map

| Feature | File | Key Lines |
|---------|------|-----------|
| **GRPO Completion Mask Creation** | `trl/trainer/grpo_trainer.py` | 1470-1473 |
| **GRPO EOS Token Stripping** | `trl/trainer/grpo_trainer.py` | 1383-1390 |
| **GRPO Truncation Masking** | `trl/trainer/grpo_trainer.py` | 1480-1484 |
| **GRPO Loss Computation** | `trl/trainer/grpo_trainer.py` | 1856-1868 |
| **RLOO Completion Mask** | `trl/trainer/rloo_trainer.py` | 1261-1269 |
| **RLOO EOS Token Stripping** | `trl/trainer/rloo_trainer.py` | 1176-1183 |
| **SFT Assistant Mask Creation** | `trl/trainer/sft_trainer.py` | 969-985, 1011-1022 |
| **SFT Completion Mask Creation** | `trl/trainer/sft_trainer.py` | 1000-1003 |
| **Data Collator (Text)** | `trl/trainer/sft_trainer.py` | 85-222 |
| **Data Collator (Vision)** | `trl/trainer/sft_trainer.py` | 253-461 |
| **Chat Template Utilities** | `trl/data_utils.py` | 186-316 |

---

## 10. Recommendations Based on TRL's Approach

### For Multi-Turn Conversations:

1. **Use assistant_masks** from chat template (requires proper template with `{% generation %}`)
2. **Do NOT rely on suffix length checking** - TRL doesn't do this
3. **Leverage completion_mask** for prompt-completion format
4. **Trust EOS token stripping** during generation phase

### For Token-After-EOS Handling:

1. **TRL strips tokens after first EOS** during generation (transformers path)
2. **vLLM/rollout_func paths** handle this internally
3. **No need for explicit suffix validation** - handled by generation logic

### For Truncation Handling:

1. **Use `mask_truncated_completions`** to exclude truncated sequences entirely
2. **Check last token** for EOS/PAD to detect truncation
3. **Sequence-level masking** rather than token-level

---

## 11. Notable Design Choices

### Why TRL Doesn't Check Suffix Length:

1. **Generation-time stripping**: Tokens after EOS are removed during generation
2. **Mask-based approach**: Focuses on masking rather than validation
3. **Efficiency**: Avoids extra validation overhead
4. **vLLM handling**: When using vLLM, it manages this internally

### Why TRL Uses Assistant Masks:

1. **Multi-turn support**: Natural way to handle conversations with multiple user/assistant turns
2. **Tokenizer integration**: Leverages transformers' built-in chat template system
3. **Flexibility**: Works with any chat template that supports `{% generation %}`

### Why TRL Has Separate completion_mask and assistant_masks:

1. **completion_mask**: For prompt-completion format (single turn)
2. **assistant_masks**: For conversational format (multi-turn)
3. **Different use cases**: SFT vs RL training scenarios

---

## 12. Comparison with Potential Alternatives

### Alternative Approach: Explicit Suffix Validation

```python
# What TRL DOESN'T do (but could):
for ids in completion_ids_list:
    first_eos_idx = (ids == eos_token_id).nonzero(as_tuple=True)[0]
    if len(first_eos_idx) > 0:
        suffix_len = len(ids) - first_eos_idx[0] - 1
        if suffix_len > 0:
            logger.warning(f"Found {suffix_len} tokens after EOS")
```

**TRL's approach instead**: Strip during generation, trust the process.

### Alternative Approach: Token-Level Truncation Masking

```python
# What TRL DOESN'T do:
# Gradually mask tokens after some threshold, not entire sequence
```

**TRL's approach instead**: Sequence-level masking with `mask_truncated_completions`.

---

## 13. Summary Table

| Aspect | TRL's Approach | File Location |
|--------|----------------|---------------|
| **Completion Mask Creation** | Create 1s for actual tokens, 0s for padding | grpo_trainer.py:1470-1473 |
| **Tokens After EOS** | Strip during generation (transformers path) | grpo_trainer.py:1383-1390 |
| **Suffix Length Checking** | ❌ Not performed | N/A |
| **Chat Template** | Use `apply_chat_template` with `return_assistant_tokens_mask` | sft_trainer.py:969-985 |
| **Multi-Turn Masking** | Use `assistant_masks` from tokenizer | sft_trainer.py:218-222 |
| **Truncation Handling** | Sequence-level masking via `mask_truncated_completions` | grpo_trainer.py:1480-1484 |
| **Loss Computation** | Element-wise multiplication with mask | grpo_trainer.py:1856 |

---

## Conclusion

TRL's masking approach is **generation-centric** and **mask-based** rather than validation-based:

1. Tokens after EOS are **stripped during generation** (not validated post-hoc)
2. Multi-turn conversations use **assistant_masks from chat templates**
3. **No explicit suffix length checking** - relies on generation-time handling
4. **Sequence-level truncation masking** available via config option
5. Clean separation between **prompt-completion** (completion_mask) and **conversational** (assistant_masks) formats

This design prioritizes efficiency and integration with the generation process over explicit validation checks.
