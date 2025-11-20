# VERL Multi-Turn Conversation Masking Research

**Research Date:** 2025-11-19
**Objective:** Understand how VERL handles multi-turn conversation masking, EOS tokens, and suffix handling

---

## Executive Summary

VERL uses a **simple masking approach** for multi-turn conversations:
- **Loss masks are created incrementally** as messages are added
- **NO special EOS suffix stripping** - tokens after EOS are naturally masked via `response_mask`
- **NO explicit suffix length checking** after EOS tokens
- Chat template tokens (newlines, special tokens) are handled through the **incremental tokenization** approach

---

## 1. Loss Mask Creation for Multi-Turn Conversations

### 1.1 Schema-Level Loss Mask (`AsyncRolloutRequest`)

**File:** `/home/felipemello/forge/verl/verl/workers/rollout/schemas.py`

**Initial Setup (Line 202):**
```python
values["loss_mask"] = values["prompt_loss_mask"] = torch.zeros_like(values["input_ids"], dtype=torch.bool)
```
- Prompt tokens start with `loss_mask=0` (not trained)
- Loss mask is **boolean tensor** same shape as input_ids

**Key Method: `_update_input_ids()` (Lines 299-334):**
```python
def _update_input_ids(
    self,
    processing_class: PreTrainedTokenizer | PreTrainedTokenizerFast | ProcessorMixin,
    new_input_ids: torch.Tensor,
    attention_mask: bool,
    loss_mask: bool,
    new_multi_modal_inputs: Optional[dict[str, torch.Tensor]] = None,
) -> None:
    """
    Update the input_ids, attention_mask, position_ids, and loss_mask in additive manner.
    """
    self.input_ids = torch.cat([self.input_ids, new_input_ids], dim=-1)
    attention_mask = torch.ones_like(new_input_ids) * int(attention_mask)
    self.attention_mask = torch.cat([self.attention_mask, attention_mask], dim=-1)
    loss_mask = torch.ones_like(new_input_ids) * int(loss_mask)
    self.loss_mask = torch.cat([self.loss_mask, loss_mask], dim=-1)
    # ... position_ids update
```

**Usage Pattern:**
- `loss_mask=True` → tokens are trained (loss computed)
- `loss_mask=False` → tokens are NOT trained (loss masked out)

---

### 1.2 Adding Messages to Conversation

**User Messages (Lines 379-393):**
```python
def add_user_message(
    self,
    processing_class: PreTrainedTokenizer | PreTrainedTokenizerFast | ProcessorMixin,
    content: str,
) -> None:
    self.messages.append(Message(role="user", content=content))
    messages = [*BASE_CHAT_HISTORY, self.messages[-1]]
    # ... tokenize
    content_ids = self._handle_apply_chat_template(...)
    self._update_input_ids(processing_class, content_ids,
                          attention_mask=True, loss_mask=False)  # ← NOT trained
```

**Assistant Messages (Lines 395-412):**
```python
def add_assistant_message(
    self,
    processing_class: PreTrainedTokenizer | PreTrainedTokenizerFast | ProcessorMixin,
    content: str,
    content_ids: Optional[torch.Tensor] = None,
    tool_calls: Optional[list[OpenAIFunctionToolCall]] = None,
) -> None:
    self.messages.append(Message(role="assistant", content=content, tool_calls=tool_calls))
    # ... tokenize
    content_ids = self._handle_apply_chat_template(...)
    self._update_input_ids(processing_class, content_ids,
                          attention_mask=True, loss_mask=True)  # ← TRAINED
```

**Tool Response Messages (Lines 414-474):**
```python
def add_tool_response_messages(
    self,
    processing_class: PreTrainedTokenizer | PreTrainedTokenizerFast | ProcessorMixin,
    contents: list[ToolResponse],
) -> None:
    # ... add tool messages
    self._update_input_ids(
        processing_class,
        content_ids,
        attention_mask=True,
        loss_mask=False,  # ← Tool outputs NOT trained
        new_multi_modal_inputs=multi_modal_inputs,
    )
```

**Summary:**
- **User messages:** `loss_mask=False`
- **Assistant messages:** `loss_mask=True`
- **Tool responses:** `loss_mask=False`

---

## 2. Handling Tokens AFTER EOS

### 2.1 Response Mask Creation

**File:** `/home/felipemello/forge/verl/verl/utils/torch_functional.py` (Lines 226-246)

**Key Function: `get_response_mask()`**
```python
def get_response_mask(response_id: torch.Tensor, eos_token: int | list[int] = 2, dtype=torch.int64):
    """
    end of sentence token can be int or list: 1 or [1, 2]
    e.g.
    response_id = torch.tensor([[20, 10, 34, 1, 0, 0, 0],
                                [78, 0, 76, 2, 1, 0, 0],
                                [23, 98, 1, 0, 0, 0, 0],
                                [33, 3, 98, 45, 1, 0, 0]])
    #eos_token=1
    response_mask:  tensor([[1, 1, 1, 1, 0, 0, 0],
                            [1, 1, 1, 1, 1, 0, 0],
                            [1, 1, 1, 0, 0, 0, 0],
                            [1, 1, 1, 1, 1, 0, 0]])
    #eos_token=[1,2]
    response_mask:  tensor([[1, 1, 1, 1, 0, 0, 0],
                            [1, 1, 1, 1, 0, 0, 0],
                            [1, 1, 1, 0, 0, 0, 0],
                            [1, 1, 1, 1, 1, 0, 0]])
    """
    eos_mask = torch.isin(response_id, torch.tensor(eos_token, device=response_id.device)).int()
    return (eos_mask.cumsum(dim=1) - eos_mask).eq(0).to(dtype)
```

**Behavior:**
- Creates mask with `1` up to and INCLUDING the first EOS token
- All tokens AFTER first EOS get mask `0`
- Supports multiple EOS tokens (can pass list)
- Uses cumulative sum trick: `(cumsum - mask).eq(0)`

### 2.2 Usage in Single-Turn Rollout

**File:** `/home/felipemello/forge/verl/verl/workers/rollout/sglang_rollout/sglang_rollout.py` (Lines 785-788)

```python
response_attention_mask = get_response_mask(
    response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype
)
attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)
```

**For Multi-Turn (Lines 1309-1311):**
```python
response_loss_mask = pad_sequence(response_loss_mask, batch_first=True, padding_value=0)
if response_loss_mask.shape[1] < self.config.response_length:
    response_loss_mask = pad_sequence_to_length(response_loss_mask, self.config.response_length, 0)
```

---

## 3. NO Suffix Stripping After EOS

### 3.1 Truncation Logic

**File:** `/home/felipemello/forge/verl/verl/workers/rollout/schemas.py` (Lines 658-673)

```python
def truncate_output_ids(
    self, processing_class: PreTrainedTokenizer | PreTrainedTokenizerFast | ProcessorMixin
) -> None:
    self.input_ids = self.input_ids[..., : self.max_model_len]
    self.attention_mask = self.attention_mask[..., : self.max_model_len]
    self.position_ids = self.position_ids[..., : self.max_model_len]
    self.loss_mask = self.loss_mask[..., : self.max_model_len]
    self.response_ids = self.input_ids[..., self.prompt_ids.shape[-1] :][..., : self.max_response_len]
    self.response_attention_mask = self.attention_mask[..., self.prompt_attention_mask.shape[-1] :][
        ..., : self.max_response_len
    ]
    self.response_position_ids = self.position_ids[..., self.prompt_position_ids.shape[-1] :][
        ..., : self.max_response_len
    ]
    self.response_loss_mask = self.loss_mask[..., self.prompt_loss_mask.shape[-1] :][..., : self.max_response_len]
```

**Observations:**
- Only truncates to `max_model_len` and `max_response_len`
- **NO checking for EOS token position**
- **NO removal of tokens after EOS**
- Tokens after EOS are naturally masked via `response_mask`

### 3.2 Finalization Process

**File:** `/home/felipemello/forge/verl/verl/workers/rollout/schemas.py` (Lines 551-648)

```python
def finalize(
    self,
    processing_class: PreTrainedTokenizer | PreTrainedTokenizerFast | ProcessorMixin,
    reward_scores: dict[str, list[float]],
    finish_reason_type: FinishReasonTypeEnum = FinishReasonTypeEnum.STOP,
) -> None:
    self.state = AsyncRolloutRequestStateEnum.COMPLETED
    self.reward_scores = reward_scores

    # Remove generation prompt if present
    self._remove_generation_prompt_ids_if_present()

    self.response_ids = self.input_ids[..., self.prompt_ids.shape[-1] :]

    # Tokenization sanity check (optional)
    if self.tokenization_sanity_check_mode != TokenizationSanityCheckModeEnum.DISABLE:
        # ... validation logic

    # Handle finish reason
    if finish_reason_type == FinishReasonTypeEnum.STOP:
        pass  # No special handling
    elif finish_reason_type == FinishReasonTypeEnum.LENGTH:
        pass  # No special handling

    self.truncate_output_ids(processing_class)  # Only length truncation
```

**Key Points:**
- `STOP` finish reason: no special handling
- `LENGTH` finish reason: no special handling
- Only calls `truncate_output_ids()` which does NOT strip after EOS

---

## 4. Chat Template Token Handling

### 4.1 Incremental Tokenization Approach

**File:** `/home/felipemello/forge/verl/verl/workers/rollout/schemas.py` (Lines 224-258)

**Key Method: `_handle_apply_chat_template()`**
```python
@staticmethod
def _handle_apply_chat_template(
    processing_class: PreTrainedTokenizer | PreTrainedTokenizerFast | ProcessorMixin,
    messages: list[Message],
    multi_modal_data: dict[str, Any],
    tools: Optional[list[OpenAIFunctionToolSchema]] = None,
    add_generation_prompt: bool = False,
    tokenize: bool = False,
    return_dict: bool = False,
):
    raw_prompt = processing_class.apply_chat_template(
        messages, tools=tools, add_generation_prompt=add_generation_prompt, tokenize=False
    )
    if not tokenize:
        return raw_prompt

    # Tokenize with processor or tokenizer
    if isinstance(processing_class, ProcessorMixin):
        images = images if len(images := multi_modal_data.get("image", [])) > 0 else None
        videos = videos if len(videos := multi_modal_data.get("video", [])) > 0 else None
        model_inputs = processing_class(text=[raw_prompt], images=images, videos=videos, return_tensors="pt")
    else:
        model_inputs = processing_class(text=[raw_prompt], return_tensors="pt")
```

**Usage Pattern:**
```python
# When adding a message, compute delta by using BASE_CHAT_HISTORY
messages = [*BASE_CHAT_HISTORY, self.messages[-1]]
content_ids = self._handle_apply_chat_template(
    processing_class, messages, multi_modal_data={},
    tools=tools, add_generation_prompt=False, tokenize=True
)[..., self.base_conv_wo_gen_prompt_end_pos :]  # Extract only the new tokens
```

**BASE_CHAT_HISTORY (Lines 31-34):**
```python
BASE_CHAT_HISTORY = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "I am a user."},
]
```

### 4.2 Generation Prompt Handling

**Lines 348-362:**
```python
def get_generation_prompt_ids(
    self, processing_class: PreTrainedTokenizer | PreTrainedTokenizerFast | ProcessorMixin
) -> list[int]:
    """
    Get the generation prompt ids for rollout engine.
    """
    generation_prompt_ids = (
        None
        if self.input_ids[..., -self.generation_prompt_ids.shape[-1] :].eq(self.generation_prompt_ids).all()
        else self.generation_prompt_ids
    )
    if generation_prompt_ids is not None:
        self._update_input_ids(processing_class, generation_prompt_ids,
                              attention_mask=True, loss_mask=False)  # Generation prompt NOT trained
```

**Generation Prompt Removal (Lines 541-549):**
```python
def _remove_generation_prompt_ids_if_present(self) -> None:
    """
    Remove generation prompt IDs from input tensors if they are present at the end.
    """
    if self.input_ids[..., -self.generation_prompt_ids.shape[-1] :].eq(self.generation_prompt_ids).all():
        self.input_ids = self.input_ids[..., : -self.generation_prompt_ids.shape[-1]]
        self.attention_mask = self.attention_mask[..., : -self.generation_prompt_ids.shape[-1]]
        self.position_ids = self.position_ids[..., : -self.generation_prompt_ids.shape[-1]]
        self.loss_mask = self.loss_mask[..., : -self.generation_prompt_ids.shape[-1]]
```

### 4.3 Tokenization Sanity Check

**File:** `/home/felipemello/forge/verl/verl/workers/rollout/schemas.py` (Lines 73-78, 566-640)

**TokenizationSanityCheckModeEnum:**
```python
class TokenizationSanityCheckModeEnum(str, Enum):
    DISABLE = "disable"
    STRICT = "strict"
    IGNORE_STRIPPABLE = "ignore_strippable"
```

**Validation Logic (Lines 566-640):**
```python
if self.tokenization_sanity_check_mode != TokenizationSanityCheckModeEnum.DISABLE:
    # Compare full chat template tokenization vs incremental
    full_prompt_ids = self._handle_apply_chat_template(
        processing_class, messages, multi_modal_data=self.multi_modal_data,
        tools=tools, add_generation_prompt=False, tokenize=True, return_dict=True
    )["input_ids"]

    if diffs := self._get_prompt_diffs(
        processing_class, full_prompt_ids, self.input_ids, diff_surrounding_chars=10
    ):
        log_warning = False
        if self.tokenization_sanity_check_mode == TokenizationSanityCheckModeEnum.STRICT:
            log_warning = True
        elif self.tokenization_sanity_check_mode == TokenizationSanityCheckModeEnum.IGNORE_STRIPPABLE:
            non_strippable_diffs_exist = any(
                d["full_prompt_chunk"].strip() or d["current_prompt_chunk"].strip() for d in diffs
            )
            if non_strippable_diffs_exist:
                log_warning = True
```

**Purpose:**
- Catches differences between full tokenization and incremental tokenization
- Useful for debugging chat template issues (e.g., extra newlines)
- `IGNORE_STRIPPABLE` mode allows whitespace-only differences

---

## 5. SFT Dataset Loss Mask Creation

**File:** `/home/felipemello/forge/verl/verl/utils/dataset/multiturn_sft_dataset.py`

### 5.1 Processing Messages (Lines 133-209)

**For Assistant Messages:**
```python
if is_assistant:
    generation_prompt_text = prev_applied_text_w_generation_prompt[len(prev_applied_text) :]
    generation_prompt_tokens = self.tokenizer.encode(
        generation_prompt_text,
        add_special_tokens=False,
    )
    _message_tokens = self.tokenizer.encode(
        cur_applied_text[len(prev_applied_text_w_generation_prompt) :],
        add_special_tokens=False,
    )
    message_tokens = generation_prompt_tokens + _message_tokens
    loss_mask = [0] * (len(generation_prompt_tokens)) + [1] * (
        len(message_tokens) - len(generation_prompt_tokens)
    )
```

**For Other Messages:**
```python
else:
    message_tokens = self.tokenizer.encode(
        cur_applied_text[len(prev_applied_text) :],
        add_special_tokens=False,
    )
    loss_mask = [0] * len(message_tokens)
```

### 5.2 Override Loss Mask (Lines 312-319)

```python
# override loss mask with mask in the dataset to handle multi-turn conversation
override_loss_mask = cur_messages.get("loss_mask", None)
if override_loss_mask is not None:
    if isinstance(override_loss_mask, np.ndarray):
        override_loss_mask = override_loss_mask.item()
    assert isinstance(override_loss_mask, int), f"loss_mask should be int, got {type(override_loss_mask)}"
    assert override_loss_mask in [0, 1], f"loss_mask should be 0 or 1, got {override_loss_mask}"
    loss_mask = [override_loss_mask] * len(tokens)
```

**Features:**
- Allows per-message `loss_mask` override in dataset
- Useful for training only specific assistant turns

---

## 6. Key Differences from Other Implementations

### 6.1 No Explicit Suffix Removal

**Unlike some implementations (e.g., OpenRLHF), VERL does NOT:**
- Check for tokens after EOS
- Strip suffix after EOS token
- Validate suffix length

**Instead, VERL:**
- Relies on `response_mask` to mask tokens after EOS during loss computation
- Keeps all generated tokens in the sequence
- Masks them out via attention mask and loss mask

### 6.2 Incremental Tokenization

**VERL uses incremental tokenization:**
- Each new message is tokenized relative to previous messages
- Uses `BASE_CHAT_HISTORY` to compute token deltas
- Validates with optional tokenization sanity check

**Benefits:**
- Explicit control over which tokens come from which messages
- Easy to assign loss masks per-message
- Handles multi-turn naturally

### 6.3 Simple Masking Philosophy

**Core principle:**
```
loss_mask[i] = 1  if token i should contribute to loss
             = 0  otherwise
```

**Applied to:**
- User messages: `loss_mask=0` (not trained)
- Assistant messages: `loss_mask=1` (trained)
- Tool responses: `loss_mask=0` (not trained)
- Tokens after EOS: `response_mask=0` (via `get_response_mask()`)

---

## 7. Code Flow Summary

### 7.1 Multi-Turn Rollout Flow

```
1. Initialize AsyncRolloutRequest
   └─> loss_mask = zeros (all prompt tokens)

2. For each turn:

   a. Generate assistant response
      └─> SGLang engine generates tokens

   b. Add assistant message
      └─> add_assistant_message(content, content_ids)
          └─> _update_input_ids(..., loss_mask=True)
              └─> Concatenate with loss_mask=1 for assistant tokens

   c. If tool call:
      └─> Execute tool
      └─> add_tool_response_messages(tool_responses)
          └─> _update_input_ids(..., loss_mask=False)
              └─> Concatenate with loss_mask=0 for tool tokens

   d. If interaction:
      └─> add_user_message(content)
          └─> _update_input_ids(..., loss_mask=False)
              └─> Concatenate with loss_mask=0 for user tokens

3. Finalize request
   └─> finalize()
       └─> Remove generation prompt if present
       └─> Truncate to max_model_len
       └─> Create response_loss_mask from loss_mask

4. Create batch data
   └─> Pad sequences
   └─> response_mask from response_loss_mask
```

### 7.2 Loss Computation Flow

```
1. During training (PPO/SFT):

   a. Forward pass
      └─> logits = model(input_ids, attention_mask)

   b. Compute loss
      └─> loss = criterion(logits, labels)
      └─> loss = loss * loss_mask  # Mask out non-assistant tokens
      └─> loss = loss * response_mask  # Mask out tokens after EOS

   c. Average
      └─> loss = loss.sum() / response_mask.sum()
```

---

## 8. File Reference Index

### Core Files

1. **`/home/felipemello/forge/verl/verl/workers/rollout/schemas.py`**
   - `AsyncRolloutRequest` class (Lines 81-673)
   - `_update_input_ids()` (Lines 299-334)
   - `add_user_message()` (Lines 379-393)
   - `add_assistant_message()` (Lines 395-412)
   - `add_tool_response_messages()` (Lines 414-474)
   - `finalize()` (Lines 551-657)
   - `truncate_output_ids()` (Lines 658-673)

2. **`/home/felipemello/forge/verl/verl/workers/rollout/sglang_rollout/sglang_rollout.py`**
   - `_async_rollout_a_request()` (Lines 807-1051)
   - `_req_level_generate_sequences()` (Lines 1103-1360)
   - Response mask creation (Lines 785-788, 1309-1311)

3. **`/home/felipemello/forge/verl/verl/utils/torch_functional.py`**
   - `get_response_mask()` (Lines 226-246)

4. **`/home/felipemello/forge/verl/verl/utils/dataset/multiturn_sft_dataset.py`**
   - `MultiTurnSFTDataset` class (Lines 47-392)
   - `_process_message_tokens()` (Lines 133-209)
   - Override loss mask (Lines 312-319)

### Supporting Files

5. **`/home/felipemello/forge/verl/verl/workers/rollout/hf_rollout.py`**
   - Single-turn rollout example (Lines 99-161)

6. **`/home/felipemello/forge/verl/verl/trainer/ppo/core_algos.py`**
   - GAE computation with response_mask (Lines 223-233, 605-615)

---

## 9. Conclusions

### What VERL Does

1. **Incremental Loss Mask Creation:**
   - Loss masks are built up incrementally as messages are added
   - Each message type has a specific loss_mask value
   - Assistant messages: trained (mask=1)
   - User/tool messages: not trained (mask=0)

2. **EOS Token Handling:**
   - Uses `get_response_mask()` to create mask with 0s after first EOS
   - **NO explicit suffix stripping**
   - Tokens after EOS remain in sequence but are masked
   - Supports multiple EOS tokens

3. **Chat Template Tokens:**
   - Handled through incremental `apply_chat_template()` calls
   - Generation prompt tokens explicitly managed
   - Optional tokenization sanity check validates consistency

### What VERL Does NOT Do

1. **NO suffix length checking** after EOS
2. **NO explicit truncation** at EOS position
3. **NO special handling** of tokens after EOS beyond masking
4. **NO stripping** of padding tokens after EOS

### Design Philosophy

VERL's approach is **simple and mask-based**:
- Generate full sequences (including tokens after EOS)
- Use masks to control which tokens contribute to loss
- Rely on attention masks and loss masks rather than sequence manipulation
- Keep sequences intact for easier debugging and validation

This differs from approaches that actively remove or strip tokens after EOS, which can be more complex but may save memory.

---

## 10. Comparison to Your Implementation

**Your current approach (based on previous discussions):**
- Strips tokens after EOS using `cut_by_token_indices_based_on_suffix_length()`
- Explicitly checks suffix length after EOS
- Validates that no content appears after EOS

**VERL's approach:**
- Keeps all tokens after EOS
- Masks them via `response_mask`
- No explicit suffix validation

**Key Question:**
Should you adopt VERL's simpler masking approach, or continue with explicit suffix stripping?

**Trade-offs:**

| Aspect | VERL (Masking) | Your Approach (Stripping) |
|--------|----------------|---------------------------|
| Simplicity | ✅ Simpler | ❌ More complex |
| Memory | ❌ Stores unused tokens | ✅ Removes unused tokens |
| Debugging | ✅ Full sequence visible | ❌ Truncated sequence |
| Validation | ❌ No suffix checks | ✅ Explicit validation |
| Multi-turn | ✅ Natural fit | ⚠️ Requires care |

**Recommendation:**
For multi-turn conversations, VERL's masking approach is likely **simpler and less error-prone**. Consider adopting it unless memory is a critical constraint.
