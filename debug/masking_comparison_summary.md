# Multi-Turn Masking: Library Comparison Summary

**Date:** 2025-11-19
**Purpose:** Compare how different RL libraries handle tokens after EOS in multi-turn conversations

---

## Quick Comparison Table

| Library | Strips After EOS? | Checks Suffix Length? | How They Handle Post-EOS Tokens |
|---------|-------------------|----------------------|----------------------------------|
| **VERL** | ❌ No | ❌ No | Masks them out with `get_response_mask()` using cumsum trick |
| **TRL** | ✅ Yes | ❌ No | Strips during generation using `argmax` to find first EOS |
| **Prime-RL** | ❌ No | ❌ No | Takes ALL tokens from vLLM, delegates to verifiers library |
| **Tinker-Cookbook** | ❌ No (training)<br>✅ Yes (inference) | ❌ No | Includes EOS in training, strips only during parsing |
| **NeMo-RL** | ❌ No | ❌ No | Role-based masking, trusts chat template |
| **Forge (Current)** | ✅ Yes | ✅ Yes | Validates suffix_len==0, strips in TokenAccumulator |

---

## Detailed Findings

### 1. VERL - Mask-Based Approach

**Philosophy:** Keep sequences intact, use masks to control training

```python
# verl/verl/utils/reward_score/rl.py:165-173
def get_response_mask(sequences, eos_token_id):
    """Create mask: 1 up to (and including) first EOS, 0 after"""
    eos_mask = sequences.eq(eos_token_id)
    # Cumsum trick: once we hit EOS, all future positions become 1
    # Subtract eos_mask to exclude positions before first EOS
    # Result: 0 for valid tokens (including first EOS), 1 for post-EOS
    return (eos_mask.cumsum(dim=1) - eos_mask).eq(0)
```

**Key Points:**
- ✅ Elegant solution using cumsum
- ✅ No sequence manipulation
- ✅ Preserves full sequence for debugging
- ⚠️ Still has tokens after EOS in the tensor

**Files:**
- `verl/verl/utils/reward_score/rl.py:165-173`
- `verl/verl/workers/rollout/vllm_rollout/vllm_rollout.py:400-500`

---

### 2. TRL - Stripping Approach

**Philosophy:** Remove tokens after first EOS during generation

```python
# trl/grpo_trainer.py:1383-1390
# Find first occurrence of EOS
eos_indices = (completions == generation_config.eos_token_id).long().argmax(dim=-1)

# Strip everything after first EOS
for i, (eos_idx, completion) in enumerate(zip(eos_indices, completions)):
    if eos_idx > 0:  # If EOS found
        # Exclude tokens after EOS
        completions[i, eos_idx + 1:] = tokenizer.pad_token_id
        completion_masks[i, eos_idx + 1:] = 0
```

**Key Points:**
- ✅ Actively removes post-EOS tokens
- ✅ Simple argmax approach
- ⚠️ No validation of how many tokens removed
- ⚠️ Assumes first EOS is the real one

**Files:**
- `trl/trl/trainer/grpo_trainer.py:1383-1390`
- `trl/trl/trainer/rloo_trainer.py:1340-1347`

---

### 3. Prime-RL - Trust vLLM Approach

**Philosophy:** Accept whatever vLLM generates, no post-processing

```python
# Prime-RL delegates to verifiers library
# Uses vLLM response tokens directly without re-tokenization
# No stripping or validation of post-EOS tokens
```

**Key Points:**
- ✅ Simple - trusts vLLM output
- ✅ Uses external verifiers library
- ⚠️ Could train on garbage if vLLM generates extra tokens
- ⚠️ No safeguards for malformed responses

**Files:**
- `prime-rl/src/prime_rl/trainer/rl/rollout_worker.py`
- External: `verifiers` library

---

### 4. Tinker-Cookbook - Hybrid Approach

**Philosophy:** Include EOS in training, strip only during parsing

```python
# tinker_cookbook/renderers.py:140-162
def parse_chat_message_assistant(text):
    """Parse response, stopping at first EOS"""
    for stop_sequence in self.renderer.stop_sequences:
        if stop_sequence in text:
            text = text.split(stop_sequence)[0]
    return text
```

**Key Points:**
- ✅ EOS tokens get weight=1.0 (trained)
- ✅ Uses stop sequences during sampling
- ✅ Only strips during inference/parsing
- ⚠️ Training data includes full sequences

**Files:**
- `tinker_cookbook/renderers.py:84-162`
- `tinker_cookbook/configs/training.py`

---

### 5. NeMo-RL - Role-Based Masking

**Philosophy:** Mask based on message role, trust chat template

```python
# RL/nemo_rl/data/llm_message_utils.py:141-176
def add_loss_mask_to_message_log(message_log):
    """Add loss masks based on role"""
    for message in message_log:
        if message['role'] == 'assistant':
            message['loss_mask'] = torch.ones_like(token_ids)
        else:
            message['loss_mask'] = torch.zeros_like(token_ids)
```

**Key Points:**
- ✅ Simple role-based approach
- ✅ Trusts tokenizer.apply_chat_template()
- ⚠️ No validation of token sequences
- ⚠️ No special EOS handling

**Files:**
- `RL/nemo_rl/data/llm_message_utils.py:141-176`
- `RL/nemo_rl/models/generation/vllm/vllm_worker_async.py:40-121`

---

## Our Bug: Tokens After EOS with response_mask=1

### The Problem

In our `TokenAccumulator`, when adding an assistant response:

```python
# Current code in TokenAccumulator.add_assistant_response
assistant_tokens = self._tokenize_delta(message, "assistant")
# assistant_tokens includes: [prefix, content, EOS, NEWLINE]
#                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#                              ALL marked as response_mask=True!

mask = [False] * prefix_len + [True] * (len(assistant_tokens) - prefix_len)
self._accumulate(assistant_tokens, mask=mask)
```

Then when we create loss_mask:
```python
loss_mask = torch.roll(response_mask, shifts=-1, dims=0).float()
loss_mask[-1] = 0.0
```

Result:
```
Pos 653: content      response_mask=1  loss_mask=1  ✓
Pos 654: EOS          response_mask=1  loss_mask=1  ✗ BUG! Training to predict newline
Pos 655: newline      response_mask=1  loss_mask=0  ✗ BUG! Newline is part of response!
Pos 656: <|im_start|> response_mask=0  loss_mask=0  ✓
```

---

## Solutions Comparison

### Option 1: VERL Approach - Mask Post-EOS Tokens

**What to do:**
- Keep tokens in sequence
- Create `get_response_mask()` to mask positions after first EOS
- Use this when creating loss_mask

**Pros:**
- ✅ No sequence manipulation
- ✅ Full sequence preserved for debugging
- ✅ Clean separation of concerns

**Cons:**
- ⚠️ Need to implement cumsum logic
- ⚠️ Tokens still in memory (minor)

**Code change:**
```python
def create_loss_mask_with_eos_handling(response_mask, all_token_ids, eos_token_id):
    # First, shift response_mask
    loss_mask = torch.roll(response_mask, shifts=-1, dims=0).float()
    loss_mask[-1] = 0.0

    # Then, mask out positions at or after EOS
    eos_mask = (all_token_ids == eos_token_id)
    # Cumsum: after first EOS, all positions become > 0
    post_eos_mask = (eos_mask.cumsum(dim=0) > 0)
    loss_mask[post_eos_mask] = 0.0

    return loss_mask
```

### Option 2: TRL Approach - Strip After EOS in TokenAccumulator

**What to do:**
- When adding assistant response, find first EOS and truncate
- Only add tokens up to (and including) EOS

**Pros:**
- ✅ Simple - just find and truncate
- ✅ Cleaner sequences

**Cons:**
- ⚠️ Modifies sequences
- ⚠️ Loses information about what was generated

**Code change:**
```python
def add_assistant_response(self, response_text, response_token_ids, ...):
    # Find first EOS
    if self.eos_token_id in response_token_ids:
        eos_idx = response_token_ids.index(self.eos_token_id)
        response_token_ids = response_token_ids[:eos_idx + 1]  # Include EOS
        # Re-decode to get matching text
        response_text = self.tokenizer.decode(response_token_ids)

    # Continue with delta tokenization...
```

### Option 3: Tinker-Cookbook Approach - Include EOS, Rely on Stop Sequences

**What to do:**
- Accept that sequences may have tokens after EOS
- Mask them in loss_mask creation
- Use stop sequences during sampling

**Pros:**
- ✅ Matches vLLM behavior
- ✅ Simple

**Cons:**
- ⚠️ Doesn't solve our current bug

---

## Recommendation

**Best solution: Hybrid of VERL + TRL**

1. **In TokenAccumulator** (TRL approach):
   - Strip tokens after first EOS when adding assistant responses
   - This prevents the newline from being added to `accumulated_tokens`

2. **In loss_mask creation** (VERL approach as safeguard):
   - Add EOS masking logic as defensive programming
   - Handle edge cases where EOS might slip through

**Why this is best:**
- ✅ Prevents root cause (no post-EOS tokens in accumulator)
- ✅ Defensive (mask them anyway if they appear)
- ✅ Matches what vLLM actually generates
- ✅ Cleaner sequences

---

## Implementation Plan

1. **Fix TokenAccumulator.add_assistant_response():**
```python
def add_assistant_response(self, response_text, response_token_ids, ...):
    # Check for EOS and truncate
    if response_token_ids and response_token_ids[-1] != self.eos_token_id:
        return self._mark_truncated(TruncationReason.AGENT_TOO_LONG)

    # Find first EOS (in case there are multiple)
    eos_positions = [i for i, tid in enumerate(response_token_ids) if tid == self.eos_token_id]
    if eos_positions:
        first_eos = eos_positions[0]
        if first_eos < len(response_token_ids) - 1:
            # There are tokens after first EOS - truncate
            response_token_ids = response_token_ids[:first_eos + 1]
            # Note: response_text may be stale now, but we don't use it for tokenization

    # Continue with existing delta tokenization logic...
```

2. **Add defensive EOS masking in do_single_rollout():**
```python
# After creating loss_mask with torch.roll
loss_mask_tensor = torch.roll(response_mask_tensor, shifts=-1, dims=0).float()
loss_mask_tensor[-1] = 0.0

# Defensive: mask positions AT eos tokens
eos_positions = (all_tokens_tensor == eos_token_id)
loss_mask_tensor[eos_positions] = 0.0
```

This gives us defense-in-depth!

---

## Testing

After implementation, verify with `debug/verify_eos_hypothesis.py`:
- Should show 0 EOS positions with loss_mask=1
- Should show 0 suspicious tokens after EOS with response_mask=1
- KL at EOS should be same as non-EOS (near zero)
