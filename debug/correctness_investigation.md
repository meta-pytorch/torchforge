# Multi-Turn RL Training Correctness Investigation (UPDATED)

**Date:** 2025-11-19
**Code:** `apps/blackjack/main_v2.py`
**Objective:** Root-cause analysis and first-principles fix for next-token prediction in GRPO training

---

## Executive Summary

### THE FUNDAMENTAL PROBLEM

**Current Implementation Confuses "Response Tokens" with "Trainable Positions"**

- **response_mask marks which tokens ARE responses** (the generated output)
- **But we need a mask for which POSITIONS contribute to loss** (shifted by 1!)
- These are NOT the same due to next-token prediction shift

### Root Causes Identified:

1. **❌ CRITICAL: Logits-Tokens Misalignment** - `compute_logprobs` uses wrong positions
2. **❌ CRITICAL: Mask Naming Confusion** - "response_mask" should be "response_token_mask"
3. **❌ CRITICAL: Missing Training Mask** - Need `training_mask[i] = 1.0 if response_token_mask[i+1]`
4. **❌ Targets Created But Unused** - Extra computation that's never used

---

## Part 1: Understanding Next-Token Prediction

### The Fundamental Shift

In causal language models:

```
Input tokens:    [A,  B,  C,  D,  E]
Model processes: A→  AB→ ABC→ ABCD→ ABCDE→

Logits produced:
  logits[0] = P(? | A)      → predicts B
  logits[1] = P(? | AB)     → predicts C
  logits[2] = P(? | ABC)    → predicts D
  logits[3] = P(? | ABCD)   → predicts E
  logits[4] = P(? | ABCDE)  → predicts F (next token after E)
```

**Key Insight:** `logits[i]` predicts `tokens[i+1]`, NOT `tokens[i]`

### Why This Matters for Masks

```
Sequence: [System, User, Agent_Response, EOS, User, ...]

response_token_mask:  [0, 0, 1, 1, 0, ...]
                       ↑  ↑  ↑  ↑  ↑
                   Which tokens ARE responses

training_mask:        [0, 1, 1, 0, 0, ...]
                       ↑  ↑  ↑  ↑  ↑
              Which POSITIONS contribute to loss

Position 1 predicts token 2 (Agent_Response) → trainable!
Position 2 predicts token 3 (EOS) → trainable!
Position 3 predicts token 4 (User) → NOT trainable! (don't predict after EOS)
```

**Formula:** `training_mask[i] = 1.0 if (response_token_mask[i+1] == 1 AND tokens[i] != EOS)`

---

## Part 2: How Other Libraries Handle This

### 2.1 VERL Approach

**File:** `/home/felipemello/forge/verl/verl/workers/rollout/schemas.py`

VERL **explicitly separates** three different masks:

1. **`attention_mask`** - Valid tokens vs padding (for attention ops)
2. **`response_mask`** - Which tokens are responses (what was generated)
3. **`loss_mask`** - Which positions contribute to loss (trainable positions)

**Key Code:**
```python
class AsyncRolloutRequest:
    loss_mask: Optional[torch.Tensor] = None           # Trainable positions
    response_mask: Optional[torch.Tensor] = None       # Response tokens

# When adding assistant message:
self._update_input_ids(new_tokens, attention_mask=True, loss_mask=True)

# When adding user message:
self._update_input_ids(new_tokens, attention_mask=True, loss_mask=False)
```

**Loss Computation:**
```python
# File: verl/workers/roles/utils/losses.py
response_mask = data["response_mask"].to(bool)
loss = -masked_sum(log_prob, response_mask) / batch_num_tokens
```

**Insight:** VERL uses `response_mask` in loss, but this is actually the loss_mask (confusing naming). They handle the shift by rolling the mask.

### 2.2 TRL Approach

**File:** `/home/felipemello/forge/trl` (multiple files)

TRL uses **`completion_mask`** to mark trainable tokens:

```python
completion_mask = torch.ones_like(completion_ids)  # All response tokens trainable
completion_mask = completion_mask * (~is_truncated)  # Except truncated ones

# Loss:
masked_loss = per_token_loss * completion_mask
loss = masked_loss.sum() / completion_mask.sum()
```

**Insight:** TRL's `completion_mask` marks response tokens, and they apply it directly in loss (assumes logprobs are already properly aligned).

### 2.3 Prime-RL Approach

**File:** `/home/felipemello/forge/prime-rl/src/prime_rl/trainer/rl/loss.py`

Prime-RL explicitly passes **`loss_mask`** to the loss function:

```python
def compute_loss(
    trainer_logprobs: Float[Tensor, "seq"],
    inference_logprobs: Float[Tensor, "seq"],
    advantages: Float[Tensor, "seq"],
    loss_mask: Bool[Tensor, "seq"],  # <-- Explicit trainable positions mask
    ...
):
    # Apply mask
    keep_mask = loss_mask & ~is_masked
    loss = (-importance_ratio * advantages)[keep_mask].sum()
```

**Insight:** Prime-RL makes it explicit - `loss_mask` indicates which positions are trainable.

### 2.4 Common Pattern Across Libraries

All three libraries:
1. **Store a mask with episodes** (response_mask, completion_mask, or loss_mask)
2. **Use it in loss computation** via element-wise multiplication or indexing
3. **Treat mask as float (0.0/1.0)** for easy multiplication in loss

**None of them derive the mask from targets!** The mask is a first-class citizen in the episode data.

---

## Part 3: Current Implementation Issues

### Issue 1: ❌ Logits-Tokens Misalignment in `compute_logprobs`

**Location:** `apps/blackjack/main_v2.py` line 1020, `src/forge/actors/reference_model.py` line 190

**Current Code:**
```python
# In simple_grpo_loss:
logprobs = compute_logprobs(logits, all_tokens, align=False)

# In ReferenceModel.forward:
logprobs = compute_logprobs(logits, input_ids, align=False)
```

**What `compute_logprobs` does (align=False):**
```python
# From src/forge/util/ops.py
logprobs = -F.cross_entropy(
    scaled_logits_fp32.reshape(-1, vocab_size),
    input_ids.reshape(-1).long(),
    reduction="none",
)
```

This computes: `logprobs[i] = log P(input_ids[i] | logits[i])`

**But `logits[i]` predicts `input_ids[i+1]`, NOT `input_ids[i]`!**

**Correct Approach (Option 1 - Use targets):**
```python
# Create targets (already shifted)
targets = create_next_token_targets(all_tokens, response_mask, eos_token_id)

# Compute logprobs for targets
logprobs = compute_logprobs(logits, targets, align=False)

# Mask out IGNORE positions
valid_mask = (targets != CROSS_ENTROPY_IGNORE_IDX)
logprobs = logprobs * valid_mask.float()
```

**Correct Approach (Option 2 - Manual shift):**
```python
# Shift both logits and tokens
logits_shifted = logits[:, :-1, :]   # [b, seq_len-1, vocab]
tokens_to_pred = all_tokens[:, 1:]    # [b, seq_len-1]

# Compute logprobs
logprobs = compute_logprobs(logits_shifted, tokens_to_pred, align=False)

# Pad back to original length
logprobs = F.pad(logprobs, (1, 0), value=0.0)  # [b, seq_len]
```

### Issue 2: ❌ Mask Naming and Semantics

**Current Name:** `response_mask`

**Current Definition (from your comment):**
```python
response_mask: torch.Tensor  # CRITICAL: Mask for training
                             # Shape: (seq_len,)
                             # 1.0 = train on this token (LLM output)
                             # 0.0 = skip this token (prompt, tool result)
```

**The Problem:** The comment says "train on this token", but due to the shift, **we actually train on the PREVIOUS position!**

**Better Naming:**
- `response_token_mask` - Marks which tokens ARE responses
- `training_mask` or `loss_mask` - Marks which POSITIONS contribute to loss

**Relationship:**
```python
# Convert from response tokens to trainable positions
training_mask = torch.zeros_like(response_token_mask, dtype=torch.float)
for i in range(len(tokens) - 1):
    if response_token_mask[i+1] and tokens[i] != eos_token_id:
        training_mask[i] = 1.0
```

**Or derive from targets:**
```python
training_mask = (targets != CROSS_ENTROPY_IGNORE_IDX).float()
```

### Issue 3: ❌ Targets Created But Never Used

**Created:** Line 796-798 in `do_single_rollout`
**Used:** Nowhere! (not in collate, not in loss)

**Current `collate` function** (lines 950-957):
```python
target = {
    "all_tokens": all_tokens,
    "response_mask": response_masks,  # This is actually response_token_mask
    "ref_logprobs": ref_logprobs,
    "advantages": advantages,
}
# targets field is missing!
```

**Options:**
1. **DELETE** `create_next_token_targets` call (unused code)
2. **USE** targets to derive training_mask: `mask = (targets != IGNORE).float()`
3. **USE** targets in loss instead of all_tokens (cleaner, more explicit)

---

## Part 4: Concrete Example - "Hello there" and "I am bob"

See `debug/test_create_next_token_targets.py` for executable code.

### Sequence:

```
Index  Token       ID   Response_Mask  Target       Training_Mask
-----  --------  ----  -------------  -----------  -------------
0      Prompt      1        0          IGNORE          0.0
1      prompt      2        0          IGNORE          1.0  ← predicts "Hello" (idx 2)
2      Hello       3        1          4 (there)       1.0  ← predicts "there"
3      there       4        1          100 (EOS)       1.0  ← predicts EOS
4      EOS       100        1          IGNORE          0.0  ← don't predict after EOS
5      Prompt      5        0          IGNORE          0.0
6      prompt      6        0          IGNORE          1.0  ← predicts "I" (idx 7)
7      I           7        1          8 (am)          1.0  ← predicts "am"
8      am          8        1          9 (bob)         1.0  ← predicts "bob"
9      bob         9        1          100 (EOS)       1.0  ← predicts EOS
10     EOS       100        1          IGNORE          0.0  ← don't predict after EOS
```

### Key Observations:

1. **Response tokens (response_mask=1):** 7 tokens (Hello, there, EOS, I, am, bob, EOS)
2. **Training positions (training_mask=1):** 5 tokens (indices 1, 2, 3, 6, 7, 8, 9)
3. **The shift:** Position 1 (token="prompt") trains to predict position 2 (token="Hello")
4. **EOS handling:** EOS is in response_mask, but its position has training_mask=0

### Loss Computation:

```python
# Current (WRONG):
logprobs = compute_logprobs(logits, all_tokens, align=False)  # Misaligned!
masked_loss = per_token_loss * response_mask  # Wrong mask!
loss = masked_loss.sum() / response_mask.sum()

# Correct (Option 1 - fix alignment + use training_mask):
logprobs = compute_logprobs(logits[:, :-1], all_tokens[:, 1:], align=False)
logprobs = F.pad(logprobs, (1, 0), value=0.0)
training_mask = derive_training_mask(response_mask, all_tokens, eos_token_id)
masked_loss = per_token_loss * training_mask
loss = masked_loss.sum() / training_mask.sum()

# Correct (Option 2 - use targets):
targets = create_next_token_targets(all_tokens, response_mask, eos_token_id)
training_mask = (targets != CROSS_ENTROPY_IGNORE_IDX).float()
logprobs = compute_logprobs_from_targets(logits, targets)  # Helper function
masked_loss = per_token_loss * training_mask
loss = masked_loss.sum() / training_mask.sum()
```

---

## Part 5: Recommended Fix (First Principles)

### Step 1: Update Episode Data Structure

**In `apps/blackjack/main_v2.py` lines 92-112:**

```python
@dataclass
class Episode:
    """Episode data for GRPO training."""

    # Required fields
    episode_id: str
    all_token_ids: torch.Tensor  # [seq_len] - Full conversation tokens
    targets: torch.Tensor        # [seq_len] - Next-token targets (with IGNORE)
    reward: float

    # Optional fields
    task_name: str = "blackjack"
    policy_version: int = 0
    is_truncated: bool = False
    advantage: float | None = None
    logprobs: torch.Tensor | None = None      # [seq_len]
    ref_logprobs: torch.Tensor | None = None  # [seq_len]
    metadata: dict[str, Any] = field(default_factory=dict)
    message_log: list[dict[str, str]] | None = None
```

**Key Change:** Remove `response_mask` from Episode, keep `targets`. The training mask is derived from targets.

### Step 2: Update Collate Function

**In `apps/blackjack/main_v2.py` lines 914-962:**

```python
def collate(
    batches: list[list[Episode]],
    pad_id: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    inputs = []
    targets_list = []

    for batch in batches:
        # Stack all tensors
        all_tokens = [e.all_token_ids for e in batch]
        all_tokens = torch.nn.utils.rnn.pad_sequence(
            all_tokens, batch_first=True, padding_value=pad_id
        )

        # Stack targets
        targets_batch = [e.targets for e in batch]
        targets_batch = torch.nn.utils.rnn.pad_sequence(
            targets_batch, batch_first=True, padding_value=CROSS_ENTROPY_IGNORE_IDX
        )

        # Derive training mask from targets
        training_mask = (targets_batch != CROSS_ENTROPY_IGNORE_IDX).float()

        # Stack ref_logprobs
        ref_logprobs = [e.ref_logprobs for e in batch]
        ref_logprobs = torch.nn.utils.rnn.pad_sequence(
            ref_logprobs, batch_first=True, padding_value=0.0
        )

        # Advantages
        advantages = torch.tensor([e.advantage for e in batch]).unsqueeze(-1)

        # Create input and target dicts
        input = {"tokens": all_tokens}
        target = {
            "targets": targets_batch,        # Now included!
            "training_mask": training_mask,   # Derived from targets
            "ref_logprobs": ref_logprobs,
            "advantages": advantages,
        }

        inputs.append(input)
        targets_list.append(target)

    return inputs, targets_list
```

### Step 3: Fix `simple_grpo_loss`

**In `apps/blackjack/main_v2.py` lines 997-1039:**

```python
def simple_grpo_loss(
    logits: torch.Tensor,      # [b, seq_len, vocab]
    targets: torch.Tensor,     # [b, seq_len] - Next-token targets
    training_mask: torch.Tensor,  # [b, seq_len] - 1.0 for trainable positions
    ref_logprobs: torch.Tensor,   # [b, seq_len]
    advantages: torch.Tensor,     # [b, 1]
    beta: float = 0.1,
) -> torch.Tensor:
    """
    Simple GRPO loss with proper next-token prediction alignment.

    Args:
        logits: Model logits [b, seq_len, vocab_size]
        targets: Next-token targets [b, seq_len] (with IGNORE for non-trainable)
        training_mask: 1.0 for trainable positions, 0.0 otherwise
        ref_logprobs: Reference logprobs [b, seq_len]
        advantages: Advantages [b, 1]
        beta: KL penalty coefficient
    """
    # Compute policy logprobs using targets (properly aligned)
    # Option 1: Use a helper that handles IGNORE
    logprobs = compute_logprobs_from_targets(logits, targets)  # [b, seq_len]

    # Option 2: Manual computation
    # Shift logits to align with targets
    logits_shifted = logits[:, :-1, :]  # [b, seq_len-1, vocab]
    targets_shifted = targets[:, 1:]     # [b, seq_len-1]

    # Compute logprobs
    logprobs_shifted = compute_logprobs(logits_shifted, targets_shifted, align=False)
    logprobs = F.pad(logprobs_shifted, (1, 0), value=0.0)  # [b, seq_len]

    # Mask out IGNORE positions
    logprobs = logprobs * training_mask
    ref_logprobs = ref_logprobs * training_mask

    # KL divergence (only on trainable positions)
    kl = torch.exp(ref_logprobs - logprobs) - (ref_logprobs - logprobs) - 1

    # Policy loss
    per_token_policy_loss = torch.exp(logprobs - logprobs.detach()) * advantages
    per_token_loss = -(per_token_policy_loss - beta * kl)

    # Masked average
    loss = (
        (per_token_loss * training_mask).sum(dim=1) / (training_mask.sum(dim=1).clamp(min=1.0))
    ).mean()

    return loss
```

### Step 4: Fix Reference Model

**In `src/forge/actors/reference_model.py` lines 127-194:**

```python
@endpoint
async def forward(
    self, input_ids: torch.Tensor, return_logprobs: bool, targets: torch.Tensor = None
) -> torch.Tensor:
    """
    Args:
        input_ids: Input token ids [batch, seq_len]
        return_logprobs: Whether to return logprobs
        targets: Next-token targets [batch, seq_len] (optional, for proper alignment)
    """
    # ... forward pass code ...

    logits = self.model(input_ids)

    if not return_logprobs:
        return logits
    else:
        if targets is not None:
            # Use targets for proper alignment
            logprobs = compute_logprobs_from_targets(logits, targets)
        else:
            # Fallback: manual shift
            logits_shifted = logits[:, :-1, :]
            tokens_shifted = input_ids[:, 1:]
            logprobs = compute_logprobs(logits_shifted, tokens_shifted, align=False)
            logprobs = F.pad(logprobs, (1, 0), value=0.0)

        return logprobs
```

### Step 5: Create Helper Function

**In `src/forge/util/ops.py`:**

```python
def compute_logprobs_from_targets(
    logits: torch.Tensor,      # [b, seq_len, vocab]
    targets: torch.Tensor,     # [b, seq_len] with IGNORE for non-trainable
    ignore_index: int = -100,
) -> torch.Tensor:
    """
    Compute log probabilities for next-token targets.

    Properly handles the shift: logits[i] predicts targets[i+1].
    Positions with targets[i] == ignore_index get logprob = 0.0.

    Args:
        logits: Model logits [b, seq_len, vocab_size]
        targets: Next-token targets [b, seq_len]
        ignore_index: Value in targets to ignore

    Returns:
        logprobs: Log probabilities [b, seq_len]
    """
    batch_size, seq_len, vocab_size = logits.shape

    # Shift: logits[i] predicts targets[i+1]
    # But targets are already shifted! targets[i] = all_tokens[i+1]
    # So we compute: logits[i] should match targets[i]

    # Actually, there's confusion here. Let me reclarify:
    # If targets[i] = all_tokens[i+1], then logits[i-1] predicts targets[i]
    # So we need: logits[:-1] vs targets[1:]? No...

    # CORRECTION: targets are created such that targets[i] is what position i should predict.
    # create_next_token_targets does: targets[i] = all_tokens[i+1]
    # This means: at position i, we should predict targets[i]
    # And logits[i] gives the distribution for position i's prediction
    # So they're ALREADY aligned!

    # Cast to fp32 for numerical stability
    logits_fp32 = logits.float()

    # Compute cross-entropy (negative log prob)
    logprobs = -F.cross_entropy(
        logits_fp32.reshape(-1, vocab_size),
        targets.reshape(-1).long(),
        reduction="none",
        ignore_index=ignore_index,
    )

    logprobs = logprobs.reshape(batch_size, seq_len)

    # Set logprobs to 0 for ignored positions
    logprobs = logprobs * (targets != ignore_index).float()

    return logprobs
```

---

## Part 6: Summary of Findings

| Issue | Severity | Current State | Recommended Fix |
|-------|----------|---------------|-----------------|
| Logits-tokens misalignment | **CRITICAL** | ❌ Wrong alignment in compute_logprobs | Use targets or shift manually |
| Mask naming confusion | High | ❌ "response_mask" is ambiguous | Rename or use targets-derived mask |
| Targets unused | Medium | ❌ Created but never used | Use targets in loss + collate |
| Missing training_mask | High | ❌ Using response_mask incorrectly | Derive from targets: `(targets != IGNORE).float()` |

---

## Part 7: Testing Plan

1. **Run updated test script:**
   ```bash
   python debug/test_create_next_token_targets.py
   ```

2. **Verify mask alignment:**
   - Check that training_mask[i] = 1.0 when targets[i] != IGNORE
   - Check that positions at EOS have training_mask = 0.0
   - Check that positions before EOS can have training_mask = 1.0 (to predict EOS)

3. **Integration test:**
   - Run a short training job
   - Print logprobs and verify they're reasonable (not NaN, not too negative)
   - Check that loss decreases over iterations

4. **Gradient flow test:**
   - Add hooks to model to track which positions get gradients
   - Verify only training_mask=1.0 positions get gradients

---

## Conclusion

The root cause is **conceptual confusion between "response tokens" (what was generated) and "trainable positions" (where to compute loss)**. Due to next-token prediction's inherent shift, these are offset by 1.

**The fix:** Use `targets` (which already encodes the shift) throughout the pipeline, and derive `training_mask` from it. This makes the code clearer and more correct.
