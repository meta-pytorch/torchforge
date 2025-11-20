# Multi-Turn Training with Masks: Same-Shape Approach

## The Problem

**Old approach (single-turn):**
```python
# Works only for single turn where response starts at fixed position
response = all_tokens[prompt_len:]
```

**New approach (multi-turn):**
```
Conversation: [system] [user] [agent] [tool] [agent] [user] [agent]
Train only on:              ^^^^^^          ^^^^^^          ^^^^^^
```

We need masks to identify which tokens are agent responses across multiple turns.

**Key principle:**
- **Keep everything the same shape `[seq_len]`**
- Use `response_mask` to mark agent tokens
- Use `IGNORE_INDEX` in targets for non-agent positions
- Let PyTorch's cross_entropy handle the masking

---

## Current Bugs

### Bug 1: reference_model.py
```python
# WRONG: Assumes single-turn, response starts at max_req_tokens
logprobs = compute_logprobs(logits, input_ids[:, max_req_tokens:])
```

### Bug 2: main_v2.py continuous_rollouts
```python
# WRONG: Slicing instead of using full-sequence masks
ref_logprobs_padded = await ref_model.forward.route(input_ids, 0, return_logprobs=True)
for i, episode in enumerate(episodes):
    seq_len = len(episode.all_token_ids)
    episode.ref_logprobs = ref_logprobs_padded[i, :seq_len]
```

### Bug 3: main_v2.py simple_grpo_loss
```python
# WRONG: For loop over batch, not tensorized
for i in range(batch_size):
    mask_i = response_mask[i] == 1
    ...
```

---

## Design Principles

1. **Same shape everywhere**: All tensors are `[seq_len]`, pad to `[batch, max_seq_len]` in collate
2. **Use bool masks**: `response_mask` is `dtype=torch.bool` to avoid `== 1` comparisons
3. **IGNORE_INDEX for masking**: Set `targets[i] = IGNORE_INDEX` where position i is not a response
4. **Tensorized operations**: No for loops over batch dimension in loss function

---

## Solution

### Constants

Add to main_v2.py:
```python
IGNORE_INDEX = -100  # PyTorch cross_entropy default
```

---

### 1. Create Targets for Full Sequence

**Add utility function to main_v2.py:**

```python
def create_next_token_targets(
    all_token_ids: torch.Tensor,    # [seq_len]
    response_mask: torch.Tensor,    # [seq_len] bool
) -> torch.Tensor:
    """
    Create next-token prediction targets for full sequence.

    For next-token prediction:
    - logits[:, i] predicts tokens[:, i+1]
    - targets[i] = all_token_ids[i+1] if position i+1 is a response token
    - targets[i] = IGNORE_INDEX otherwise

    Args:
        all_token_ids: All conversation tokens [seq_len]
        response_mask: Boolean mask, True for agent response tokens [seq_len]

    Returns:
        targets: [seq_len] where:
            - targets[i] = all_token_ids[i+1] if response_mask[i+1] is True
            - targets[i] = IGNORE_INDEX otherwise
    """
    targets = torch.full_like(all_token_ids, IGNORE_INDEX)

    # Shift: targets[i] should predict all_token_ids[i+1]
    targets[:-1] = all_token_ids[1:]

    # Mask: Only keep targets where the predicted token is a response
    # If response_mask[i+1] is False, set targets[i] = IGNORE_INDEX
    targets[:-1][~response_mask[1:]] = IGNORE_INDEX
    targets[-1] = IGNORE_INDEX  # Last position has nothing to predict

    return targets
```

---

### 2. Update Episode Dataclass

**main_v2.py - Episode:**

```python
@dataclass
class Episode:
    """Episode data for GRPO training (multi-turn structure)."""

    # Required fields - ALL same shape [seq_len]
    episode_id: str
    all_token_ids: torch.Tensor      # All tokens [seq_len]
    response_mask: torch.Tensor      # Boolean mask: True = agent token [seq_len]
    targets: torch.Tensor            # Next-token targets with IGNORE_INDEX [seq_len]
    reward: float

    # Optional fields
    task_name: str = "blackjack"
    generator_version: int = 0
    is_truncated: bool = False
    logprobs: torch.Tensor | None = None  # vLLM logprobs [seq_len] (optional)
    ref_logprobs: torch.Tensor | None = None  # Ref model logprobs [seq_len]
    advantage: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    message_log: list[dict[str, str]] | None = None
```

**Key changes:**
- `response_mask` is now `torch.bool` dtype
- `targets` is a required field, same shape as `all_token_ids`
- All core tensors are `[seq_len]`

---

### 3. do_single_rollout - Create Episode with Targets

**main_v2.py - do_single_rollout (around line 765):**

Replace the episode creation section:

```python
# ============ Create episode ============
print(f"\n[do_single_rollout] Creating episode {game_id}")

# Convert to tensors
all_tokens_tensor = torch.tensor(accumulator.accumulated_tokens, dtype=torch.long)
response_mask_tensor = torch.tensor(accumulator.response_mask, dtype=torch.bool)  # bool dtype
logprobs_tensor = torch.tensor(accumulator.logprobs, dtype=torch.float)

# Create targets for full sequence
targets_tensor = create_next_token_targets(all_tokens_tensor, response_mask_tensor)

print(f"  Total tokens: {len(all_tokens_tensor)}")
print(f"  Response tokens: {response_mask_tensor.sum().item()}")
print(f"  Response ratio: {response_mask_tensor.float().mean().item():.2%}")

return Episode(
    episode_id=game_id,
    task_name="blackjack",
    generator_version=generator_version,
    is_truncated=accumulator.is_truncated,
    all_token_ids=all_tokens_tensor,       # [seq_len]
    response_mask=response_mask_tensor,    # [seq_len] bool
    targets=targets_tensor,                # [seq_len] with IGNORE_INDEX
    reward=final_reward,
    logprobs=logprobs_tensor,              # [seq_len] from vLLM
    message_log=accumulator.messages.copy(),
    metadata={
        "truncation_reason": (
            accumulator.truncation_reason.value
            if accumulator.truncation_reason
            else None
        ),
        "hit_max_turns": hit_max_turns,
        "num_turns": turn_num,
        "num_response_tokens": response_mask_tensor.sum().item(),
        **(result.metadata if "result" in locals() else {}),
    },
)
```

---

### 4. Update compute_logprobs (No Mask Parameter)

**forge/util/ops.py - Keep existing compute_logprobs, no changes needed**

The existing `compute_logprobs` function works fine. We'll just use it with full sequences.

**In reference_model.py, we'll call it like:**
```python
# Compute logprobs for full sequence
logprobs = compute_logprobs(logits, input_ids, align=False)  # [batch, seq_len]
```

No new function needed! The masking happens via IGNORE_INDEX in targets.

---

### 5. Update ReferenceModel.forward

**forge/actors/reference_model.py - forward endpoint:**

Replace the entire forward method (lines 128-194):

```python
@endpoint
async def forward(
    self,
    input_ids: torch.Tensor,      # [batch, seq_len]
    return_logprobs: bool
) -> torch.Tensor:
    """
    Forward pass through reference model.

    Args:
        input_ids: Input token ids [batch, seq_len]
        return_logprobs: Whether to return log probabilities

    Returns:
        If return_logprobs=False: logits [batch, seq_len, vocab_size]
        If return_logprobs=True: logprobs [batch, seq_len]
    """
    # Record reference model metrics
    record_metric("reference_perf/forward/count_forward_passes", 1, Reduce.SUM)
    record_metric(
        "reference_perf/forward/avg_sequence_length",
        input_ids.shape[1],
        Reduce.MEAN,
    )

    t = Tracer("reference_perf/forward", timer="gpu", track_memory=True)
    t.start()
    self.engine.gc_handler.run(self.step)
    t.step("garbage_collection")

    input_ids = input_ids.to("cuda")
    t.step("to_device")

    optional_context_parallel_ctx = None
    if self.engine.parallel_dims.pp_enabled:
        raise NotImplementedError("PP not implemented yet")
    else:
        with self.engine.train_context(optional_context_parallel_ctx):
            with self.engine.maybe_enable_amp:
                with torch.inference_mode():
                    logits = self.model(input_ids)

    self.step += 1
    if isinstance(logits, DTensor):
        logits = logits.full_tensor()
    t.step("forward")

    if not return_logprobs:
        t.stop()
        return logits
    else:
        # Compute logprobs for full sequence
        # Use align=False since we're passing the same sequence we used for forward
        logprobs = compute_logprobs(logits, input_ids, align=False)

        t.step("compute_logprobs")
        t.stop()
        return logprobs
```

**Changes:**
- Removed `max_req_tokens` parameter (single-turn assumption)
- Removed mask parameter (masking handled via IGNORE_INDEX in targets)
- Returns `[batch, seq_len]` tensor (same shape as input)
- Uses existing `compute_logprobs` function with `align=False`

---

### 6. Update continuous_rollouts

**main_v2.py - continuous_rollouts (lines 1190-1232):**

Replace the ref_model section:

```python
# ============ Step 4: Compute ref_model ============
print(f"\n[continuous_rollouts] Preparing ref_model input")
max_len = max(len(e.all_token_ids) for e in episodes)
print(f"  Max episode length: {max_len}")

# Pad input_ids
padded_input_ids = []

for i, e in enumerate(episodes):
    seq_len = len(e.all_token_ids)
    pad_len = max_len - seq_len

    print(f"  Episode {i}: tokens={seq_len}, response_tokens={e.response_mask.sum().item():.0f}")

    # Pad tokens
    padded_tokens = F.pad(e.all_token_ids, (0, pad_len), value=pad_id)
    padded_input_ids.append(padded_tokens)

input_ids = torch.stack(padded_input_ids)  # [batch, max_len]

print(f"  input_ids shape: {input_ids.shape}")

# Call ref_model - returns [batch, max_len]
ref_logprobs_padded = await ref_model.forward.route(
    input_ids,
    return_logprobs=True
)

t.step("reference_model_calculate_logprobs")

# Assign ref_logprobs to episodes (unpad to original length)
for i, episode in enumerate(episodes):
    seq_len = len(episode.all_token_ids)
    episode.ref_logprobs = ref_logprobs_padded[i, :seq_len]  # [seq_len]
    print(f"  Episode {i} ref_logprobs shape: {episode.ref_logprobs.shape}")

    # Verify shape matches other tensors
    assert episode.ref_logprobs.shape == episode.targets.shape == episode.all_token_ids.shape, \
        f"Shape mismatch in episode {i}"

del ref_logprobs_padded, input_ids
```

**Key changes:**
- Only pad input_ids (no mask needed)
- Call ref_model with just input_ids
- Receive `[batch, max_len]` tensor back
- Unpad to original sequence length for each episode

---

### 7. Update collate

**main_v2.py - collate function (lines 880-948):**

Replace entire function:

```python
def collate(
    batches: list[list[Episode]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Collates a list of batches (groups) into inputs and targets.

    All tensors are padded to max_seq_len within each batch.

    Args:
        batches: List of groups, where each group is a list of Episodes

    Returns:
        (inputs, targets) for training
    """
    inputs = []
    targets_list = []

    for batch in batches:
        # Find max sequence length in this batch
        max_seq_len = max(len(e.all_token_ids) for e in batch)

        pad_id = 0  # For token padding

        # Collect batch data
        all_tokens = []
        response_masks = []
        targets_batch = []
        ref_logprobs_batch = []
        advantages_list = []

        for e in batch:
            seq_len = len(e.all_token_ids)
            pad_len = max_seq_len - seq_len

            # Pad all_token_ids
            padded_tokens = F.pad(
                e.all_token_ids,
                (0, pad_len),
                value=pad_id
            )
            all_tokens.append(padded_tokens)

            # Pad response_mask (False for padding)
            padded_mask = F.pad(
                e.response_mask,
                (0, pad_len),
                value=False
            )
            response_masks.append(padded_mask)

            # Pad targets (IGNORE_INDEX for padding)
            padded_targets = F.pad(
                e.targets,
                (0, pad_len),
                value=IGNORE_INDEX
            )
            targets_batch.append(padded_targets)

            # Pad ref_logprobs (0.0 for padding, but ignored via IGNORE_INDEX)
            padded_ref_logprobs = F.pad(
                e.ref_logprobs,
                (0, pad_len),
                value=0.0
            )
            ref_logprobs_batch.append(padded_ref_logprobs)

            # Advantage is scalar
            advantages_list.append(e.advantage)

        # Stack everything
        all_tokens_tensor = torch.stack(all_tokens)            # [b, max_seq_len]
        response_mask = torch.stack(response_masks)            # [b, max_seq_len]
        targets_tensor = torch.stack(targets_batch)            # [b, max_seq_len]
        ref_logprobs_tensor = torch.stack(ref_logprobs_batch)  # [b, max_seq_len]
        advantages = torch.tensor(advantages_list).unsqueeze(-1)  # [b, 1]

        # Input: full conversation tokens
        input = {"tokens": all_tokens_tensor}

        # Target: all data with same shape [b, max_seq_len]
        target = {
            "targets": targets_tensor,           # [b, max_seq_len]
            "ref_logprobs": ref_logprobs_tensor, # [b, max_seq_len]
            "advantages": advantages,            # [b, 1]
            "response_mask": response_mask,      # [b, max_seq_len] bool (for metrics)
        }

        inputs.append(input)
        targets_list.append(target)

    return inputs, targets_list
```

**Key changes:**
- Everything padded to `max_seq_len` (only one max length)
- `response_mask` padded with `False`
- `targets` padded with `IGNORE_INDEX`
- All tensors have shape `[batch, max_seq_len]`

---

### 8. Update simple_grpo_loss (Tensorized, No For Loops)

**main_v2.py - simple_grpo_loss (lines 951-981):**

Replace entire function:

```python
def simple_grpo_loss(
    logits: torch.Tensor,        # [b, seq_len, v]
    targets: torch.Tensor,       # [b, seq_len]
    ref_logprobs: torch.Tensor,  # [b, seq_len]
    advantages: torch.Tensor,    # [b, 1]
    response_mask: torch.Tensor, # [b, seq_len] bool
    beta: float = 0.1,
) -> torch.Tensor:
    """
    Simple GRPO loss with multi-turn masking (fully tensorized).

    Args:
        logits: Model logits [b, seq_len, vocab_size]
        targets: Next-token targets [b, seq_len] with IGNORE_INDEX for non-response
        ref_logprobs: Reference logprobs [b, seq_len]
        advantages: Advantages [b, 1]
        response_mask: Boolean mask for response positions [b, seq_len]
        beta: KL penalty coefficient

    Returns:
        Loss scalar
    """
    batch_size, seq_len, vocab_size = logits.shape

    # Shift for next-token prediction
    # logits[:, i] predicts tokens[:, i+1]
    shifted_logits = logits[:, :-1, :]      # [b, seq_len-1, vocab]
    shifted_targets = targets[:, 1:]         # [b, seq_len-1]
    shifted_ref_logprobs = ref_logprobs[:, 1:]  # [b, seq_len-1]

    # Compute policy logprobs (IGNORE_INDEX positions are automatically masked)
    logprobs = -F.cross_entropy(
        shifted_logits.reshape(-1, vocab_size),
        shifted_targets.reshape(-1).long(),
        reduction="none",
        ignore_index=IGNORE_INDEX,
    ).reshape(batch_size, seq_len - 1)

    # Create mask from targets (True where we have valid targets)
    mask = (shifted_targets != IGNORE_INDEX).float()  # [b, seq_len-1]

    # KL divergence (only computed where mask is True, but safe to compute everywhere)
    kl = torch.exp(shifted_ref_logprobs - logprobs) - (shifted_ref_logprobs - logprobs) - 1

    # Policy loss
    per_token_policy_loss = torch.exp(logprobs - logprobs.detach()) * advantages
    per_token_loss = -(per_token_policy_loss - beta * kl)

    # Masked average (fully tensorized)
    loss = (per_token_loss * mask).sum() / mask.sum().clamp(min=1.0)

    return loss
```

**Key changes:**
- **Fully tensorized**: No for loops over batch dimension
- Shift all tensors for next-token prediction
- Use `IGNORE_INDEX` for automatic masking in cross_entropy
- Create mask from targets for KL and policy loss
- Single global average (not per-sample)

---

## Summary of All Changes

| File | Function/Class | Change |
|------|----------------|--------|
| `main_v2.py` | Constants | Add `IGNORE_INDEX = -100` |
| `main_v2.py` | NEW | Add `create_next_token_targets()` |
| `main_v2.py` | Episode | `response_mask` is bool, `targets` is required, all `[seq_len]` |
| `main_v2.py` | do_single_rollout | Create targets, use bool mask |
| `main_v2.py` | continuous_rollouts | Remove mask parameter to ref_model |
| `main_v2.py` | collate | Pad everything to max_seq_len |
| `main_v2.py` | simple_grpo_loss | Fully tensorized, shift tensors, use IGNORE_INDEX |
| `ops.py` | - | No changes needed |
| `reference_model.py` | forward | Remove max_req_tokens, return full sequence |

---

## Shape Flow Example

**Episode creation:**
```
all_token_ids:   [250]  (system + user1 + agent1 + user2 + agent2)
response_mask:   [250]  (bool: True for agent tokens)
targets:         [250]  (shifted, with IGNORE_INDEX for non-agent)
ref_logprobs:    [250]  (computed later, full sequence)
```

**In collate (batch of 4 episodes):**
```
max_seq_len = 250

Input:
  tokens:         [4, 250]

Target:
  targets:        [4, 250]  (with IGNORE_INDEX)
  ref_logprobs:   [4, 250]  (0.0 for non-response, ignored via IGNORE_INDEX)
  advantages:     [4, 1]
  response_mask:  [4, 250]  (bool, for metrics/debugging)
```

**In loss:**
```
logits:          [4, 250, vocab_size]  (from model)
Shift:
  shifted_logits: [4, 249, vocab_size]
  shifted_targets: [4, 249]

Compute loss only where shifted_targets != IGNORE_INDEX
```

---

## Testing

1. **Shape assertions:**
```python
# After episode creation
assert episode.all_token_ids.shape == episode.response_mask.shape == episode.targets.shape
assert episode.response_mask.dtype == torch.bool

# After ref_model
assert episode.ref_logprobs.shape == episode.all_token_ids.shape

# After collate
assert targets.shape == ref_logprobs.shape == (batch_size, max_seq_len)
```

2. **Value checks:**
```python
# Targets should have IGNORE_INDEX for non-response positions
# For response positions: targets[i] = all_token_ids[i+1]
response_positions = torch.where(response_mask)[0]
for pos in response_positions[:-1]:  # Exclude last position
    if pos + 1 < len(all_token_ids) and response_mask[pos + 1]:
        # Next token is also a response, should not be IGNORE_INDEX
        assert targets[pos] != IGNORE_INDEX
```

---

## Breaking Changes

**ref_model.forward API:**

**Before:**
```python
ref_logprobs = await ref_model.forward.route(
    input_ids, max_req_tokens=0, return_logprobs=True
)  # Returns: [batch, variable_response_len]
```

**After:**
```python
ref_logprobs = await ref_model.forward.route(
    input_ids, return_logprobs=True
)  # Returns: [batch, seq_len] (full sequence)
```

All callers of ref_model must be updated.
