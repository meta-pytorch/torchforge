# response_mask vs loss_mask: Final Design (torch.roll approach)

Based on exploration of VERL, TRL, Prime-RL, and first-principles analysis.

---

## TL;DR: The Final Design

**No frameworks keep `targets` - it's pointless! Just `torch.roll(input_ids, -1)` at loss time.**

### Episode Fields:
```python
@dataclass
class Episode:
    all_token_ids: torch.Tensor  # [seq_len] - All conversation tokens
    response_mask: torch.Tensor  # [seq_len] bool - Which tokens ARE responses
    loss_mask: torch.Tensor      # [seq_len] float - Which POSITIONS contribute to loss (0.0/1.0)
    reward: float
    # ... other fields ...
```

### Key Insight:
- `response_mask[i] = True` means token i IS a response token
- `loss_mask[i] = 1.0` means position i contributes to loss (predicts token i+1)
- **loss_mask is just response_mask shifted by 1!**

---

## Part 1: loss_mask = response_mask Shifted by 1

### Simple Truth

```python
# In do_single_rollout:
loss_mask_tensor = torch.roll(response_mask_tensor, shifts=-1, dims=0).float()
loss_mask_tensor[-1] = 0.0  # Last position should not train
```

**That's it!** No need for complex `finalize()` logic.

### Why the EOS check is redundant

You might think: "What if position i is EOS but position i+1 is a response?"

**This can't happen in your code!** Because:
1. `add_assistant_response` only succeeds if response ends with EOS
2. After EOS, next message is ALWAYS user (response_mask=False) or end of sequence
3. So: `tokens[i] == EOS` → `response_mask[i+1] == False` (always!)

**Therefore:** The EOS check in `finalize()` is redundant. Simple shift is sufficient.

---

## Part 2: Utility Function for Target Creation

Since we create targets in multiple places (loss function, ref model), use a utility:

```python
def create_shifted_targets(
    input_ids: torch.Tensor,
    loss_mask: torch.Tensor | None = None,
    ignore_index: int = CROSS_ENTROPY_IGNORE_IDX,
) -> torch.Tensor:
    """
    Create next-token prediction targets using torch.roll.
    Maintains same shape as input_ids.

    Args:
        input_ids: [batch, seq_len] or [seq_len] - Input token IDs
        loss_mask: [batch, seq_len] or [seq_len] - Trainable positions (bool or float)
                   If None, all positions are trainable
        ignore_index: Value for masked positions (default: -100)

    Returns:
        targets: Same shape as input_ids
                 targets[i] = input_ids[i+1] where trainable, else ignore_index
    """
    # If no loss_mask provided, all positions trainable
    if loss_mask is None:
        loss_mask = torch.ones_like(input_ids, dtype=torch.float)

    if input_ids.dim() == 1:
        # 1D case
        targets = torch.roll(input_ids, shifts=-1, dims=0)
        targets[-1] = ignore_index  # Last position wraps, mask it

        # Apply loss_mask
        targets = torch.where(
            loss_mask.bool(),
            targets,
            torch.full_like(targets, ignore_index)
        )
    else:
        # 2D case (batched)
        targets = torch.roll(input_ids, shifts=-1, dims=-1)
        targets[:, -1] = ignore_index  # Last position wraps, mask it

        # Apply loss_mask
        targets = torch.where(
            loss_mask.bool(),
            targets,
            torch.full_like(targets, ignore_index)
        )

    return targets
```

**Key benefit:** Positions with `target=ignore_index` get **automatic 0.0 logprob** from cross_entropy, no need to multiply by mask afterward!

---

## Part 3: Update compute_logprobs

Update `compute_logprobs` to take `targets` instead of `input_ids` and remove `align` parameter:

```python
# In src/forge/util/ops.py

def compute_logprobs(
    logits: torch.Tensor,
    targets: torch.Tensor,
    temperature: float = 1.0,
    ignore_index: int = CROSS_ENTROPY_IGNORE_IDX,
) -> torch.Tensor:
    """
    Computes the log probabilities of target tokens given the model logits.

    Args:
        logits: Model logits [batch, seq_len, vocab]
        targets: Target token IDs [batch, seq_len]
        temperature: Temperature for scaling
        ignore_index: Positions with this value in targets are masked (get 0.0 logprob)

    Returns:
        logprobs: [batch, seq_len] - Positions with ignore_index automatically get 0.0
    """
    scaled_logits = logits / temperature
    scaled_logits_fp32 = scaled_logits.float()

    batch_size, seq_len, vocab_size = scaled_logits_fp32.shape
    logprobs = -F.cross_entropy(
        scaled_logits_fp32.reshape(-1, vocab_size),
        targets.reshape(-1).long(),
        reduction="none",
        ignore_index=ignore_index,
    )

    return logprobs.reshape(batch_size, seq_len)
```

---

## Part 4: Loss Function with torch.roll

### Updated simple_grpo_loss:

```python
def simple_grpo_loss(
    logits: torch.Tensor,      # [b, seq_len, vocab]
    input_ids: torch.Tensor,   # [b, seq_len]
    loss_mask: torch.Tensor,   # [b, seq_len] - 0.0/1.0 float
    ref_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    beta: float = 0.1,
) -> torch.Tensor:
    """
    GRPO loss with proper next-token prediction using torch.roll.

    Per-sequence normalization: Each sequence's loss is averaged by its own
    trainable token count, then averaged across the batch.

    Args:
        logits: Model logits [b, seq_len, vocab]
        input_ids: Input token IDs [b, seq_len]
        loss_mask: Loss mask [b, seq_len] - 1.0 for trainable positions
        ref_logprobs: Reference logprobs [b, seq_len]
        advantages: Advantages [b, 1]
        beta: KL penalty
    """
    # Create targets using utility function
    targets = create_shifted_targets(input_ids, loss_mask)  # [b, seq_len]

    # Compute policy logprobs (ignore_index automatically zeros masked positions)
    logprobs = compute_logprobs(
        logits,
        targets,
        ignore_index=CROSS_ENTROPY_IGNORE_IDX
    )  # [b, seq_len] - masked positions already 0.0!

    # Note: ref_logprobs were computed the same way, so also have 0.0 at masked positions

    # KL divergence (masked positions are 0.0, so they don't contribute)
    kl = torch.exp(ref_logprobs - logprobs) - (ref_logprobs - logprobs) - 1

    # Policy loss
    per_token_policy_loss = torch.exp(logprobs - logprobs.detach()) * advantages
    per_token_loss = -(per_token_policy_loss - beta * kl)  # [b, seq_len]

    # Per-sequence normalization, then batch average
    # .sum(dim=1) creates [b] where each element is sum for ONE sequence
    # Each sequence averaged by its own trainable count
    loss = (
        (per_token_loss * loss_mask).sum(dim=1) / loss_mask.sum(dim=1).clamp(min=1.0)
    ).mean()  # [b] → scalar

    return loss
```

**Important:** The loss computation IS per-sequence!
```python
per_token_loss = [batch, seq_len]  # e.g., [8, 100]

(per_token_loss * loss_mask).sum(dim=1)  # → [8] (one value per sequence)
loss_mask.sum(dim=1)                      # → [8] (trainable count per sequence)
division                                  # → [8] (average loss per sequence)
.mean()                                   # → scalar (average across batch)
```

Each sequence contributes equally, regardless of length!

---

## Part 5: Reference Model with torch.roll

### Updated ReferenceModel.forward:

```python
# In src/forge/actors/reference_model.py

@endpoint
async def forward(
    self,
    input_ids: torch.Tensor,       # [b, seq_len]
    return_logprobs: bool,
    loss_mask: torch.Tensor = None, # [b, seq_len] optional
) -> torch.Tensor:
    """
    Args:
        input_ids: Input token ids
        return_logprobs: Whether to return logprobs
        loss_mask: Optional mask for which positions to compute logprobs
    """
    # Record metrics
    record_metric("reference_perf/forward/count_forward_passes", 1, Reduce.SUM)
    record_metric("reference_perf/forward/avg_sequence_length", input_ids.shape[1], Reduce.MEAN)

    t = Tracer("reference_perf/forward", timer="gpu", track_memory=True)
    t.start()
    self.engine.gc_handler.run(self.step)
    t.step("garbage_collection")

    # Forward pass
    model_parts = self.engine.model_parts
    parallel_dims = self.engine.parallel_dims
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
        # Create targets using utility function (loss_mask=None means all trainable)
        targets = create_shifted_targets(input_ids, loss_mask)

        # Compute logprobs using updated compute_logprobs
        logprobs = compute_logprobs(
            logits,
            targets,
            ignore_index=CROSS_ENTROPY_IGNORE_IDX
        )

        t.step("compute_logprobs")
        t.stop()
        return logprobs
```

---

## Part 6: Update Episode and Collate

### Episode Dataclass (UNCHANGED):

```python
@dataclass
class Episode:
    """Episode data for GRPO training."""

    episode_id: str
    all_token_ids: torch.Tensor   # [seq_len] - All conversation tokens
    response_mask: torch.Tensor   # [seq_len] bool - Which tokens ARE responses
    loss_mask: torch.Tensor       # [seq_len] float - Which POSITIONS train (0.0/1.0)
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

### Collate Function (use loss_mask):

```python
def collate(
    batches: list[list[Episode]],
    pad_id: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    inputs = []
    targets_list = []

    for batch in batches:
        # Stack tokens
        all_tokens = [e.all_token_ids for e in batch]
        all_tokens = torch.nn.utils.rnn.pad_sequence(
            all_tokens, batch_first=True, padding_value=pad_id
        )

        # Stack loss_mask
        loss_masks = [e.loss_mask for e in batch]
        loss_masks = torch.nn.utils.rnn.pad_sequence(
            loss_masks, batch_first=True, padding_value=0.0
        )

        # Stack ref_logprobs
        ref_logprobs = [e.ref_logprobs for e in batch]
        ref_logprobs = torch.nn.utils.rnn.pad_sequence(
            ref_logprobs, batch_first=True, padding_value=0.0
        )

        advantages = torch.tensor([e.advantage for e in batch]).unsqueeze(-1)

        # Create input and target dicts
        input = {"tokens": all_tokens}
        target = {
            "input_ids": all_tokens,      # For torch.roll in loss
            "loss_mask": loss_masks,       # Trainable positions
            "ref_logprobs": ref_logprobs,
            "advantages": advantages,
        }

        inputs.append(input)
        targets_list.append(target)

    return inputs, targets_list
```

---

## Part 7: Changes to do_single_rollout

### REMOVE create_next_token_targets, ADD simple shift:

```python
async def do_single_rollout(
    env: BlackjackEnv,
    policy,
    tokenizer,
    max_seq_len: int,
    max_turns: int,
    messages: list[dict],
    game_id: str | None = None,
) -> Episode:
    # ... existing rollout logic ...

    # At the end, convert to tensors:
    all_tokens_tensor = torch.tensor(
        accumulator.accumulated_tokens, dtype=torch.long
    )
    response_mask_tensor = torch.tensor(
        accumulator.response_mask, dtype=torch.bool
    )

    # CREATE loss_mask by shifting response_mask
    loss_mask_tensor = torch.roll(response_mask_tensor, shifts=-1, dims=0).float()
    loss_mask_tensor[-1] = 0.0  # Last position should not train

    logprobs_tensor = torch.tensor(accumulator.logprobs, dtype=torch.float)

    return Episode(
        episode_id=game_id,
        all_token_ids=all_tokens_tensor,
        response_mask=response_mask_tensor,
        loss_mask=loss_mask_tensor,  # NEW!
        reward=final_reward,
        logprobs=logprobs_tensor,
        ref_logprobs=None,  # Filled in later
        # ... rest of fields
    )
```

**DELETE the create_next_token_targets function entirely!**

---

## Part 8: Update continuous_rollouts

### Pass loss_mask to ref_model:

```python
# In continuous_rollouts, before calling ref_model:

# Pad input_ids and loss_masks to same length
max_len = max(len(e.all_token_ids) for e in episodes)

padded_input_ids = []
padded_loss_masks = []

for e in episodes:
    seq_len = len(e.all_token_ids)
    pad_len = max_len - seq_len

    # Pad tokens
    padded_tokens = F.pad(e.all_token_ids, (0, pad_len), value=pad_id)
    padded_input_ids.append(padded_tokens)

    # Pad loss_mask
    padded_mask = F.pad(e.loss_mask, (0, pad_len), value=0.0)
    padded_loss_masks.append(padded_mask)

input_ids = torch.stack(padded_input_ids)       # [batch, max_len]
loss_mask_batch = torch.stack(padded_loss_masks) # [batch, max_len]

# Call ref_model with loss_mask
ref_logprobs_padded = await ref_model.forward.route(
    input_ids,
    return_logprobs=True,
    loss_mask=loss_mask_batch  # NEW!
)

# Assign ref_logprobs to episodes (unpad to original length)
for i, episode in enumerate(episodes):
    seq_len = len(episode.all_token_ids)
    episode.ref_logprobs = ref_logprobs_padded[i, :seq_len]
```

---

## Part 9: Summary of All Changes

### Files to Edit:

1. **`src/forge/util/ops.py`**:
   - Add `ignore_index` parameter to `compute_logprobs`
   - Add new utility function `create_shifted_targets`

2. **`apps/blackjack/main_v2.py`**:
   - **DELETE** `create_next_token_targets` function (lines 965-994)
   - Update `do_single_rollout`: create loss_mask with simple shift
   - Update `collate()`: pass loss_mask instead of response_mask
   - Update `simple_grpo_loss()`: use `create_shifted_targets`, call `compute_logprobs`
   - Update `continuous_rollouts`: pass loss_mask to ref_model

3. **`src/forge/actors/reference_model.py`**:
   - Update `forward()`: accept loss_mask, use `create_shifted_targets` and `compute_logprobs`

4. **Update assertions** (lines 1331-1357):
   - Simplify to: `assert len(ep.all_token_ids) == len(ep.loss_mask)`

### New utility function location:

Add to **`src/forge/util/ops.py`** (or `src/forge/data/common.py` if you prefer):

```python
def create_shifted_targets(
    input_ids: torch.Tensor,
    loss_mask: torch.Tensor | None = None,
    ignore_index: int = CROSS_ENTROPY_IGNORE_IDX,
) -> torch.Tensor:
    """Create next-token prediction targets using torch.roll."""
    # If no loss_mask provided, all positions trainable
    if loss_mask is None:
        loss_mask = torch.ones_like(input_ids, dtype=torch.float)

    # ... (see Part 2 above)
```

---

## Part 10: Why This Design is Better

### Comparison:

| Aspect | Old Design | New Design |
|--------|-----------|------------|
| **Episode fields** | `targets` (redundant!) | No targets, just `loss_mask` |
| **loss_mask creation** | Complex finalize() logic | Simple shift: `torch.roll(mask, -1)` |
| **Shape changes** | Slicing changes shapes | torch.roll maintains shape |
| **Mask semantics** | Confusing response_mask | Clear loss_mask (shifted) |
| **Utility reuse** | Inline everywhere | `create_shifted_targets()` utility |
| **Auto-masking** | Manual `* loss_mask` | ignore_index auto-zeros |
| **compute_logprobs** | Takes input_ids with align | Takes targets, no align |

### Benefits:

1. **No redundant data**: Don't store targets, create on-the-fly
2. **Constant shapes**: All tensors stay [seq_len] throughout
3. **Simple loss_mask**: Just shift response_mask with `torch.roll`, no complex logic
4. **Utility function**: Reuse `create_shifted_targets` everywhere
5. **Auto-masking**: ignore_index makes masked positions 0.0 automatically
6. **Per-sequence normalization**: Each sequence contributes equally to loss
7. **Simplified API**: `compute_logprobs` takes targets directly, no align parameter
8. **Optional loss_mask**: `create_shifted_targets` handles None (all trainable)

---

## Testing Checklist

Run `python debug/test_loss_mask_torch_roll.py` and verify:

1. ✅ torch.roll creates correct targets
2. ✅ loss_mask = response_mask shifted by 1
3. ✅ Truncated responses have loss_mask=0.0 at last position
4. ✅ Shape is maintained ([seq_len] → [seq_len])
5. ✅ Logprobs computation works correctly
6. ✅ Multi-turn example matches expected behavior
7. ✅ Per-sequence normalization in loss
