# Refactoring Proposal 05: Streamline Training Loop and Collate Function

## Overview
Building on Proposals 01-04, this iteration focuses on the training loop and data collation. We align the collate function more closely with grpo/main.py while keeping the improvements from blackjack (loss_mask instead of padding_mask).

## Key Changes

### 1. Simplify Collate Function
Current implementation (lines 1163-1211) is more complex than needed.

**Before:**
```python
def collate(
    batches: list[list[Episode]],
    pad_id: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Collates a list of batches (groups) into inputs and targets."""
    inputs = []
    targets = []

    for batch in batches:
        # Stack all tensors (pad to max length in batch)
        all_tokens = [e.all_token_ids for e in batch]
        all_tokens = torch.nn.utils.rnn.pad_sequence(
            all_tokens, batch_first=True, padding_value=pad_id
        )

        loss_masks = [e.loss_mask for e in batch]
        loss_masks = torch.nn.utils.rnn.pad_sequence(
            loss_masks, batch_first=True, padding_value=0.0
        )

        ref_logprobs = [e.ref_logprobs for e in batch]
        ref_logprobs = torch.nn.utils.rnn.pad_sequence(
            ref_logprobs, batch_first=True, padding_value=0.0
        )

        advantages = torch.tensor([e.advantage for e in batch]).unsqueeze(-1)

        input = {"tokens": all_tokens}
        target = {
            "input_ids": all_tokens,  # For torch.roll in loss
            "loss_mask": loss_masks,
            "ref_logprobs": ref_logprobs,
            "advantages": advantages,
        }

        inputs.append(input)
        targets.append(target)

    return inputs, targets
```

**After:**
```python
def collate(
    batches: list[list[Episode]],
    pad_id: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Collate episode batches into model inputs and targets."""
    inputs, targets = [], []

    for batch in batches:
        # Pad sequences to max length in batch
        tokens = torch.nn.utils.rnn.pad_sequence(
            [e.all_token_ids for e in batch],
            batch_first=True,
            padding_value=pad_id,
        )
        loss_mask = torch.nn.utils.rnn.pad_sequence(
            [e.loss_mask for e in batch],
            batch_first=True,
            padding_value=0.0,
        )
        ref_logprobs = torch.nn.utils.rnn.pad_sequence(
            [e.ref_logprobs for e in batch],
            batch_first=True,
            padding_value=0.0,
        )
        advantages = torch.tensor([e.advantage for e in batch]).unsqueeze(-1)

        inputs.append({"tokens": tokens})
        targets.append({
            "input_ids": tokens,
            "loss_mask": loss_mask,
            "ref_logprobs": ref_logprobs,
            "advantages": advantages,
        })

    return inputs, targets
```

**Rationale:** More concise, single-pass construction of tensors.

### 2. Simplify Continuous Training Loop
The training loop (lines 1875-1920) has unnecessary complexity around tracer restarts.

**Before:**
```python
async def continuous_training():
    training_step = 0
    restart_tracer = True

    while max_steps == -1 or training_step < max_steps:
        if restart_tracer:
            t = Tracer("main_perf/continuous_training")
            t.start()
            restart_tracer = False

        batch = await replay_buffer.sample.call_one(curr_policy_version=training_step)
        if batch is None:
            if training_step > 2 and training_step % 5 == 0:
                print(f"[TRAINING] Step {training_step}: Waiting for buffer...")
            await asyncio.sleep(1.0)
        else:
            t.step("waiting_for_buffer")
            print(f"[TRAINING] Step {training_step}: Starting training")

            inputs, targets = batch
            await trainer.train_step.call(inputs, targets)
            training_step += 1
            t.step("train_step")

            await trainer.push_weights.call(training_step)
            t.step("push_weights")

            await policy.update_weights.fanout(training_step)
            t.step("update_weights")

            if training_step >= 2:
                await drop_weights(training_step - 1)
                t.step("drop_weights")

            t.stop()
            restart_tracer = True

            await mlogger.flush.call_one(training_step)
```

**After:**
```python
async def continuous_training():
    training_step = 0

    while max_steps == -1 or training_step < max_steps:
        t = Tracer("main_perf/continuous_training")
        t.start()

        # Wait for buffer
        batch = await replay_buffer.sample.call_one(curr_policy_version=training_step)
        if batch is None:
            await asyncio.sleep(0.5)
            t.stop()
            continue
        t.step("waiting_for_buffer")

        # Train
        inputs, targets = batch
        await trainer.train_step.call(inputs, targets)
        training_step += 1
        t.step("train_step")

        # Update policy
        await trainer.push_weights.call(training_step)
        await policy.update_weights.fanout(training_step)
        t.step("update_weights")

        # Clean up old weights
        if training_step >= 2:
            await drop_weights(training_step - 1)

        t.stop()
        await mlogger.flush.call_one(training_step)

    print(f"Training complete: {max_steps} steps")
```

**Rationale:** Simpler control flow, no restart_tracer flag needed. Use continue for early exit.

### 3. Remove Conditional Logging in Training Loop
The conditional print (line 1891-1894) is noise.

**Before:**
```python
if training_step > 2 and training_step % 5 == 0:
    print(f"[TRAINING] Step {training_step}: Waiting for buffer...")
```

**After:** (removed - metrics already track this)

### 4. Simplify Reference Model Call in Rollout
The padding logic (lines 1795-1820) can be more concise.

**Before:**
```python
# ============ Step 4: Compute ref_model ============
max_len = max(len(e.all_token_ids) for e in episodes)

# Pad input_ids and loss_masks
padded_input_ids = []
padded_loss_masks = []

for i, e in enumerate(episodes):
    seq_len = len(e.all_token_ids)
    pad_len = max_len - seq_len

    # Pad tokens
    padded_tokens = F.pad(e.all_token_ids, (0, pad_len), value=pad_id)
    padded_input_ids.append(padded_tokens)

    # Pad loss_mask
    padded_mask = F.pad(e.loss_mask, (0, pad_len), value=0.0)
    padded_loss_masks.append(padded_mask)

input_ids = torch.stack(padded_input_ids)
loss_mask_batch = torch.stack(padded_loss_masks)

# Call ref_model
ref_logprobs_padded = await ref_model.forward.route(
    input_ids, return_logprobs=True, loss_mask=loss_mask_batch
)

# Unpad and assign
for i, episode in enumerate(episodes):
    seq_len = len(episode.all_token_ids)
    episode.ref_logprobs = ref_logprobs_padded[i, :seq_len]
```

**After:**
```python
# Compute reference logprobs (pad to batch max length)
input_ids = torch.nn.utils.rnn.pad_sequence(
    [e.all_token_ids for e in episodes],
    batch_first=True,
    padding_value=pad_id,
)
loss_mask = torch.nn.utils.rnn.pad_sequence(
    [e.loss_mask for e in episodes],
    batch_first=True,
    padding_value=0.0,
)

ref_logprobs_padded = await ref_model.forward.route(
    input_ids, return_logprobs=True, loss_mask=loss_mask
)

# Assign unpadded logprobs to episodes
for i, ep in enumerate(episodes):
    ep.ref_logprobs = ref_logprobs_padded[i, : len(ep.all_token_ids)]
```

**Rationale:** Use same padding utility as collate function. More concise.

## Impact
- **Collate function:** 49 lines → 32 lines
- **Training loop:** More readable, simpler control flow
- **Ref model call:** Cleaner, reuses utilities
- **Code size:** Additional ~40 lines removed
- **Risk:** Low - mostly simplification, no logic changes
