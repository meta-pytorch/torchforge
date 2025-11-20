# Refactoring Proposal 04: Simplify Rollout Logic and Debug Output

## Overview
Building on Proposals 01-03, this iteration simplifies the rollout loop, removes excessive debug printing, and streamlines episode creation.

## Key Changes

### 1. Remove Verbose Debug Printing from Rollout Loop
Lines 1751-1781 print full episode details every rollout. This is excessive.

**Before:**
```python
# ============ Debug: Print first episode ============
if episodes:
    ep = episodes[0]
    print(f"\n{'='*80}")
    print(f"[ROLLOUT {rollout_count}] Episode 0 Debug Info")
    print(f"{'='*80}")
    print(f"Reward: {ep.reward}, Truncated: {ep.is_truncated}, ...")
    print(f"Total tokens: {len(ep.all_token_ids)}, ...")
    print(f"\n--- Messages ---")
    for i, msg in enumerate(ep.message_log):
        # ... print all messages
    print(f"\n--- Decoded all_token_ids ---")
    decoded_text = tokenizer.decode(ep.all_token_ids.tolist())
    print(decoded_text)
    print(f"{'='*80}\n")
    print(f"\n--- decoded_response_text ---")
    # ... more printing
```

**After:**
```python
# Conditional debug logging
if rollout_count % 100 == 0:  # Only every 100 rollouts
    ep = episodes[0]
    print(f"[ROLLOUT {rollout_count}] Reward: {ep.reward:.2f}, "
          f"Tokens: {len(ep.all_token_ids)}, Truncated: {ep.is_truncated}")
```

**Rationale:** Debug info should be occasional, not every iteration. Add a config flag `debug_rollouts` if needed.

### 2. Simplify Episode Creation in do_single_rollout
The episode creation logic (lines 1046-1071) mixes tensor operations with metadata.

**Before:**
```python
# Create loss_mask by shifting response_mask using torch.roll
loss_mask_tensor = torch.roll(
    episode_data.response_mask, shifts=-1, dims=0
).float()
loss_mask_tensor[-1] = 0.0

return Episode(
    episode_id=game_id,
    task_name="blackjack",
    policy_version=policy_version,
    is_truncated=episode_data.is_truncated,
    all_token_ids=episode_data.token_ids,
    response_mask=episode_data.response_mask,
    loss_mask=loss_mask_tensor,
    reward=final_reward,
    logprobs=episode_data.logprobs,
    message_log=accumulator.messages.copy(),
    metadata={
        "truncation_reason": episode_data.truncation_reason,
        "hit_max_turns": hit_max_turns,
        "num_turns": turn_num,
        "num_trainable_tokens": episode_data.response_mask.sum().item(),
        **(result.metadata if "result" in locals() else {}),
    },
)
```

**After:**
```python
# Create loss_mask (shift response_mask by 1 for next-token prediction)
loss_mask = torch.roll(episode_data.response_mask, shifts=-1, dims=0).float()
loss_mask[-1] = 0.0

return Episode(
    episode_id=game_id,
    all_token_ids=episode_data.token_ids,
    loss_mask=loss_mask,
    reward=final_reward,
    ref_logprobs=None,  # Set later by ref model
    advantage=None,     # Set later by advantage computation
    policy_version=policy_version,
    is_truncated=episode_data.is_truncated,
    message_log=accumulator.messages.copy() if debug_mode else None,
)
```

**Rationale:** Simpler, matches updated Episode dataclass from Proposal 03.

### 3. Remove Redundant Metrics in Rollout
Lines 1037-1044 record per-episode metrics that are rarely useful.

**Before:**
```python
if episode_data.truncation_reason:
    record_metric(
        f"episode/truncated_{episode_data.truncation_reason}",
        1,
        Reduce.SUM,
    )
record_metric("episode/total_tokens", len(episode_data.token_ids), Reduce.MEAN)
record_metric("episode/turns", turn_num, Reduce.MEAN)
```

**After:**
```python
# Aggregate metrics only
record_metric("episode/truncation_rate",
              1 if episode_data.is_truncated else 0,
              Reduce.MEAN)
record_metric("episode/avg_tokens", len(episode_data.token_ids), Reduce.MEAN)
```

### 4. Simplify Sequential Rollout Loop
The comment says "run games SEQUENTIALLY" but the code is unnecessarily verbose (lines 1728-1747).

**Before:**
```python
# ============ Step 1: Create environments ============
# Run games SEQUENTIALLY to avoid race conditions on shared server
# (each thread has its own server, but games within a thread share it)

# ============ Step 2: Rollout group (SEQUENTIALLY) ============
episodes = []
for i in range(group_size):
    env = BlackjackEnv(server_url=server_url)
    game_id = f"game_{i}_{uuid.uuid4().hex[:8]}"

    episode = await do_single_rollout(
        env=env,
        policy=policy,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        max_turns=max_turns,
        messages=initial_messages,
        game_id=game_id,
    )
    episodes.append(episode)

t.step("play_games")
```

**After:**
```python
# Rollout group (sequential to avoid server race conditions)
episodes = [
    await do_single_rollout(
        env=BlackjackEnv(server_url),
        policy=policy,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        max_turns=max_turns,
        messages=initial_messages,
        game_id=f"game_{i}_{uuid.uuid4().hex[:8]}",
    )
    for i in range(group_size)
]
t.step("play_games")
```

**Rationale:** More concise, equally clear.

### 5. Remove Unused result.metadata
Since EnvStepResult.metadata was removed in Proposal 03, clean up references.

**Before:**
```python
metadata={
    ...,
    **(result.metadata if "result" in locals() else {}),
}
```

**After:** (removed)

## Impact
- **Rollout loop:** Much cleaner, less verbose
- **Debug output:** Reduced by 95% (only occasional logging)
- **Code size:** Additional ~100 lines removed
- **Performance:** Slightly better (less string formatting/printing)
- **Risk:** Low - mostly removing debug code
