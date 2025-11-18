# Truncation Handling - Design Decisions for Blackjack (Updated)

**Date:** 2025-01-16
**Last Updated:** 2025-01-16 (simplified based on user feedback)
**Context:** Multi-turn blackjack with tool calling - design decisions based on library investigation

---

## Design Questions & Decisions

### Q1: How to Detect Truncation?

**Question:** How do we know if vLLM truncated the response due to `max_tokens`?

**Options:**

**A) Check if last token is EOS/PAD (TRL approach)**
```python
eos_and_pad = [tokenizer.eos_token_id, tokenizer.pad_token_id]
is_truncated = response.token_ids[-1] not in eos_and_pad
```

**B) Check vLLM's `stop_reason` field**
```python
is_truncated = response.stop_reason == "length"
```

**C) Track cumulative token budget and flag when exceeded**
```python
cumulative_tokens = len(all_tokens) + len(response.token_ids)
is_truncated = cumulative_tokens >= max_seq_len
```

**Decision: Use B (stop_reason) as primary, with C as additional check**

**Reasoning:**
- `stop_reason == "length"` is explicit and reliable
- Avoids edge cases where model generates EOS but was still truncated
- Additional budget check (C) catches cases where prompt itself is too long
- **Implementation:**
  ```python
  # After generation
  if response.stop_reason == "length":
      is_truncated = True
      truncation_reason = "generation_length"

  # Also check cumulative budget
  if len(all_tokens) >= max_seq_len:
      is_truncated = True
      truncation_reason = "max_seq_len"
  ```

---

### Q2: What to Do with Truncated Generations?

**Question:** When a generation is truncated, should we drop it or mask it?

**Options:**

**A) Drop the truncated turn entirely (Tinker approach)**
- Remove the partial response from the trajectory
- Episode continues with previous turns intact
- Pros: Clean, no masking confusion
- Cons: Lose partial information

**B) Keep partial response but mask it (TRL/Verifiers approach)**
- Include partial tokens in batch
- Set `completion_mask = 0` for truncated turn
- Pros: Debugging visibility, no data loss
- Cons: Philosophically weird (rewarded but not trained)

**Decision: Use A (drop) by default, with B (mask) as config option**

**Reasoning:**
- Tinker's approach is cleanest for multi-turn
- For blackjack: if model says "HIT" but next turn truncates, we keep the "HIT" turn
- We only drop the INCOMPLETE turn
- **Libraries only use drop or mask - no one trains with gradient on truncated tokens**

**Implementation:**
```python
# In play_game()
if response.stop_reason == "length":
    if cfg.truncation.drop_truncated_generation:
        # Don't add this turn to all_tokens/response_mask
        # Episode ends here with previous turns intact
        is_truncated = True
        break
    else:
        # Add partial tokens but mask them
        all_tokens.extend(response.token_ids)
        response_mask.extend([0] * len(response.token_ids))  # Mask out
        is_truncated = True
        break
```

**Config:**
```yaml
truncation:
  drop_truncated_generation: true  # Drop incomplete turn (Tinker approach)
  # If false, masks it instead (TRL approach)
```

---

### Q3: What to Do with Truncated Episodes?

**Question:** When an episode is truncated (hit max_seq_len or max_turns), should we train on it?

**Decision: Filter at GRPO loop level with acceptance criteria (not in replay buffer)**

**Reasoning:**
- Check acceptance BEFORE calling `replay_buffer.add()` to minimize communication
- Acceptance logic stays in GRPO loop, not buried in buffer
- Cleaner separation of concerns

**Implementation:**
```python
# In continuous_rollouts() - NO FILTERING before ref_model
episodes = [await play_game(...) for _ in range(group_size)]

# Compute ref_model for ALL episodes
ref_logprobs = await ref_model.forward.route(episodes)

# Compute advantages for ALL episodes
advantages = await compute_advantages.compute.call_one(episodes)

# Check acceptance BEFORE adding to buffer (minimize communication)
accepted_episodes = []
for episode, advantage in zip(episodes, advantages):
    episode.advantage = advantage

    # Acceptance criteria (inline, not in replay buffer)
    should_accept = True
    if episode.is_truncated and not cfg.grpo.accept_truncated:
        should_accept = False
        record_metric("buffer/rate_rejected_truncated", 1, Reduce.MEAN)
    else:
        record_metric("buffer/rate_rejected_truncated", 0, Reduce.MEAN)
    # Future: Add min_advantage filter here if needed

    if should_accept:
        accepted_episodes.append(episode)

# TODO: Add all episodes at once instead of one by one
for episode in accepted_episodes:
    await replay_buffer.add.call_one(episode)
```

**Config:**
```yaml
grpo:
  accept_truncated: true  # Accept truncated episodes (learn from partial success)
  # Future: min_advantage, etc.
```

---

### Q4: Group-Level Filtering?

**Question:** Should we filter groups before computing advantages?

**Decision: Drop groups with constant rewards only - keep it simple**

**Reasoning:**
- If all rewards are identical: std=0, advantages=0/0=NaN → no learning signal
- Simple check: `if len(set(rewards)) == 1: drop group`
- Don't complicate with truncation logic - let acceptance criteria handle that per-episode

**Implementation:**
```python
# In continuous_rollouts()
# Generate groups (each group is exactly group_size episodes)
all_groups = []
for group_idx in range(num_groups):
    group = [await play_game(...) for _ in range(group_size)]
    all_groups.append(group)

# Filter: Drop groups with constant rewards (no variance = no learning signal)
valid_groups = []
for group in all_groups:
    rewards = [e.reward for e in group]
    if len(set(rewards)) > 1:  # At least 2 different reward values
        valid_groups.append(group)
        record_metric("groups/rate_dropped", 0, Reduce.MEAN)  # Not dropped
    else:
        record_metric("groups/rate_dropped", 1, Reduce.MEAN)  # Dropped

if not valid_groups:
    continue  # Skip this rollout

# Compute ref_model and advantages for valid groups
# (Groups remain size group_size throughout)
```

---

### Q5: When to Compute Reference Model?

**Question:** Should we compute ref_logprobs before or after filtering?

**Decision: After group filtering, before episode-level acceptance**

**Reasoning:**
- Filter groups first (constant rewards) to save computation
- Then compute ref_model for all episodes in valid groups
- Episode-level acceptance happens after advantages are computed

**Implementation:**
```python
# 1. Generate all groups
all_groups = [...]

# 2. Filter groups FIRST (constant rewards)
valid_groups = [g for g in all_groups if len(set([e.reward for e in g])) > 1]

# 3. Compute ref_model for all episodes in valid groups
all_valid_episodes = [e for g in valid_groups for e in g]
ref_logprobs = await ref_model.forward.route(all_valid_episodes)

# 4. Compute advantages per group
for group in valid_groups:
    advantages = compute_group_advantages(group)

# 5. Episode-level acceptance (truncated, min_advantage, etc.)
for episode in all_valid_episodes:
    if should_accept(episode):
        await replay_buffer.add.call_one(episode)
```

---

### Q6: Fixed vs Variable Group Sizes?

**Question:** Should we maintain fixed group sizes or allow variable sizes?

**Decision: Fixed until advantages, then dissolve**

**Reasoning:**
- "if a group is size 16, it will stay 16 until its advantages are computed. After that, the concept of group is useless."
- Simplifies advantage computation (no need to handle variable sizes)
- Training doesn't need groups anyway (packed dataset handles variable lengths)

**Implementation:**
```python
# Groups stay exactly group_size until advantages computed
group_size = cfg.grpo.group_size  # e.g., 16

# Generate groups (FIXED SIZE)
all_groups = [[await play_game(...) for _ in range(group_size)] for _ in range(num_groups)]

# Filter groups (maintains FIXED SIZE per group)
valid_groups = [g for g in all_groups if len(set([e.reward for e in g])) > 1]

# Compute ref_model (groups still FIXED SIZE)
# Compute advantages (groups still FIXED SIZE)

# NOW groups dissolve - pass individual episodes to acceptance check
for group in valid_groups:
    for episode in group:
        if should_accept(episode):
            await replay_buffer.add.call_one(episode)
```

---

### Q7: Truncate Tool Results or Drop Entire Turn?

**Question:** When tool result exceeds budget, should we truncate it or drop the turn?

**Decision: Truncate to budget by default, drop as config option**

**Reasoning:**
- Per-tool limits are environment's responsibility, not config
- We only care about overall `max_seq_len` budget
- Similar to `drop_truncated_generation` but for tool results

**Implementation:**
```python
# In play_game() - when processing tool results
tool_result = await execute_tool(tool_call)

# Tokenize to check length
tool_result_tokens = tokenizer.encode(tool_result, add_special_tokens=False)

# Check if it fits in remaining budget
remaining = max_seq_len - len(all_tokens)

if len(tool_result_tokens) > remaining:
    if cfg.truncation.drop_truncated_tool_response:
        # Drop the turn entirely (Tinker approach)
        is_truncated = True
        truncation_reason = "tool_response_too_long"
        break
    else:
        # Truncate to fit (default)
        tool_result_tokens = tool_result_tokens[:remaining]
        tool_result = tokenizer.decode(tool_result_tokens)
        record_metric("truncation/rate_tool_response_truncated", 1, Reduce.MEAN)

# Add tool response to messages
messages.append({"role": "tool", "content": tool_result})
```

**Config:**
```yaml
truncation:
  drop_truncated_generation: true       # Drop incomplete LLM generation
  drop_truncated_tool_response: false   # Truncate tool response by default (don't drop)
```

---

### Q8: Where to Check Budget - Before or After Generation?

**Question:** Should we check budget before generating (to prevent partial tokens) or after (to detect truncation)?

**Decision: Check BEFORE entering while loop, then rely on `stop_reason` during loop**

**Reasoning:**
- Initial prompt might already exceed budget - check before ANY generation
- Inside loop: `remaining` will always be >= 0 after first check
- Use `stop_reason == "length"` to detect truncation during loop
- Simpler than checking before every generation

**Tinker's pattern (for reference):**
```python
# tinker-cookbook/tinker_cookbook/rl/rollouts.py
async def do_single_rollout(policy: TokenCompleter, env: Env) -> Trajectory:
    """Simple rollout loop - one episode"""
    transitions = []
    ob, stop_condition = await env.initial_observation()

    while True:
        ac_with_logprobs = await policy(ob, stop_condition)
        step_result = await env.step(ac_with_logprobs.tokens)
        transition = Transition(
            ob=ob,
            ac=ac_with_logprobs,
            reward=step_result.reward,
            episode_done=step_result.episode_done,
            metrics=step_result.metrics,
        )
        transitions.append(transition)

        if step_result.episode_done:  # Env decides when to stop
            break

        ob = step_result.next_observation
        stop_condition = step_result.next_stop_condition

    return Trajectory(transitions=transitions, final_ob=ob)

# And the outer function:
async def do_group_rollout(env_group_builder, policy) -> TrajectoryGroup:
    """Rollout a group of episodes in parallel"""
    envs = await env_group_builder.make_envs()
    trajectories = await asyncio.gather(*[
        do_single_rollout(policy, env) for env in envs
    ])
    # ... compute rewards ...
    return TrajectoryGroup(trajectories, rewards, metrics)
```

**Our implementation:**
```python
async def play_single_game(
    game_id: str,
    server_url: str,
    policy: Generator,
    tokenizer,
    max_seq_len: int,
    max_turns: int,
) -> Episode:
    """Play one game - returns single episode"""
    messages = [{"role": "system", "content": "..."}]
    all_tokens = []
    all_logprobs = []
    response_mask = []
    is_truncated = False

    env = OpenSpielEnv(base_url=server_url)
    result = env.reset()

    # Initial prompt check (BEFORE while loop)
    initial_prompt = tokenizer.apply_chat_template(messages, ...)
    initial_tokens = tokenizer.encode(initial_prompt, add_special_tokens=False)

    if len(initial_tokens) >= max_seq_len:
        # Initial prompt too large - return truncated episode immediately
        return Episode(
            is_truncated=True,
            truncation_reason="initial_prompt_exceeds_budget",
            all_token_ids=torch.tensor(initial_tokens[:max_seq_len]),
            # ... minimal episode
        )

    turn_num = 0
    while not result.done and turn_num < max_turns:
        # Build prompt for this turn
        messages.append({"role": "user", "content": format_game_state(result.observation)})
        prompt_text = tokenizer.apply_chat_template(messages, ...)
        prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=False)

        # Calculate remaining budget
        remaining = max_seq_len - len(prompt_tokens)

        if remaining <= 0:
            # No budget left for generation
            is_truncated = True
            truncation_reason = "max_seq_len"
            break

        # Generate with remaining budget
        response = await policy.generate.route(
            [prompt_text],
            sampling_params={"max_tokens": remaining}
        )

        # Check if truncated by vLLM
        if response.stop_reason == "length":
            is_truncated = True
            truncation_reason = "generation_length"
            if cfg.truncation.drop_truncated_generation:
                break  # Drop this turn
            else:
                # Mask this turn
                all_tokens.extend(prompt_tokens)
                all_tokens.extend(response.token_ids)
                response_mask.extend([0] * (len(prompt_tokens) + len(response.token_ids)))
                break

        # Accumulate tokens
        all_tokens.extend(prompt_tokens)
        all_tokens.extend(response.token_ids)
        response_mask.extend([0] * len(prompt_tokens))
        response_mask.extend([1] * len(response.token_ids))
        all_logprobs.extend([0.0] * len(prompt_tokens))
        all_logprobs.extend(response.logprobs)

        # Add to messages and continue
        messages.append({"role": "assistant", "content": response.text})
        action = parse_action(response.text)
        result = env.step(OpenSpielAction(action_id=action, game_name="blackjack"))
        turn_num += 1

    # Create episode
    return Episode(
        episode_id=game_id,
        is_truncated=is_truncated,
        truncation_reason=truncation_reason,
        all_token_ids=torch.tensor(all_tokens),
        logprobs=torch.tensor(all_logprobs),
        response_mask=torch.tensor(response_mask),
        reward=calculate_reward(result.reward),
        message_log=messages,
        # ...
    )

# Outer function for group rollout
async def rollout_group(
    group_size: int,
    server_url: str,
    policy: Generator,
    tokenizer,
    max_seq_len: int,
    max_turns: int,
) -> list[Episode]:
    """Rollout group_size games in parallel"""
    games = [
        play_single_game(
            game_id=str(uuid.uuid4()),
            server_url=server_url,
            policy=policy,
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            max_turns=max_turns,
        )
        for _ in range(group_size)
    ]
    return await asyncio.gather(*games)
```

---

## Final Configuration Schema

```yaml
# apps/blackjack/qwen3_1_7b.yaml

blackjack_env:
  max_seq_len: 2048              # Episode-level budget (all turns)
  max_turns: 10                  # Hard limit on turns per episode

grpo:
  group_size: 16                 # Fixed group size (stays 16 until advantages computed)
  accept_truncated: true         # Accept truncated episodes (learn from partial success)
  # Future: min_advantage, etc.

truncation:
  # How to handle truncated generations (LLM responses)
  drop_truncated_generation: true     # Drop incomplete turn (Tinker approach)
                                      # If false, masks it (TRL approach)

  # How to handle truncated tool responses
  drop_truncated_tool_response: false # Truncate to budget (default)
                                      # If true, drop turn entirely (Tinker approach)

policy:
  engine_args:
    enable_prefix_caching: true  # Critical for multi-turn
    max_model_len: 4096
```

---

## Summary Decision Table

| Design Question | Decision | Reasoning |
|----------------|----------|-----------|
| **Detect truncation** | `stop_reason == "length"` + budget check | Explicit and reliable |
| **Truncated generation** | Drop by default | Clean, libraries only drop or mask (never train with gradient) |
| **Truncated episode** | Filter at GRPO loop level | Check before adding to buffer, minimize communication |
| **Group filtering** | Drop groups with constant rewards only | Simple, efficient |
| **Ref model timing** | After group filtering, before episode acceptance | Process all valid groups (fixed size) |
| **Group sizes** | Fixed until advantages, then dissolve | Simplifies advantage computation |
| **Tool results** | Truncate by default, drop as option | Env controls per-tool limits |
| **Budget check** | Before while loop + stop_reason during loop | Simpler than checking every iteration |
| **Rollout structure** | Separate `play_single_game()` and `rollout_group()` | Matches Tinker pattern, clean separation |

---

**End of Document**
