# Part 3: Truncation Handling for Multi-Turn Episodes

## Problem

**Multi-turn episodes can exceed token budgets in multiple ways:**
1. Initial prompt already too large (rare but possible)
2. Generation truncated mid-response by vLLM (hit `max_tokens` limit)
3. Cumulative tokens across turns exceed `max_seq_len` (episode budget)
4. Tool results too long to fit in remaining budget
5. Episode hits `max_turns` limit before natural completion

**Why this matters:**
- Truncated generations produce incomplete responses (e.g., "HI" instead of "HIT")
- Training on partial tokens can confuse the model
- Groups with all-truncated episodes have no variance (no learning signal)
- Need to decide: drop incomplete data or mask it out during training?

**Root cause:** No unified strategy for detecting truncation, handling partial episodes, and filtering at group vs episode level.

---

## Solution: Episode-Level Budget with Multi-Level Filtering

**Key insights from library investigation (TRL, VERL, NeMo-RL, Tinker, Verifiers):**
1. All libraries check vLLM's `stop_reason == "length"` to detect truncation
2. All libraries only **drop** or **mask** truncated generations - none train with gradient on partial tokens
3. Most filter at two levels: **group-level** (constant rewards) and **episode-level** (acceptance criteria)
4. Reference model timing varies: compute for all episodes (TRL) vs only kept episodes (Tinker)

**Our architecture (based on Tinker's efficient pattern):**
```
Rollout                   Group Filter              Episode Filter              Replay Buffer
   ↓                           ↓                          ↓                            ↓
do_single_rollout()    Drop constant reward     Acceptance criteria         Add accepted episodes
returns Episode       groups (no variance)     (truncated, min_adv)        for training
```

**Fixed group sizes until advantages computed, then dissolve** - training doesn't need groups (packed dataset handles variable lengths).

---

## Current State (from PLAN.md)

### Rollout Loop Checks Budget Per-Turn
```python
async def play_game(..., max_seq_len: int = 2048, max_turns: int = 10):
    # Check if prompt exceeds budget
    if len(prompt_tokens) >= max_seq_len:
        is_truncated = True
        truncation_reason = "max_seq_len"
        break

    # Generate with remaining budget
    remaining = max_seq_len - len(prompt_tokens)
    responses = await policy.generate.route([prompt_text],
                                           sampling_params={"max_tokens": remaining})

    # Check if generation was cut off
    if response.stop_reason == "length":
        is_truncated = True
        truncation_reason = "generation_length"
```

**Problems:**
1. Budget check happens inside while loop on every iteration (inefficient)
2. No group-level filtering for constant rewards
3. No episode-level acceptance criteria (truncated episodes always added to buffer)
4. Reference model computed for all episodes even if we'll drop them
5. No structured rollout pattern (mixing game logic with token tracking)

---

## New State: Complete Rollout and Training Loop

### Architecture Overview

**Two-function pattern (from Tinker):**
- `do_single_rollout()`: Plays one game, returns one Episode
- `rollout_group()`: Plays group_size games in parallel, returns list[Episode]

**Filtering happens at three levels:**
1. **Generation-level**: Drop or mask truncated LLM responses (per-turn decision)
2. **Group-level**: Drop groups with constant rewards (no learning signal)
3. **Episode-level**: Acceptance criteria before adding to buffer (is_truncated, min_advantage, etc.)

### Design Decisions

Below are the 8 key design decisions for truncation handling. Each section includes a brief explanation of the decision and how it's implemented in the loop.

---

#### Decision 1: Detecting Truncation

**Decision:** Use `stop_reason == "length"` as primary signal, with budget check as fallback.

**Why:** vLLM's `stop_reason` field is explicit and reliable - no need to guess based on EOS tokens. We also check cumulative budget to catch cases where the prompt itself exceeds `max_seq_len`.

**Implementation notes:**
- Check initial prompt length BEFORE entering while loop (avoid wasted generation)
- Inside loop: rely on `stop_reason == "length"` to detect mid-generation truncation
- After each turn: budget check happens naturally (prompt includes all previous turns)

---

#### Decision 2: Handling Truncated Generations

**Decision:** Drop incomplete turn by default (Tinker approach), with masking as config option.

**Why:** Clean and simple - if model says "HI" (truncated "HIT"), we don't want to train on that. All investigated libraries offer only two options: drop or mask. **No library trains with gradient on truncated tokens** - masking means `response_mask=0` (zero gradient but kept in batch for ref_model).

**Implementation notes:**
- If `stop_reason == "length"` and `drop_truncated_generation=True`: break loop, don't add tokens
- If `stop_reason == "length"` and `drop_truncated_generation=False`: add tokens but set `response_mask=0`
- Episode still gets final reward (it influenced the outcome), but incomplete turn doesn't contribute gradients

---

#### Decision 3: Handling Truncated Episodes

**Decision:** Filter at GRPO loop level with acceptance criteria, checked BEFORE adding to replay buffer.

**Why:** Minimize communication by checking acceptance before `replay_buffer.add()`. Keeps acceptance logic in GRPO loop (visible), not buried in buffer internals. Allows flexibility for future criteria (min_advantage, etc.).

**Implementation notes:**
- Compute ref_model and advantages for all episodes first
- Loop through episodes and check acceptance criteria
- Only call `replay_buffer.add()` for accepted episodes
- Record metrics for rejection reasons (rate_rejected_truncated, etc.)

---

#### Decision 4: Group-Level Filtering

**Decision:** Drop groups with constant rewards only - keep it simple.

**Why:** If all rewards are identical, `std=0` and advantages become `NaN` (no learning signal). Simple check: `if len(set(rewards)) == 1: drop group`. Don't complicate with truncation logic - episode-level acceptance handles that.

**Implementation notes:**
- Generate all groups (each exactly `group_size` episodes)
- Filter groups before ref_model computation (save compute)
- Record `groups/rate_dropped` metric with 0 or 1 values
- If no valid groups, skip this rollout iteration

---

#### Decision 5: Reference Model Timing

**Decision:** Compute after group filtering, before episode-level acceptance.

**Why:** Filter out useless groups first (constant rewards) to save compute. Then compute ref_model for all episodes in valid groups. Episode-level acceptance happens after advantages computed (need advantages to check min_advantage criterion).

**Implementation notes:**
- Group filtering reduces episode count (saves ref_model compute)
- Ref_model processes all episodes in valid groups (still fixed size per group)
- Episode-level acceptance happens after advantages assigned
- Groups maintain fixed size until advantages computed, then dissolve

---

#### Decision 6: Fixed vs Variable Group Sizes

**Decision:** Fixed group size (e.g., 16) until advantages computed, then dissolve.

**Why:** Simplifies advantage computation (no need to handle variable sizes). Training doesn't need groups anyway - packed dataset handles variable lengths. Groups are only for GRPO advantage normalization.

**Implementation notes:**
- Generate exactly `group_size` episodes per group
- Group filtering maintains fixed size (drop entire group, not individual episodes)
- After advantages computed, pass individual episodes to acceptance check
- Replay buffer receives individual episodes (no concept of groups)

---

#### Decision 7: Handling Truncated Tool Responses

**Decision:** Truncate to budget by default, drop turn as config option.

**Why:** Environment controls per-tool limits (not our config). We only care about overall `max_seq_len` budget. Truncating tool response is less destructive than dropping entire turn.

**Implementation notes:**
- Tokenize tool result and check remaining budget
- If exceeds: truncate tokens to fit (default) or drop turn entirely (config option)
- Record `truncation/rate_tool_response_truncated` metric
- Similar pattern to `drop_truncated_generation` but for tool results

---

#### Decision 8: Budget Check Timing

**Decision:** Check BEFORE entering while loop (initial prompt), then rely on `stop_reason` during loop.

**Why:** Initial prompt might already exceed budget - catch this early. Inside loop: budget is implicitly checked (prompt includes all turns, we set `max_tokens=remaining`). Simpler than checking before every generation.

**Implementation notes:**
- Before while loop: tokenize initial prompt and check `len >= max_seq_len`
- If exceeds: return truncated episode immediately (avoid wasted generation)
- Inside loop: calculate `remaining = max_seq_len - len(prompt_tokens)` and pass to vLLM
- vLLM handles truncation via `stop_reason == "length"`, we react accordingly

---

## Complete Implementation

### 1. Play Single Game (Rollout Function)

This function follows Tinker's `do_single_rollout()` pattern - simple while loop, environment decides when to stop.

```python
async def do_single_rollout(
    game_id: str,
    server_url: str,
    policy: Generator,
    tokenizer,
    max_seq_len: int = 2048,
    max_turns: int = 10,
) -> Episode:
    """
    Play one blackjack game - returns single episode with all turns.

    Budget tracking (Decision 1, 8):
    - Check initial prompt BEFORE while loop
    - Inside loop: rely on stop_reason to detect truncation
    - Dynamic max_tokens = max_seq_len - len(prompt_tokens)

    Truncation handling (Decision 2):
    - If stop_reason == "length": drop or mask based on config
    - Episode marked as is_truncated with reason
    """
    messages = [
        {"role": "system", "content": "You are an expert BlackJack player..."}
    ]

    all_tokens = []
    all_logprobs = []
    response_mask = []
    is_truncated = False
    truncation_reason = None

    env = OpenSpielEnv(base_url=server_url)
    result = env.reset()

    # ============ Decision 8: Check initial prompt BEFORE while loop ============
    initial_prompt = tokenizer.apply_chat_template(messages,
                                                   add_generation_prompt=True,
                                                   tokenize=False)
    initial_tokens = tokenizer.encode(initial_prompt, add_special_tokens=False)

    if len(initial_tokens) >= max_seq_len:
        # Initial prompt too large - return truncated episode immediately
        return Episode(
            episode_id=game_id,
            task_name="blackjack",
            is_truncated=True,
            truncation_reason="initial_prompt_exceeds_budget",
            all_token_ids=torch.tensor(initial_tokens[:max_seq_len]),
            logprobs=torch.zeros(max_seq_len),
            response_mask=torch.zeros(max_seq_len),
            reward=0,  # No game played
            metadata={"num_turns": 0}
        )

    turn_num = 0
    while not result.done and turn_num < max_turns:
        # Build user message with game state
        player_total = result.observation.metadata.get("player_total", "?")
        dealer_card = result.observation.metadata.get("dealer_card", "?")

        state_desc = f"Your hand total: {player_total}\n"
        state_desc += f"Dealer shows: {dealer_card}\n"
        state_desc += "What do you do? Output only 'HIT' or 'STAND'."

        messages.append({"role": "user", "content": state_desc})

        # ============ Decision 1, 8: Format and check budget ============
        prompt_text = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False
        )
        prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=False)

        # Check remaining budget
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
        response = response[0]

        # ============ Decision 1, 2: Check if truncated by vLLM ============
        if response.stop_reason == "length":
            is_truncated = True
            truncation_reason = "generation_length"

            if cfg.truncation.drop_truncated_generation:
                # Drop this turn entirely - don't add tokens
                break
            else:
                # Mask this turn - add tokens but set response_mask=0
                all_tokens.extend(prompt_tokens)
                all_tokens.extend(response.token_ids)
                response_mask.extend([0] * (len(prompt_tokens) + len(response.token_ids)))
                all_logprobs.extend([0.0] * len(prompt_tokens))
                all_logprobs.extend(response.logprobs)
                break

        # ============ Accumulate tokens (normal case) ============
        all_tokens.extend(prompt_tokens)
        all_tokens.extend(response.token_ids)
        response_mask.extend([0] * len(prompt_tokens))  # Don't train on prompts
        response_mask.extend([1] * len(response.token_ids))  # Train on responses
        all_logprobs.extend([0.0] * len(prompt_tokens))
        all_logprobs.extend(response.logprobs)

        # Parse and execute action
        messages.append({"role": "assistant", "content": response.text})
        action = parse_action(response.text)  # Returns "HIT", "STAND", or "INVALID"

        if action == "INVALID":
            action = "STAND"  # Fallback
            action_id = 1
        elif action == "HIT":
            action_id = 0
        else:  # STAND
            action_id = 1

        result = env.step(OpenSpielAction(action_id=action_id, game_name="blackjack"))
        turn_num += 1

    # Check if hit max_turns
    if turn_num >= max_turns and not result.done:
        is_truncated = True
        truncation_reason = "max_turns"

    # Calculate final reward
    env_reward = result.reward
    reward = calculate_reward(env_reward)  # Custom shaping: Win=+3, Loss=-1

    # Create episode
    return Episode(
        episode_id=game_id,
        task_name="blackjack",
        generator_version=0,  # TODO: Get from policy
        is_truncated=is_truncated,
        all_token_ids=torch.tensor(all_tokens, dtype=torch.long),
        logprobs=torch.tensor(all_logprobs, dtype=torch.float),
        response_mask=torch.tensor(response_mask, dtype=torch.float),
        reward=reward,
        advantage=None,  # Computed later
        ref_logprobs=None,  # Computed later
        message_log=messages,
        metadata={
            "num_turns": turn_num,
            "env_reward": env_reward,
            "truncation_reason": truncation_reason,
        }
    )
```

**Key implementation notes:**
- Initial prompt check happens once before loop (Decision 8)
- Budget naturally enforced inside loop via `max_tokens=remaining` (Decision 1)
- Truncated generation handling: drop or mask based on config (Decision 2)
- Returns single Episode with all turns concatenated

---

### 2. Rollout Group (Outer Function)

This function follows Tinker's `do_group_rollout()` pattern - parallel execution, fixed group size.

```python
async def rollout_group(
    group_size: int,
    server_url: str,
    policy: Generator,
    tokenizer,
    max_seq_len: int,
    max_turns: int,
) -> list[Episode]:
    """
    Rollout group_size games in parallel.

    Group stays exactly group_size until returned (Decision 6).
    No filtering at this level - happens in continuous_rollouts().
    """
    rollouts = [
        do_single_rollout(
            game_id=str(uuid.uuid4()),
            server_url=server_url,
            policy=policy,
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            max_turns=max_turns,
        )
        for _ in range(group_size)
    ]
    return await asyncio.gather(*rollouts)
```

**Key implementation notes:**
- Exactly `group_size` episodes returned (Decision 6)
- Parallel execution via `asyncio.gather()`
- Simple wrapper - filtering happens at higher level

---

### 3. Continuous Rollouts (Main GRPO Loop)

This is where all filtering decisions happen (Decisions 3, 4, 5, 6).

```python
async def continuous_rollouts(tokenizer):
    """
    Main GRPO rollout loop with multi-level filtering.

    Flow:
    1. Generate groups (fixed size)
    2. Filter groups (constant rewards) - Decision 4
    3. Compute ref_model for valid groups - Decision 5
    4. Compute advantages (groups still fixed size)
    5. Episode-level acceptance (groups dissolve) - Decision 3, 6
    6. Add accepted episodes to buffer
    """
    server_url = cfg.blackjack_env.server_url
    max_seq_len = cfg.blackjack_env.max_seq_len
    max_turns = cfg.blackjack_env.max_turns
    group_size = cfg.grpo.group_size
    num_groups = cfg.grpo.get("num_groups_per_rollout", 4)

    while not shutdown_event.is_set(): # TODO: why shutdown_event and not just while true?
        # ============ Step 1: Generate all groups (Decision 6: Fixed size) ============
        all_groups = [] #TODO: remove this logic of "all_groups". We do one group per loop, no?
        for group_idx in range(num_groups):
            group = await rollout_group(
                group_size=group_size,
                server_url=server_url,
                policy=policy,
                tokenizer=tokenizer,
                max_seq_len=max_seq_len,
                max_turns=max_turns,
            )
            all_groups.append(group)

        # ============ Step 2: Filter groups (Decision 4: Constant rewards) ============
        valid_groups = []
        for group in all_groups:
            rewards = [e.reward for e in group]
            if len(set(rewards)) > 1:  # At least 2 different reward values
                valid_groups.append(group)
                record_metric("groups/rate_dropped", 0, Reduce.MEAN)
            else:
                record_metric("groups/rate_dropped", 1, Reduce.MEAN)

        if not valid_groups:
            # All groups had constant rewards - skip this rollout
            continue

        # ============ Step 3: Compute ref_model for valid groups (Decision 5) ============
        # Flatten valid groups to list of episodes (groups still conceptually intact)
        all_valid_episodes = [e for g in valid_groups for e in g]

        # Pad to max length in batch
        max_len = max(len(e.all_token_ids) for e in all_valid_episodes)
        padded_tokens = []
        for episode in all_valid_episodes:
            seq_len = len(episode.all_token_ids)
            pad_len = max_len - seq_len
            padded = F.pad(episode.all_token_ids, (0, pad_len), value=pad_id)
            padded_tokens.append(padded)

        input_ids = torch.stack(padded_tokens)  # [batch, max_len]

        # Compute ref_model logprobs
        ref_logprobs = await ref_model.forward.route(
            input_ids,
            0,  # No separate prompt length (response_mask handles it)
            return_logprobs=True
        )

        # Assign ref_logprobs to episodes (unpad)
        for i, episode in enumerate(all_valid_episodes):
            seq_len = len(episode.all_token_ids)
            episode.ref_logprobs = ref_logprobs[i, :seq_len]

        del ref_logprobs, input_ids

        # ============ Step 4: Compute advantages per group (Decision 6: Groups still fixed) ============
        for group in valid_groups:
            advantages = await compute_advantages.compute.call_one(group)
            for episode, advantage in zip(group, advantages):
                episode.advantage = advantage

        # ============ Step 5: Episode-level acceptance (Decision 3, 6: Groups dissolve) ============
        accepted_episodes = []
        for group in valid_groups:
            for episode in group:
                should_accept = True

                # Acceptance criterion: is_truncated
                if episode.is_truncated and not cfg.grpo.accept_truncated:
                    should_accept = False
                    record_metric("buffer/rate_rejected_truncated", 1, Reduce.MEAN)
                else:
                    record_metric("buffer/rate_rejected_truncated", 0, Reduce.MEAN)

                # Future: Add min_advantage criterion here
                # if episode.advantage < cfg.grpo.min_advantage:
                #     should_accept = False

                if should_accept:
                    accepted_episodes.append(episode)

        # ============ Step 6: Add to replay buffer (Decision 3) ============
        # TODO: Add all episodes at once instead of one by one
        for episode in accepted_episodes:
            await replay_buffer.add.call_one(episode)

        record_metric("buffer/episodes_accepted", len(accepted_episodes), Reduce.SUM)
        record_metric("buffer/episodes_generated", len(all_valid_episodes), Reduce.SUM)
```

**Key implementation notes:**
- Groups generated with fixed size (Decision 6)
- Group filtering before ref_model saves compute (Decision 4, 5)
- Ref_model computed for all episodes in valid groups (Decision 5)
- Advantages computed per group (groups still intact, Decision 6)
- Episode-level acceptance after advantages (groups dissolve, Decision 3, 6)
- Acceptance logic in GRPO loop, not replay buffer (Decision 3)

---

## Configuration Schema

All design decisions are controlled via config:

```yaml
# apps/blackjack/qwen3_1_7b.yaml

blackjack_env:
  max_seq_len: 2048              # Episode-level budget (all turns) - Decision 8
  max_turns: 10                  # Hard limit on turns per episode

grpo:
  group_size: 16                 # Fixed group size (stays 16 until advantages computed) - Decision 6
  num_groups_per_rollout: 4      # How many groups to generate per rollout iteration
  accept_truncated: true         # Accept truncated episodes - Decision 3
                                 # Set to false to drop incomplete episodes
  # Future: min_advantage filter

truncation:
  # How to handle truncated generations (LLM responses) - Decision 2
  drop_truncated_generation: true     # Drop incomplete turn (Tinker approach)
                                      # If false, masks it (TRL approach)

  # How to handle truncated tool responses - Decision 7
  drop_truncated_tool_response: false # Truncate to budget (default)
                                      # If true, drop turn entirely

policy:
  engine_args:
    enable_prefix_caching: true  # Critical for multi-turn (2-3x speedup)
    max_model_len: 4096          # vLLM model context length
```

---

## Summary of Design Decisions

| Decision | Choice | Config |
|----------|--------|--------|
| **1. Detect truncation** | `stop_reason == "length"` + budget check | N/A |
| **2. Truncated generation** | Drop by default (Tinker) | `truncation.drop_truncated_generation` |
| **3. Truncated episode** | Filter at GRPO loop before buffer | `grpo.accept_truncated` |
| **4. Group filtering** | Drop groups with constant rewards | N/A (always enabled) |
| **5. Ref model timing** | After group filter, before episode filter | N/A |
| **6. Group sizes** | Fixed (16) until advantages, then dissolve | `grpo.group_size` |
| **7. Tool results** | Truncate by default, drop as option | `truncation.drop_truncated_tool_response` |
| **8. Budget check** | Before while loop + stop_reason during loop | `blackjack_env.max_seq_len` |

**Key principle:** All libraries only **drop** or **mask** truncated generations - none train with gradient on partial tokens. Masking means `response_mask=0` (zero gradient but kept in batch for ref_model).

---

## Benefits

1. **Efficient budget tracking**: Check initial prompt once, rely on `stop_reason` during loop
2. **Flexible truncation handling**: Drop or mask via config (matches library patterns)
3. **Multi-level filtering**: Groups (constant rewards) → Episodes (acceptance criteria)
4. **Optimized ref_model**: Compute after group filtering (save compute on dropped groups)
5. **Fixed group sizes**: Simplifies advantage computation (variable lengths handled in training)
6. **Clean rollout structure**: Separate `do_single_rollout()` and `rollout_group()` (matches Tinker)
7. **Extensible acceptance**: Easy to add min_advantage, max_length, etc.
8. **Proper metrics**: Track truncation reasons, rejection rates, group drop rates

---

## Migration from Current PLAN.md

### Changes to `play_game()`:
1. Move budget check BEFORE while loop (only check initial prompt once)
2. Add truncated generation handling (drop vs mask based on config)
3. Return truncated episode immediately if initial prompt exceeds budget

### Changes to `continuous_rollouts()`:
1. Add group generation loop (`rollout_group()` wrapper)
2. Add group-level filtering (constant rewards)
3. Compute ref_model for valid groups only
4. Add episode-level acceptance criteria before buffer
5. Record new metrics (rate_dropped, rate_rejected_truncated)

### Changes to config:
1. Add `grpo.accept_truncated` flag
2. Add `truncation.drop_truncated_generation` flag
3. Add `truncation.drop_truncated_tool_response` flag (future tool calling)
4. Add `grpo.num_groups_per_rollout` parameter

---

**End of Document**
