# Truncation Strategy for Multi-Turn Episodes

**Dependencies:**
- `1_message_format_for_tool_calling.md` (dataset returns messages, format in rollout loop)
- `2_episode_class.md` (new Episode class with response_mask)

---

## Problem

Single-turn blackjack has fixed `max_tokens` per generation with no episode-level budget tracking.

**Why this breaks multi-turn:**
1. Episode can grow unbounded (turn1: 100 tokens, turn2: 200 tokens, turn3: 500 tokens → 800 tokens total)
2. Can exceed model's `max_model_len` (crashes inference)
3. Tool results can be arbitrarily long (web search: 10K tokens)
4. No clear strategy for when to stop adding turns

**Root cause:** Need episode-level budget (`max_seq_len`) that spans all turns.

---

## Solution: Episode-Level Budget + Per-Turn Checks

All frameworks (Tinker, VERL, NeMo-RL) check prompt length each turn and terminate when budget exhausted.

**Architecture:**
```
Dataset → Rollout Loop → Generator
   ↓           ↓             ↓
Returns    Each Turn:    Receives
messages   1. Build prompt from messages (includes full history)
           2. Check: len(prompt_tokens) >= max_seq_len? → STOP
           3. Generate with remaining budget
           4. Add response to messages
           5. Parse tools, execute, add results → Loop
```

Prompt already includes all history, so no cumulative tracking needed.

---

## Implementation

### Prerequisites (from docs 1 & 2)

**From `1_message_format_for_tool_calling.md`:**
- Dataset returns `{"messages": [...], "target": ...}` instead of formatted strings
- Tokenizer passed from main → rollout loop → play_game
- `apply_chat_template()` called in rollout loop each turn

**From `2_episode_class.md`:**
- New Episode class with `all_token_ids`, `response_mask`, `logprobs`
- Drop old `pad_id`, `request_len`, `response_len` fields
- Add `generator_version`, `is_truncated`, `task_name`, `message_log`

### 1. Config Parameters

```yaml
blackjack_env:
  max_seq_len: 2048              # Total episode budget (all turns)
  max_turns: 10                  # Hard limit on turns
  max_tool_result_length: 1024   # Global, token-based (for future tool calling)

grpo:
  include_truncated_in_buffer: false  # Drop incomplete episodes

policy:
  engine_args:
    enable_prefix_caching: true  # Critical for multi-turn (2-3x speedup)
    # max_model_len: 4096        # this is defined dinamically on generate

### 2. Dataset Returns Messages

```python
async def sample_blackjack_episode():
    """Dataset returns initial messages for the game."""
    return {
        "messages": [
            {"role": "system", "content": "You are a blackjack expert..."}
        ],
        "target": None,
        "task_name": "blackjack",  # TODO: Investigate how other frameworks structure dataset output
    }
```

**Note:** `task_name` should probably come from the dataset. Need to investigate how other frameworks handle dataset
 schema (likely using TypedDict or dataclass for consistent fields across datasets). This investigation should be done in a separate document.

### 3. Main: Get Tokenizer and Pass to Rollout Loop

```python
async def main(cfg: DictConfig):
    # ... after service initialization ...

    # Get tokenizer for use in rollout loop
    from vllm.transformers_utils.tokenizer import get_tokenizer
    tokenizer = get_tokenizer(cfg.dataset.model)

    # Start rollout threads with tokenizer
    rollout_tasks = [
        asyncio.create_task(continuous_rollouts(tokenizer))
        for _ in range(num_rollout_threads)
    ]
```

### 4. Rollout Loop: Format Messages Each Turn

```python
async def continuous_rollouts(tokenizer):
    while not shutdown_event.is_set():
        # Sample structured data from dataset
        sample = await dataloader.sample.call_one()
        initial_messages = sample["messages"]
        target = sample["target"]
        task_name = sample["task_name"]

        # Play episode with budget tracking
        episode = await play_game(
            game_id=str(uuid.uuid4()),
            messages=initial_messages,
            task_name=task_name,
            policy=policy,
            tokenizer=tokenizer,
            max_seq_len=cfg.max_seq_len,
            max_turns=cfg.max_turns,
        )

        # Add to buffer, calculate advantages, etc.
        ...
```

### 5. Play Game: Budget Tracking Each Turn

```python
async def play_game(
    game_id: str,
    messages: list[dict],
    task_name: str,
    policy: Generator,
    tokenizer,
    max_seq_len: int,
    max_turns: int,
) -> Episode:
    messages = messages.copy()

    all_tokens = []
    all_logprobs = []
    response_mask = []
    is_truncated = False

    env = OpenSpielEnv(base_url=server_url)
    result = env.reset()

    for turn in range(max_turns):
        if result.done:
            break

        # Add user message with current game state
        messages.append({"role": "user", "content": format_game_state(result.observation)})

        # Format prompt from full message history
        prompt_text = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False
        )

        # Encode to check if prompt exceeds budget
        prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=False)

        if len(prompt_tokens) >= max_seq_len:
            is_truncated = True
            record_metric("episode/terminated_budget_exceeded", 1, Reduce.MEAN)
            break

        # Calculate remaining budget for this turn
        remaining = max_seq_len - len(prompt_tokens)

        # Generate with remaining budget
        responses = await policy.generate.route(
            [prompt_text],
            sampling_params={"max_tokens": remaining}
        )
        response = responses[0]

        # Check if generation was cut off by max_tokens
        if response.stop_reason == "length":
            is_truncated = True
            record_metric("episode/generation_truncated", 1, Reduce.MEAN)
            break

        # Accumulate tokens and build response mask
        all_tokens.extend(prompt_tokens)
        all_tokens.extend(response.token_ids)
        response_mask.extend([0] * len(prompt_tokens))  # Don't train on prompts
        response_mask.extend([1] * len(response.token_ids))  # Train on responses
        all_logprobs.extend([0.0] * len(prompt_tokens))
        all_logprobs.extend(response.logprobs)

        # Add assistant response to message history
        messages.append({"role": "assistant", "content": response.text})

        # Execute action in environment
        action = parse_action(response.text)
        result = env.step(OpenSpielAction(action_id=action, game_name="blackjack"))

    # Create episode with accumulated data
    return Episode(
        episode_id=game_id,
        task_name=task_name,
        generator_version=get_policy_version(),
        is_truncated=is_truncated,
        all_token_ids=torch.tensor(all_tokens, dtype=torch.long),
        logprobs=torch.tensor(all_logprobs, dtype=torch.float),
        response_mask=torch.tensor(response_mask, dtype=torch.float),
        reward=result.reward,
        message_log=messages,
        metadata={"num_turns": turn + 1}
    )
```

### 6. Tool Result Truncation (Future)

```python
def truncate_to_budget(
    text: str,
    tokenizer,
    max_tokens: int,
    side: str = "left"
) -> str:
    """Truncate text to max_tokens. Side: 'left', 'right', or 'middle'."""
    tokens = tokenizer.encode(text, add_special_tokens=False)

    if len(tokens) <= max_tokens:
        return text

    if side == "left":
        return tokenizer.decode(tokens[:max_tokens]) + "...(truncated)"
    elif side == "right":
        return "(truncated)..." + tokenizer.decode(tokens[-max_tokens:])
    else:
        half = max_tokens // 2
        return (tokenizer.decode(tokens[:half]) +
                "...(truncated)..." +
                tokenizer.decode(tokens[-half:]))

# Usage in multi-turn loop with tools
for tool_call in tool_calls:
    result = await execute_tool(tool_call)

    # Truncate tool result to prevent budget overflow
    truncated_result = truncate_to_budget(
        str(result),
        tokenizer,
        max_tool_result_length,
        side="left"
    )

    messages.append({"role": "tool", "content": truncated_result})
```

---

## Key Design Decisions

| Decision | Choice | Reasoning |
|----------|--------|-----------|
| **Dataset format** | Messages | Dataset returns structured messages, formatting happens in rollout loop |
| **Episode fields** | New class | `response_mask` instead of `pad_id/request_len/response_len` for variable-length multi-turn |
| **Encoding location** | Inside loop | Need to check budget before generating. Prompt includes full history |
| **Cumulative tracking** | No | Redundant - prompt already contains all turns |
| **Dynamic max_tokens** | Calculate remaining | `max_tokens = max_seq_len - len(prompt_tokens)` |
| **Tool truncation unit** | Tokens | Accurate for budget, consistent with max_seq_len |
| **Tool truncation scope** | Global | Start simple, add per-tool later if needed |
| **Mid-generation truncation** | Stop immediately | Don't parse tools if `stop_reason == "length"` |
| **Truncated episodes** | Configurable | `include_truncated_in_buffer: false` to drop them |
| **Prefix caching** | Required | 2-3x speedup for multi-turn |

---

## Research Findings Summary

Analyzed TRL, VERL, NeMo-RL, Tinker, Verifiers:

| Library | Prompt Check? | Tool Truncation? | Mid-Generation Handling |
|---------|--------------|------------------|------------------------|
| **Tinker** | Each turn | Terminates instead | No stop_reason check |
| **VERL** | Each turn | Global (256 chars) | Silent failure |
| **NeMo-RL** | Each turn | Dynamic (tokens) | No stop_reason check |
| **TRL** | Relies on vLLM | No | No check |
| **Verifiers** | Post-hoc | No | Crashes on incomplete JSON |

**Best practices:**
- Check prompt length each turn, terminate if exceeds (Tinker)
- Token-based truncation, dynamic allocation (NeMo-RL)
- Global tool result truncation config (VERL)
- Check `stop_reason == "length"` before parsing tools (new)

---

## Migration from Current Blackjack

### Breaking Changes

**From Episode class (doc 2):**
1. Drop `pad_id`, `request_len`, `response_len` → Add `response_mask`
2. Update collate function for dynamic padding
3. Update loss computation to use `response_mask`

**From message format (doc 1):**
4. Dataset returns `{"messages": [...], "target": ...}` instead of formatted strings
5. Get tokenizer in main and pass to rollout loop
6. Rollout loop passes tokenizer to play_game
7. play_game receives `messages` parameter from dataset

### Non-Breaking

- Generator API unchanged: `generate(prompt: str) → Completion`
- Single-turn still works (1 iteration of loop)
- Configs additive with defaults

---

## Next Steps

1. Update Episode class (see `2_episode_class.md`)
2. Add tokenizer to rollout loop (see `1_message_format_for_tool_calling.md`)
3. Implement budget checking in rollout loop (this doc)
4. Update dataset to return messages
5. Add truncation metrics to dashboard
6. Test with various `max_seq_len` values

---

**End of Document**
