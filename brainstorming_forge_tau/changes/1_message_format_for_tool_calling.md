# Part 5: Message Format for Tool Calling

## Problem

**Current:** Dataset calls `tokenizer.apply_chat_template()` at data loading time, converting messages to strings.

**Why this breaks tool calling:**
1. Can't add tool definitions to prompts (lost message structure)
2. Can't do multi-turn (need to rebuild prompt each turn with updated history)
3. Can't manage conversation state

**Root cause:** Formatting happens too early (dataset) instead of per-turn (rollout loop).

---

## Solution: Format in Rollout Loop

**Key insight:** All frameworks (VERL, TRL, Tinker, NeMo-RL) format messages in the rollout loop, not the dataset or generator.

**Architecture:**
```
Dataset              Rollout Loop                   Generator
   ↓                      ↓                             ↓
Return messages   apply_chat_template()      Receive string
(structured)      per turn with tools         (unchanged)
```

**Generator doesn't change** - stays stateless, keeps `generate(prompt: str) → Completion` API.

---

## Current State

### Dataset (apps/grpo/main.py:217-234)
```python
def gsm8k_transform(sample):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": sample["question"]},
    ]

    # ❌ Formatting happens HERE - too early
    formatted_request = self._tokenizer.apply_chat_template(messages, ...)
    return {"request": formatted_request, "target": formatted_target}
```

### Rollout Loop (apps/grpo/main.py:359-373)
```python
async def continuous_rollouts():
    sample = await dataloader.sample.call_one()

    prompt, target = sample["request"], sample["target"]  # Already a string
    responses = await policy.generate.route(prompt)
```

**Problem:** Once formatted to string, can't add tools or continue multi-turn conversation.

---

## New State (Single-Turn)

### 1. Dataset Returns Messages
```python
def gsm8k_transform(sample):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": sample["question"]},
    ]

    target = sample["answer"].split("#### ")[1]

    # ✅ Return structured messages
    return {"messages": messages, "target": target}
```

### 2. Add Tokenizer to Main
```python
async def main(cfg: DictConfig):
    # ... after service initialization ...

    # ✅ Get tokenizer for rollout loop
    from vllm.transformers_utils.tokenizer import get_tokenizer
    tokenizer = get_tokenizer(cfg.dataset.model)
```

### 3. Format in Rollout Loop
```python
async def continuous_rollouts(tokenizer):  # ✅ Add parameter
    sample = await dataloader.sample.call_one()

    messages, target = sample["messages"], sample["target"]  # ✅ Get messages

    # ✅ Format HERE in rollout loop
    prompt_str = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False
    )

    # Generator receives string (same as before!)
    responses = await policy.generate.route(prompt_str)
```

### 4. Pass Tokenizer to Tasks
```python
rollout_tasks = [
    asyncio.create_task(continuous_rollouts(tokenizer))  # ✅ Pass tokenizer
    for _ in range(num_rollout_threads)
]
```

---

## New State (Multi-Turn with Tools)

For multi-turn, extend the rollout loop. Generator still doesn't change.

```python
async def play_task(
    messages: list[dict],  # From dataset
    tools: list[dict],      # From environment
    env,                    # Environment client
    generator,              # Forge Generator (unchanged!)
    tokenizer,
    max_turns: int = 10,
):
    """Multi-turn rollout with tool calling."""

    for turn in range(max_turns):
        # 1. Format with tools (ROLLOUT LOOP does this each turn)
        prompt_str = tokenizer.apply_chat_template(
            messages,
            tools=tools,  # ← Add tools to prompt
            add_generation_prompt=True,
            tokenize=False
        )

        # 2. Generate (generator API unchanged)
        response = await generator.generate.route(prompt_str)

        # 3. Parse tool calls
        tool_calls = parse_tool_calls(response.text)

        if tool_calls:
            # 4. Add assistant message + tool calls
            messages.append({
                "role": "assistant",
                "content": response.text,
                "tool_calls": tool_calls
            })

            # 5. Execute tools and add results
            for tc in tool_calls:
                result = await env.execute_tool(tc["name"], tc["args"])
                messages.append({
                    "role": "tool",
                    "content": result.content
                })
            # Loop continues - reformats with updated messages
        else:
            # 6. Final answer
            messages.append({"role": "assistant", "content": response.text})
            break

    return messages, response
```

**Key:** Rollout loop manages history, formats each turn, generator stays stateless.
