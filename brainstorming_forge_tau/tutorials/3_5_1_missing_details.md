# Missing Details from 3_5 Ideal State

This document identifies critical missing details for implementing a production-ready multi-turn tool-calling RL loop with Forge + vLLM + OpenEnv.

**Organization:**
- **Section 1**: Core details to add to main 3_5 loop
- **Section 2**: Appendix items (configuration, generation args)
- **Section 3**: Open questions requiring clarification

---

## Section 1: Core Details for Main Loop

### 1. Multi-Environment Routing

**What's missing:** How to handle multiple task types (websearch, coding, airline) with different tools and configurations.

**Where it goes:** `continuous_rollouts()` function

**Pattern:** Verifiers EnvGroup (task-based routing) or Tinker CompositeDataset (batch-level mixing)

```python
# In continuous_rollouts:
task = await dataloader.sample.call_one()
# task includes: prompt, task_type, metadata

# Environment map per task type
env_map = {
    "websearch": websearch_env,
    "coding": coding_env,
    "airline": airline_env,
}

# Route to correct environment
env_client = env_map[task.task_type]
env_state = env_client.reset()
tool_schemas = env_state.observation.tools

# Different max_turns per environment
max_turns_config = {
    "websearch": 10,
    "coding": 15,
    "airline": 8,
}
max_turns = max_turns_config[task.task_type]
```

**References:**
- Verifiers: `verifiers/envs/env_group.py:218-266` (rollout routing)
- Tinker: `tinker-cookbook/distillation/datasets.py:45-83` (CompositeDataset)

---

### 2. Tool Call Parsing

**What's missing:** How `parse_tool_call()` works and format options.

**Where it goes:** Called in `play_task()` loop

**Design choice:** Use Tinker's text-based parsing (simple), with option to leverage vLLM native parsing later.

```python
# In play_task:
response = await policy.generate.route(prompt, n=1)

# Parse tool call from response
# Using Tinker pattern: XML tags <tool_call>...</tool_call>
# Alternative: vLLM native parsing (see Appendix)
tool_call = parse_tool_call(response.text)

if tool_call:
    # tool_call = {"name": "search_wiki", "args": {"query": "..."}}
    action = ToolCallAction(
        tool_name=tool_call["name"],
        parameters=tool_call["args"]
    )
```

**Note:** Can use vLLM's native `tool_call_parser="hermes"` for automatic parsing (see Appendix for configuration).

**References:**
- Tinker: `<function_call>...</function_call>` XML tags
- VERL: Uses SGLang's FunctionCallParser
- PRIME-RL: `enable_auto_tool_choice=True, tool_call_parser="hermes"`

---

### 3. Tool Response Truncation

**What's missing:** Handling very long tool outputs that could exceed context limits.

**Where it goes:** After `env.step(action)` in `play_task()`

```python
if tool_call:
    result = env.step(action)
    tool_content = result.observation.content

    # Truncate long tool responses
    tool_tokens = tokenizer.encode(tool_content, add_special_tokens=False)
    tool_tokens = truncate(tool_tokens, max_length=256)  # TODO: Decide where truncate() lives (env vs explicit in loop)
    tool_content = tokenizer.decode(tool_tokens)

    # Add to messages
    messages.append({"role": "tool", "content": tool_content})
```

**TODO:** Decide where `truncate()` utility lives:
- Option A: Environment handles truncation before returning
- Option B: Explicit in rollout loop (shown above)
- Option C: Utility function shared across environments

**References:**
- VERL: `max_tool_response_length=256`, `tool_response_truncate_side="middle"`
- VERL: `verl/experimental/agent_loop/tool_agent_loop.py:457-464`

---

### 4. Parallel Episode Collection

**What's missing:** Currently sequential episode collection blocks on each `play_task()` call.

**Where it goes:** `continuous_rollouts()` when creating G samples per task

```python
# In continuous_rollouts:

# TODO: Investigate how to parallelize this instead of sequential execution
# Current (sequential):
episodes = []
for _ in range(group_size):
    episode = await play_task(policy, task_prompt, tool_schemas, env, max_turns)
    episodes.append(episode)

# Future (parallel with asyncio.gather):
# episode_tasks = [
#     play_task(policy, task_prompt, tool_schemas, env, max_turns)
#     for _ in range(group_size)
# ]
# episodes = await asyncio.gather(*episode_tasks)
```

**References:**
- NeMo-RL: `RL/nemo_rl/experience/rollouts.py:780-936` (per-sample async tasks)
- BlackJack: Sequential execution (current pattern)

---

### 5. Episode Metadata

**What's missing:** Tracking episode statistics for debugging and analysis.

**Where it goes:** `play_task()` and Episode dataclass

```python
# In play_task:
turn = 0
metadata = {}  # Track episode stats

while not done and turn < max_turns:
    # ... generation and tool execution ...
    turn += 1

# Populate metadata
metadata = {
    "num_turns": turn,
    "truncated": turn >= max_turns,
    # ... other stats moved to appendix
}

# Store in Episode
episode = Episode(
    ...,
    metadata=metadata  # New field
)
```

**See Appendix** for full list of metadata fields (num_tool_calls, termination_reason, etc.)

**References:**
- NeMo-RL: `RL/nemo_rl/experience/rollouts.py:512,523-526` (truncation tracking)
- Tinker: `Transition.metrics` field

---

### 6. System Prompt Formatting

**What's missing:** How system prompt with tool instructions is created.

**Where it goes:** Dataset definition or tokenizer's chat template handles this.

**Design choice:** System prompt comes from either:
1. Dataset provides it per task type
2. Tokenizer's `apply_chat_template()` handles it when `tools=` parameter is passed

```python
# In play_task:
# Option 1: Dataset provides system prompt
messages = [
    {"role": "system", "content": task.system_prompt},  # From dataset
    {"role": "user", "content": task_prompt}
]

# Option 2: Tokenizer handles it via tools parameter
messages = [{"role": "user", "content": task_prompt}]
prompt = tokenizer.apply_chat_template(
    messages,
    tools=tool_schemas,  # Tokenizer injects system prompt with tool definitions
    add_generation_prompt=True,
    tokenize=False
)
```

**Clarification needed:** Determine if Forge's current tokenizer setup supports `tools=` parameter.

**References:**
- Tinker: `SEARCH_TOOL_SYSTEM_PROMPT` in `tinker-cookbook/recipes/tool_use/search/search_env.py`
- Verifiers: System message with tool definitions

---

### 7. Response Mask in Training

**What's missing:** How `response_mask` is passed to trainer.

**Where it goes:** `continuous_training()` and `trainer.train_step()`

```python
# In continuous_training:
batch = await replay_buffer.sample(batch_size)

# Train on batch
await trainer.train_step(
    inputs=batch["prompt_ids"],
    targets=batch["response_ids"],
    advantages=batch["advantages"],
    ref_logprobs=batch["ref_logprobs"],
    response_mask=batch["response_mask"],  # Pass mask to trainer
)
```

**Note:** No need to show implementation of mask application in 3_5. Just show the API.

**References:**
- VERL: `verl/trainer/ppo/core_algos.py:787-808` (masked loss aggregation)
- Verifiers: `mask_env_responses` flag

---

### 8. Error Handling

**What's missing:** Handling tool execution failures and malformed responses.

**Where it goes:** `play_task()` around `env.step()`

```python
# In play_task:
if tool_call:
    try:
        result = env.step(action)
    except Exception as e:
        # Add error message instead of tool result
        messages.append({
            "role": "tool",
            "content": f"Error: {str(e)}"
        })
        # Continue to next turn or terminate based on policy
```

**References:**
- VERL: `verl/experimental/agent_loop/tool_agent_loop.py:1329-1357` (try/except with cleanup)

---

### 9. Parallel Tool Execution (Multiple Tools Per Turn)

**What's missing:** Handling multiple tool calls in a single response and executing them in parallel.

**Where it goes:** `play_task()` loop

```python
# In play_task:
# Parse multiple tool calls (if model calls multiple tools)
tool_calls = parse_tool_calls(response.text)  # Returns list

if tool_calls:
    # TODO: Confirm environment can handle parallel requests
    # Execute all tools in parallel
    tool_tasks = [
        env.execute_tool(tc["name"], tc["args"])
        for tc in tool_calls
    ]
    tool_results = await asyncio.gather(*tool_tasks)

    # Add assistant message with all tool calls
    messages.append({
        "role": "assistant",
        "tool_calls": tool_calls
    })

    # Add all tool results
    for tool_result in tool_results:
        messages.append({
            "role": "tool",
            "content": tool_result.content
        })
```

**References:**
- VERL: `verl/experimental/agent_loop/tool_agent_loop.py:1256-1266` (parallel execution)
- NeMo-RL: `max_parallel_calls` configuration

---

## Section 2: Appendix Items

### A. Generation Arguments

**What to include:**
- `stop_strings` - List of strings to stop generation
- `stop_token_ids` - List of token IDs to stop generation
- `temperature`, `top_p` - Sampling parameters
- `max_tokens` - Maximum generation length

**Where it goes:** Appendix section on generation configuration

```python
# Example generation call with all parameters:
response = await policy.generate.route(
    prompt,
    n=1,
    stop_strings=["</tool_call>", "<|im_end|>"],
    stop_token_ids=[tokenizer.eos_token_id],
    temperature=0.7,
    top_p=0.95,
    max_tokens=512,
)
```

**References:**
- NeMo-RL: `RL/nemo_rl/models/generation/interfaces.py:127-128`
- NeMo-RL: `RL/nemo_rl/experience/rollouts.py:280,291` (next_stop_strings)

---

### B. vLLM Configuration Flags

**What to include:**
- `enable_auto_tool_choice` - Enable native tool calling
- `tool_call_parser` - Tool format parser (hermes/mistral/llama)
- `enable_prefix_caching` - Cache prompt prefixes (helps multi-turn)

**Where it goes:** Appendix section on vLLM setup

```python
# In Generator initialization:
policy = Generator(
    model="Qwen/Qwen2.5-7B-Instruct",
    engine_args={
        # Tool calling support
        "enable_auto_tool_choice": True,
        "tool_call_parser": "hermes",

        # Performance
        "enable_prefix_caching": True,
        "gpu_memory_utilization": 0.9,
        "max_model_len": 4096,
    }
)
```

**References:**
- PRIME-RL: `prime-rl/examples/wiki_search/rl.toml`
- NeMo-RL: `async_engine: true` for pipelining

---

### C. Episode Metadata Fields (Full List)

**Complete metadata dictionary:**

```python
metadata = {
    # Basic stats
    "num_turns": turn,
    "num_tool_calls": tool_call_count,

    # Termination
    "truncated": turn >= max_turns,
    "termination_reason": "max_turns" | "done" | "error",

    # Performance
    "total_tokens": len(all_tokens),
    "prompt_tokens": len(prompt_ids),
    "response_tokens": len(all_tokens),

    # Task info
    "task_type": task.task_type,
    "env_name": env_client.name,
}
```

**References:**
- NeMo-RL: `RL/nemo_rl/experience/rollouts.py:512,523-526`
- Tinker: `Transition.metrics`

---

## Section 3: Open Questions

### Q1: Attention Mask & Position IDs

**Question:** Do we need explicit `attention_mask` and `position_ids` fields in Episode?

**Context from frameworks:**
- VERL includes `attention_mask`, `position_ids` in batch dict
- NeMo-RL has full batch preparation with these fields

**Clarification needed:**
1. Does Forge's current Episode → batch conversion handle these automatically?
2. Are they required for training, or does the trainer build them?
3. For multi-turn with concatenated tokens, do we need special handling?

**Potential answer:** If needed, they can be computed from token IDs:
- `attention_mask`: 1 for real tokens, 0 for padding
- `position_ids`: Sequential positions for all tokens

**References:**
- VERL: `verl/workers/rollout/sglang_rollout.py` (batch dict construction)
- NeMo-RL: `RL/nemo_rl/experience/rollouts.py` (batch preparation)

---

## Summary

**To add to main 3_5 loop:**
1. ✅ Multi-environment routing (env_map, task_type)
2. ✅ Tool call parsing (parse_tool_call with format note)
3. ✅ Tool response truncation (truncate() utility with TODO)
4. ✅ Parallel episode collection (TODO for asyncio.gather)
5. ✅ Episode metadata (minimal fields, full list in appendix)
6. ✅ System prompt (clarify dataset vs tokenizer)
7. ✅ Response mask API (pass to trainer)
8. ✅ Error handling (try/except around env.step)
9. ✅ Parallel tool execution (with TODO for env support)

**To add to appendix:**
- Generation arguments (stop_strings, temperature, etc.)
- vLLM configuration flags
- Full metadata fields

**Requires clarification:**
- Attention mask & position IDs necessity
