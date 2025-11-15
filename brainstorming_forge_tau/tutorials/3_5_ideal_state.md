# Part 3.5: Ideal State - Multi-Turn Tool Calling with Forge + vLLM + OpenEnv

For tool calling, we extend Forge's GRPO pattern to handle **multi-turn interactions** where:
- One task → multiple LLM generations + tool executions → one Episode
- Episode contains **concatenated tokens** from all turns
- Training and replay buffer logic remains unchanged

**Key Principle:** Multi-turn only changes the **rollout phase**. Training stays the same.

---

## Setup: Services + Multi-Environment Support

```python
# Reference: Adapted from apps/grpo/main.py for multi-turn
# OpenEnv RFC 001: "We separate tasks from environments"

# 1. Setup services (same as single-turn, plus environments)
policy = Generator(...)
trainer = TitanTrainer(...)
replay_buffer = ReplayBuffer(...)
ref_model = ReferenceModel(...)

# Dataloader provides tasks (prompts + metadata)
# Reference: OpenEnv/rfcs/001-abstractions.md:308-381
dataloader = DataLoader(Tau2BenchDataset(...))

# NEW: Environment map for multiple task types
# Different environments = different tools, max_turns, rewards
# Reference: verifiers/envs/env_group.py:218-266 (task-based routing)
env_map = {
    "websearch": WebSearchEnv.from_docker_image("tau2bench/websearch:latest"),
    "coding": CodingEnv.from_docker_image("tau2bench/coding:latest"),
    "airline": AirlineEnv.from_docker_image("tau2bench/airline:latest"),
}

# Environment-specific configuration
max_turns_config = {
    "websearch": 10,
    "coding": 15,
    "airline": 8,
}
```

**Why environment map?** Tau2Bench has multiple domains with different tools. Tasks include a `task_type` field to route to the correct environment.

**References:**
- Verifiers: `verifiers/envs/env_group.py` (EnvGroup pattern)
- Tinker: `tinker-cookbook/distillation/datasets.py:45-83` (CompositeDataset)

---

## Rollout Loop: Multi-Turn with Environment Routing

```python
# 2. Rollout loop (continuous_rollouts with multi-turn)
async def continuous_rollouts():
    while True:
        # Sample task from dataloader
        task = await dataloader.sample.call_one()
        # task.prompt: "Book a flight from SF to NYC on March 15th"
        # task.task_type: "websearch" | "coding" | "airline"
        # task.metadata: Additional task-specific info

        # Route to correct environment based on task type
        env_client = env_map[task.task_type]
        max_turns = max_turns_config[task.task_type]

        # Reset environment to get tools (env doesn't know the task)
        # Reference: OpenEnv/src/core/http_env_client.py:142-154
        env_state = env_client.reset()
        tool_schemas = env_state.observation.tools  # Available tools for this env

        # Generate G samples for this task
        # TODO: Investigate parallelizing with asyncio.gather() instead of sequential
        episodes = []
        for _ in range(group_size):  # G samples per task
            episode = await play_task(
                policy=policy,
                task_prompt=task.prompt,  # From dataloader
                tool_schemas=tool_schemas,  # From environment
                env=env_client,
                max_turns=max_turns
            )
            episodes.append(episode)

        # Add to replay buffer (same as single-turn)
        for episode in episodes:
            await replay_buffer.add.call_one(episode)
```

**Key differences from single-turn:**

| Aspect | Single-Turn (GSM8K) | Multi-Turn (Tau2Bench) |
|--------|---------------------|------------------------|
| **Dataloader** | ✅ `DataLoader(GSM8K)` | ✅ `DataLoader(Tau2Bench)` |
| **Task routing** | N/A | `env_map[task.task_type]` |
| **Environment** | None | `env.reset()` provides tools |
| **Generation** | One `policy.generate()` | Loop of `policy.generate()` calls |
| **Actions** | None | `env.step(ToolCallAction)` |
| **Episode tokens** | `response.token_ids` | Concatenated: `llm + tool + llm + ...` |
| **Reward** | `reward_actor.evaluate()` | `env.step().reward` |

**Critical insight:** Dataset provides tasks, environment provides tools. They are separate.

---

## Multi-Turn Rollout: play_task()

This replaces the single `policy.generate()` call in single-turn GRPO.

```python
# Reference: OpenEnv/src/core/client_types.py (StepResult)
from openenv.core.client_types import StepResult
from openenv.core.env_server import ToolCallAction

async def play_task(
    policy: Generator,
    task_prompt: str,  # From dataloader
    tool_schemas: list[dict],  # From env.reset()
    env: OpenEnvClient,
    max_turns: int = 10
) -> Episode:
    """
    Play one task to completion, return single Episode.

    Args:
        policy: Generator actor for LLM generation
        task_prompt: Task from dataloader (e.g., "Book flight SF->NYC")
        tool_schemas: Available tools from env.reset()
        env: Environment client for tool execution
        max_turns: Maximum conversation turns

    Returns:
        Episode with all turns concatenated
    """

    # Initialize conversation with task
    # System prompt handled by tokenizer.apply_chat_template() with tools=
    # Or dataset can provide task.system_prompt if needed
    messages = [{"role": "user", "content": task_prompt}]

    # Storage: concatenate all turns into single sequence
    all_tokens = []
    all_logprobs = []
    response_mask = []  # 1=train on LLM output, 0=skip tool results
    metadata = {}  # Track episode stats

    done = False
    turn = 0

    while not done and turn < max_turns:
        # 1. Format prompt with conversation history + tools
        # Tokenizer injects system prompt with tool definitions when tools= is passed
        prompt = tokenizer.apply_chat_template(
            messages,
            tools=tool_schemas,  # From env.reset()
            add_generation_prompt=True,
            tokenize=False
        )

        # 2. Generate response
        response = await policy.generate.route(prompt, n=1)

        # 3. Parse tool call from response
        # Using Tinker pattern: XML tags <tool_call>...</tool_call>
        # Alternative: vLLM native parsing with tool_call_parser="hermes" (see Appendix)
        tool_calls = parse_tool_calls(response.text)  # Returns list of tool calls

        if tool_calls:
            # Tool execution path
            # Add assistant message with tool calls
            messages.append({
                "role": "assistant",
                "content": response.text,
                "tool_calls": tool_calls  # Structured tool call data
            })

            # Collect LLM output tokens - TRAIN on these
            all_tokens.extend(response.token_ids)
            all_logprobs.extend(response.logprobs)
            response_mask.extend([1] * len(response.token_ids))

            # Execute tools (parallel if multiple calls)
            # TODO: Confirm environment can handle parallel requests
            try:
                tool_tasks = [
                    env.execute_tool(tc["name"], tc["args"])
                    for tc in tool_calls
                ]
                tool_results = await asyncio.gather(*tool_tasks)
            except Exception as e:
                # Handle tool execution errors
                tool_results = [{"content": f"Error: {str(e)}"}]

            # Add tool results to messages and tokens
            for tool_result in tool_results:
                tool_content = tool_result.content

                # Truncate long tool responses to avoid context overflow
                tool_tokens = tokenizer.encode(tool_content, add_special_tokens=False)
                tool_tokens = truncate(tool_tokens, max_length=256)
                # TODO: Decide where truncate() lives (env vs rollout loop vs utility)
                tool_content = tokenizer.decode(tool_tokens)

                # Add tool result to messages
                messages.append({
                    "role": "tool",
                    "content": tool_content
                })

                # Collect tool result tokens - DON'T TRAIN on these
                all_tokens.extend(tool_tokens)
                all_logprobs.extend([0.0] * len(tool_tokens))
                response_mask.extend([0] * len(tool_tokens))

            # Check if environment signals done
            done = tool_results[-1].get("done", False) if tool_results else False

        else:
            # Final answer (no tool call)
            messages.append({
                "role": "assistant",
                "content": response.text
            })

            # Collect final response tokens - TRAIN on these
            all_tokens.extend(response.token_ids)
            all_logprobs.extend(response.logprobs)
            response_mask.extend([1] * len(response.token_ids))

            done = True

        turn += 1

    # Populate episode metadata
    metadata = {
        "num_turns": turn,
        "truncated": turn >= max_turns,
        # See Appendix for full metadata fields
    }

    # Get final reward from environment
    # In single-turn: reward_actor.evaluate_response()
    # In multi-turn: environment state
    final_reward = env.get_reward()  # 1.0 or 0.0

    # Create Episode (same structure as single-turn)
    # Reference: apps/grpo/main.py:44-75
    completion = Completion(
        prompt_ids=None,  # Not stored (can reconstruct from messages)
        token_ids=torch.tensor(all_tokens),
        logprobs=torch.tensor(all_logprobs),
        text=tokenizer.decode(all_tokens),
        generator_version=0
    )

    episode = Episode(
        episode_id=str(uuid.uuid4()),
        pad_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        request_len=0,  # Varies per turn, not fixed
        response_len=len(all_tokens),
        target=None,  # Tau2Bench doesn't expose ground truth during training
        completion=completion,
        response_mask=torch.tensor(response_mask),  # NEW: Mask for training
        ref_logprobs=None,  # Computed later by ref_model
        reward=final_reward,
        advantage=None,  # Computed later with group
        metadata=metadata  # NEW: Episode statistics
    )

    return episode
```

**Key details:**

1. **Tool call parsing:** Uses `parse_tool_calls()` to extract tool calls from text. Can use vLLM native parsing (see Appendix).

2. **Response mask:** Critical for multi-turn. Marks which tokens to train on:
   - `1` = LLM output (train on these)
   - `0` = Tool results (don't train on these)

3. **Truncation:** Long tool responses truncated to avoid exceeding context limits.

4. **Error handling:** Tool execution wrapped in try/except. Errors added as tool messages.

5. **Parallel tools:** Multiple tool calls in single response executed concurrently with `asyncio.gather()`.

6. **Metadata:** Track episode stats (num_turns, truncation, etc.) for analysis.

**References:**
- Tinker: `tinker-cookbook/recipes/tool_use/search/search_env.py` (multi-turn loop)
- VERL: `verl/experimental/agent_loop/tool_agent_loop.py` (parallel tools, truncation)
- TRL: `trl/examples/scripts/openenv/catch.py` (token concatenation)

---

## Training Loop: Response Mask Integration

```python
# Reference: apps/grpo/main.py

# 3. Training loop (minimal changes - just add response_mask)
async def continuous_training():
    while True:
        # Sample batch from replay buffer
        batch = await replay_buffer.sample(batch_size)

        # Get reference logprobs
        ref_logprobs = await ref_model.forward.route(
            prompt_ids=batch["prompt_ids"],
            response_ids=batch["response_ids"]
        )

        # Compute advantages (group-relative)
        advantages = compute_group_advantages(batch["rewards"])

        # Train on batch with response mask
        await trainer.train_step(
            inputs=batch["prompt_ids"],
            targets=batch["response_ids"],
            advantages=advantages,
            ref_logprobs=ref_logprobs,
            response_mask=batch["response_mask"],  # NEW: Mask tool results
        )

        # Update policy weights
        version = await trainer.push_weights()
        await policy.update_weights(version)
```

**What changed:** Added `response_mask` parameter to `trainer.train_step()`. The trainer applies the mask during loss computation to zero out gradients for tool result tokens.

**References:**
- VERL: `verl/trainer/ppo/core_algos.py:787-808` (masked loss aggregation)
- Verifiers: `mask_env_responses` flag in processing

---

## Complete Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    SINGLE-TURN (GSM8K)                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  dataloader.sample()  ──→  task.prompt                      │
│       ↓                                                     │
│  policy.generate(task.prompt, n=G)  ──→  [responses 1..G]  │
│       ↓                                                     │
│  create Episode(response)                                   │
│       ↓                                                     │
│  replay_buffer.add(episode)                                 │
│                                                             │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                   MULTI-TURN (TAU2BENCH)                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  dataloader.sample()  ──→  task (prompt + task_type)        │
│       ↓                                                     │
│  env = env_map[task.task_type]  ──→  route to environment  │
│  env.reset()  ──→  tool_schemas                             │
│       ↓                                                     │
│  FOR i in 1..G:                                             │
│    play_task(task.prompt, tool_schemas, env):               │
│      messages = [user: task.prompt]                         │
│      WHILE not done AND turn < max_turns:                   │
│        prompt = apply_chat_template(messages, tools)        │
│        response = policy.generate(prompt)                   │
│        tool_calls = parse_tool_calls(response)              │
│        IF tool_calls:                                       │
│          results = asyncio.gather(*[env.execute_tool(...)])│
│          messages.append(assistant, tool_results)           │
│          all_tokens += [llm_tokens] + [tool_tokens]         │
│          response_mask += [1, 1, ...] + [0, 0, ...]         │
│        ELSE:                                                │
│          done = True                                        │
│        turn += 1                                            │
│      create Episode(all_tokens, response_mask, reward)      │
│       ↓                                                     │
│  replay_buffer.add(episode)                                 │
│       ↓                                                     │
│  trainer.train_step(..., response_mask=mask)                │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Key components:**
- **Task routing:** `env_map[task.task_type]` selects environment
- **Tool schemas:** From `env.reset()`, passed to tokenizer
- **Token concatenation:** All turns merged into single sequence
- **Response mask:** Separates LLM output (train) from tool results (skip)
- **Training:** Same GRPO logic, just with mask applied

---

## Appendix

### A. Generation Arguments

Full parameter list for `policy.generate.route()`:

```python
response = await policy.generate.route(
    prompt,
    n=1,
    # Stop conditions
    stop_strings=["</tool_call>", "<|im_end|>"],
    stop_token_ids=[tokenizer.eos_token_id],
    # Sampling
    temperature=0.7,
    top_p=0.95,
    max_tokens=512,
)
```

**References:**
- NeMo-RL: `RL/nemo_rl/models/generation/interfaces.py:127-128`
- NeMo-RL: `RL/nemo_rl/experience/rollouts.py:280,291` (dynamic stop strings)

---

### B. vLLM Configuration Flags

Enable native tool calling and performance optimizations:

```python
policy = Generator(
    model="Qwen/Qwen2.5-7B-Instruct",
    engine_args={
        # Tool calling support (alternative to text parsing)
        "enable_auto_tool_choice": True,
        "tool_call_parser": "hermes",  # or "mistral", "llama"

        # Performance
        "enable_prefix_caching": True,  # Cache prompt prefixes (helps multi-turn!)
        "gpu_memory_utilization": 0.9,
        "max_model_len": 4096,
    }
)
```

**What these do:**
- `enable_auto_tool_choice`: vLLM parses tool calls from model output automatically
- `tool_call_parser`: Format parser (model-specific)
- `enable_prefix_caching`: Reuses cached prompts across turns (major speedup!)

**References:**
- PRIME-RL: `prime-rl/examples/wiki_search/rl.toml`
- NeMo-RL: `async_engine: true` for pipelining

---

### C. Episode Metadata (Full Fields)

Complete metadata dictionary for debugging and analysis:

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
    "prompt_tokens": sum(len(m["content"]) for m in messages if m["role"] != "assistant"),
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

### D. Tool Call Parsing Formats

**Tinker pattern (XML tags):**
```python
def parse_tool_calls(response_text: str) -> list[dict]:
    """Parse tool calls from <tool_call>...</tool_call> tags."""
    matches = re.findall(r"<tool_call>(.*?)</tool_call>", response_text, re.DOTALL)
    tool_calls = []
    for match in matches:
        try:
            tool_calls.append(json.loads(match))
        except json.JSONDecodeError:
            continue
    return tool_calls
```

**vLLM native (Hermes format):**
```python
# If enable_auto_tool_choice=True, response has structured tool_calls
if hasattr(response, 'tool_calls') and response.tool_calls:
    return [
        {
            "name": tc.name,
            "args": json.loads(tc.arguments)
        }
        for tc in response.tool_calls
    ]
```

**References:**
- Tinker: `tinker-cookbook/recipes/tool_use/search/search_env.py`
- PRIME-RL: Uses vLLM native parsing

---

### E. System Prompt Options

**Option 1: Dataset provides system prompt**
```python
# Task includes system_prompt field
messages = [
    {"role": "system", "content": task.system_prompt},
    {"role": "user", "content": task_prompt}
]
```

**Option 2: Tokenizer injects system prompt**
```python
# Tokenizer handles system prompt when tools= is passed
messages = [{"role": "user", "content": task_prompt}]
prompt = tokenizer.apply_chat_template(
    messages,
    tools=tool_schemas,  # Tokenizer adds system message with tool definitions
    add_generation_prompt=True,
    tokenize=False
)
```

**Recommendation:** Use Option 2 if your tokenizer supports it. Otherwise, have dataset provide system prompts per task type.

---

## Summary: What Changed for Multi-Turn

| Component | Single-Turn | Multi-Turn |
|-----------|-------------|------------|
| **Setup** | `env_client` (single) | `env_map` (multiple envs) |
| **Rollout** | `policy.generate()` once | `play_task()` with loop |
| **Episode tokens** | `response.token_ids` | Concatenated across turns |
| **Episode fields** | Basic | + `response_mask`, `metadata` |
| **Training** | `train_step(...)` | + `response_mask` parameter |

**Everything else stays the same:** Replay buffer, reference model, advantage computation, weight updates.
