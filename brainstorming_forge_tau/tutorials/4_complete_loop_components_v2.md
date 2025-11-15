# Part 4: Complete Multi-Turn Tool Calling Loop (Components)

This part breaks down all components needed for multi-turn tool calling

## 4.1 Overview: Multi-Turn Tool Calling in Forge

This shows how multi-turn tool calling extends Forge's current GRPO architecture.

### Current Forge GRPO Flow (Single-Turn)

```python
# Reference: apps/grpo/main.py

# 1. Setup services (distributed actors via Monarch)
policy = Generator(...)              # vLLM-based generation
trainer = TitanTrainer(...)          # Training service
replay_buffer = ReplayBuffer(...)    # Store episodes
ref_model = ReferenceModel(...)      # Reference for KL
reward_actor = RewardActor(...)      # Score responses

# 2. Rollout loop (continuous_rollouts)
async def continuous_rollouts():
    while True:
        # Sample prompt from dataset
        sample = await dataloader.sample.call_one()
        prompt, target = sample["prompt"], sample["target"]

        # Generate G responses (group)
        responses = await policy.generate.route(
            prompt,
            n=group_size  # e.g., 8 responses
        )

        # Score and create episodes
        episodes = []
        for response in responses:
            episode = Episode(
                prompt_ids=response.prompt_ids,
                completion=response,
                reward=compute_reward(response.text, target),
                ...
            )
            episodes.append(episode)

        # Add to replay buffer
        for episode in episodes:
            await replay_buffer.add.call_one(episode)
```

**Key property**: One prompt → one response → one Episode (single-turn)

---

### Multi-Turn Extension: Tool Calling with OpenEnv

For tool calling, we extend this pattern to handle **multi-turn interactions** where:
- One task → multiple LLM generations + tool executions → one Episode
- Episode contains **concatenated tokens** from all turns

**Note on Multiple Environments**: Tau2Bench has multiple domains (airline, retail, etc.). See Section 4.9 for how to handle training on mixed environments with different tools, max_turns, and rewards per domain.

```python
# Reference: Adapted from apps/grpo/main.py for multi-turn
# OpenEnv RFC 001: "We separate tasks from environments"

# 1. Setup services (same as before, plus environment)
policy = Generator(...)
trainer = TitanTrainer(...)
replay_buffer = ReplayBuffer(...)
ref_model = ReferenceModel(...)

# STILL HAVE DATALOADER!
# Reference: OpenEnv/rfcs/001-abstractions.md:308-381 (TaskDataset)
dataloader = DataLoader(Tau2BenchDataset(...))

# NEW: Environment client for tool execution
# OpenEnv runs in Docker, provides tools/execution/rewards
# NOTE: For multiple domains, see Section 4.9 (CompositeDataset pattern)
env_client = Tau2BenchEnv.from_docker_image("tau2bench/airline:latest")

# 2. Rollout loop (continuous_rollouts with multi-turn)
async def continuous_rollouts():
    while True:
        # --- SAME: Sample task from dataloader ---
        # Reference: OpenEnv RFC 001: "when training, it comes from a dataset"
        task = await dataloader.sample.call_one()
        # task.prompt: "Book a flight from SF to NYC on March 15th"
        # task.ground_truth: Expected outcome for eval
        # task.metadata: Any task-specific info

        # --- NEW: Reset environment (doesn't know the task) ---
        # Reference: OpenEnv/src/core/http_env_client.py:142-154
        # Environment provides tools, NOT the task description
        env_state = env_client.reset()
        tool_schemas = env_state.observation.tools  # Available tools

        # --- DIFFERENCE: Multi-turn rollout (play_task) ---
        # Generate G samples for this task
        episodes = []
        for _ in range(group_size):  # G samples per task
            episode = await play_task(
                policy=policy,
                task_prompt=task.prompt,  # From dataloader
                tool_schemas=tool_schemas,  # From environment
                env=env_client,
                max_turns=10
            )
            episodes.append(episode)

        # --- SAME: Add to replay buffer ---
        for episode in episodes:
            await replay_buffer.add.call_one(episode)
```

**Key differences from single-turn:**

| Aspect | Single-Turn (GSM8K) | Multi-Turn (Tau2Bench) |
|--------|---------------------|------------------------|
| **Dataloader** | ✅ `DataLoader(GSM8K)` | ✅ `DataLoader(Tau2Bench)` (still there!) |
| **Task source** | `task.prompt` | `task.prompt` (same!) |
| **Environment** | None | `env.reset()` provides tools |
| **Generation** | One `policy.generate()` | Loop of `policy.generate()` calls |
| **Actions** | None | `env.step(ToolCallAction)` for tools |
| **Episode tokens** | `response.token_ids` | Concatenated: `llm + tool + llm + ...` |
| **Reward source** | `reward_actor.evaluate(task.ground_truth)` | `env.step().reward` |
| **Multiple domains** | N/A | See Section 4.9 for mixing airline/retail/etc. |

**Critical insight from OpenEnv RFC 001**:
- "We separate tasks from environments" (line 68)
- "when training/testing, it comes from a dataset" (line 30)
- Dataset provides: task prompts, ground truth for eval
- Environment provides: tools, execution, rewards

---

### Multi-Turn Rollout (play_task)

This replaces the single `policy.generate()` call in single-turn GRPO.

```python
# Reference: OpenEnv/src/core/client_types.py (StepResult), RFC 004 (ToolCallAction)
from openenv.core.client_types import StepResult
from openenv.core.env_server import ToolCallAction

async def play_task(
    policy: Generator,
    task_prompt: str,  # From dataloader
    tool_schemas: list[dict],  # From env.reset()
    env: Tau2BenchEnv,
    max_turns: int = 10
) -> Episode:
    """
    Play one task to completion, return single Episode.

    Args:
        policy: Generator actor for LLM generation
        task_prompt: Task description from dataloader (e.g., "Book flight SF->NYC")
        tool_schemas: Available tools from env.reset()
        env: Environment client for tool execution
        max_turns: Maximum conversation turns

    Replaces: single policy.generate() call
    Returns: Episode with all turns concatenated
    """

    # Initialize messages with task from dataloader
    messages = [{"role": "user", "content": task_prompt}]

    # Storage: concatenate all turns into single sequence
    all_tokens = []
    all_logprobs = []
    response_mask = []  # 1=train, 0=skip

    done = False
    turn = 0

    while not done and turn < max_turns:
        # 1. Format prompt with full history
        prompt = tokenizer.apply_chat_template(
            messages,
            tools=tool_schemas,  # From env.reset()
            add_generation_prompt=True,
            tokenize=False
        )

        # 2. Generate (SAME as single-turn)
        response = await policy.generate.route(prompt, n=1)

        # 3. Parse tool call
        tool_call = parse_tool_call(response.text)

        if tool_call:
            # Tool execution path
            # 4. Execute via environment
            action = ToolCallAction(
                tool_name=tool_call["name"],
                parameters=tool_call["args"]
            )
            result = env.step(action)  # HTTP call to OpenEnv server

            # 5. Update messages
            messages.append({"role": "assistant", "content": response.text})
            messages.append({"role": "tool", "content": result.observation.content})

            # 6. Collect tokens
            # LLM output - TRAIN
            all_tokens.extend(response.token_ids)
            all_logprobs.extend(response.logprobs)
            response_mask.extend([1] * len(response.token_ids))

            # Tool result - DON'T TRAIN
            tool_tokens = tokenizer.encode(result.observation.content, add_special_tokens=False)
            all_tokens.extend(tool_tokens)
            all_logprobs.extend([0.0] * len(tool_tokens))
            response_mask.extend([0] * len(tool_tokens))

            done = result.done
        else:
            # Final answer
            messages.append({"role": "assistant", "content": response.text})

            all_tokens.extend(response.token_ids)
            all_logprobs.extend(response.logprobs)
            response_mask.extend([1] * len(response.token_ids))

            done = True

        turn += 1

    # 7. Get reward from environment
    # NOTE: In single-turn, reward comes from reward_actor.evaluate_response()
    # In multi-turn, reward comes from environment state
    final_reward = result.reward  # 1.0 or 0.0

    # 8. Create Episode (SAME structure as single-turn)
    # Reference: apps/grpo/main.py:44-75
    completion = Completion(
        prompt_ids=torch.tensor(prompt_ids),
        token_ids=torch.tensor(all_tokens),
        logprobs=torch.tensor(all_logprobs),
        text=tokenizer.decode(all_tokens),
        generator_version=0
    )

    episode = Episode(
        episode_id=str(uuid.uuid4()),
        pad_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        request_len=len(prompt_ids),
        response_len=len(all_tokens),
        target=None,  # Tau2Bench doesn't expose ground truth
        completion=completion,
        ref_logprobs=None,  # Computed later by ref_model
        reward=final_reward,
        advantage=None  # Computed later with group
    )

    return episode
```

**Comparison to single-turn:**

| Aspect | Single-Turn (GSM8K) | Multi-Turn (Tau2Bench) |
|--------|---------------------|------------------------|
| **Prompt source** | `dataloader.sample()` | `env.reset()` |
| **Generation** | One `policy.generate()` | Loop of `policy.generate()` calls |
| **Actions** | None (just generate text) | `env.step(ToolCallAction)` |
| **Episode tokens** | `response.token_ids` | Concatenated: `llm_tokens + tool_tokens + llm_tokens + ...` |
| **Reward source** | `reward_actor.evaluate_response()` | `env.step().reward` |
| **Episode structure** | Same `Episode` object | Same `Episode` object |

**Key insight**: Multi-turn just extends the **rollout** phase. Training, replay buffer, and everything else stays the same.

---

### Complete Flow Diagram

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
│  dataloader.sample()  ──→  task.prompt                      │
│  env.reset()  ──→  tool_schemas                             │
│       ↓                                                     │
│  FOR i in 1..G:                                             │
│    play_task(task.prompt, tool_schemas):                    │
│      messages = [user: task.prompt]                         │
│      WHILE not done:                                        │
│        policy.generate(messages)  ──→  response             │
│        IF tool_call:                                        │
│          env.step(action)  ──→  tool_result                 │
│          messages.append(response, tool_result)             │
│        ELSE:                                                │
│          done = True                                        │
│      create Episode(all_tokens, env.reward)                 │
│       ↓                                                     │
│  replay_buffer.add(episode)                                 │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Key components:**
- **Dataloader**: Still samples tasks in both cases
- **Environment**: New in multi-turn, provides tools + execution + rewards
- **play_task**: Combines task.prompt (dataloader) + tool_schemas (env)

---

### Training Loop (No Changes)

```python
# Reference: apps/grpo/main.py

# 3. Training loop (SAME as single-turn)
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

        # Train on batch
        await trainer.train_step(
            inputs=batch["prompt_ids"],
            targets=batch["response_ids"],
            advantages=advantages,
            ref_logprobs=ref_logprobs
        )

        # Update policy weights
        version = await trainer.push_weights()
        await policy.update_weights(version)
```

**No changes needed**: Training doesn't care if Episode came from single-turn or multi-turn. It just sees token sequences.

---

### Summary

**What changes for multi-turn tool calling:**
1. ✅ **Add Environment**: `env.reset()` to get tool schemas, `env.step()` for execution
2. ✅ **Rollout**: Replace `policy.generate()` with `play_task()` loop
3. ✅ **Reward source**: `env.step().reward` instead of `reward_actor.evaluate()`

**What stays the same:**
1. ✅ **Dataloader**: Still samples tasks from dataset (`task.prompt`, `task.ground_truth`)
2. ✅ **Services**: Generator, Trainer, ReplayBuffer, RefModel
3. ✅ **Episode structure**: Same `Episode` dataclass
4. ✅ **Training loop**: Same GRPO algorithm
5. ✅ **Infrastructure**: Same Monarch actors

**Separation of concerns (OpenEnv RFC 001)**:
- **Dataloader**: Provides task prompts and ground truth
- **Environment**: Provides tools, execution sandbox, and rewards
- **Agent/Policy**: Manages conversation history, tokenization, generation

**The pattern is extensible**:
- Single-turn = special case where `play_task()` does 1 iteration
- Multi-turn = generalization where `play_task()` does N iterations

Let's break down each component in detail below.

## 4.2 Component 1: Episode Initialization and Prompt Formatting

### How Tasks and Environments Work

**Key Concept:** The dataset/task and environment are separate:
- **Dataset**: Contains task descriptions (tickets, questions, etc.)
- **Environment**: Provides tool execution, state management, and rewards

**Pattern:**
```python
# 1. Load dataset
dataset = load_dataset("tau2bench/airline")
task = dataset[0]  # {"ticket": "...", "tools": [...], "target": "..."}

# 2. Create environment (knows tools, not the specific task)
env = Tau2Env(domain="airline")

# 3. Initialize episode with task
result = env.reset(task_id=task["id"])
```

### Concrete Example: Same Task, Three Approaches

We'll use this example task across all approaches:

**Task:**
```python
task = {
    "ticket": "Book a flight from SF to NYC on March 15th",
    "tools": [
        {
            "name": "search_flights",
            "description": "Search for available flights",
            "parameters": {
                "type": "object",
                "properties": {
                    "origin": {"type": "string"},
                    "destination": {"type": "string"},
                    "date": {"type": "string"}
                },
                "required": ["origin", "destination", "date"]
            }
        },
        {
            "name": "book_flight",
            "description": "Book a specific flight",
            "parameters": {
                "type": "object",
                "properties": {
                    "flight_id": {"type": "string"}
                },
                "required": ["flight_id"]
            }
        }
    ]
}
```

---

### Option A: vLLM Native (tokenizer.apply_chat_template)

**Where does the template come from?**
The tokenizer contains a Jinja2 template file that defines how to format messages and tools.

**Example for Qwen:**
```python
# Reference: Qwen tokenizer includes tokenizer_config.json with chat_template field
# The template is a Jinja2 string like:
# "{% for message in messages %}..."

from vllm.transformers_utils.tokenizer import get_tokenizer

# 1. Load tokenizer (contains Jinja2 template)
tokenizer = get_tokenizer("Qwen/Qwen2.5-1.5B-Instruct")

# 2. Build messages
messages = [
    {"role": "user", "content": task["ticket"]}
]

# 3. Apply template (Jinja2 renders messages + tools)
prompt_text = tokenizer.apply_chat_template(
    messages,
    tools=task["tools"],  # Tools injected into template
    add_generation_prompt=True,
    tokenize=False
)

# 4. Tokenize
prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=True)
```

**What `prompt_text` looks like (Qwen format):**
```
<|im_start|>system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"name": "search_flights", "description": "Search for available flights", "parameters": {...}}
{"name": "book_flight", "description": "Book a specific flight", "parameters": {...}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call><|im_end|>
<|im_start|>user
Book a flight from SF to NYC on March 15th<|im_end|>
<|im_start|>assistant
```

**How it works:**
- Tokenizer's Jinja2 template formats messages + tools automatically
- Model-specific (Qwen format shown above; Llama3 would be different)
- Used by: Forge, VERL, PrimeRL

---

### Option B: Manual System Prompt + Renderer (Thinker)

**Where does the template come from?**
You define the system prompt manually, then use a Renderer to apply the model's chat format.

```python
# Reference: tinker-cookbook/tinker_cookbook/recipes/tool_use/search/search_env.py:33-76
from tinker_cookbook.renderers import Qwen3Renderer
from vllm.transformers_utils.tokenizer import get_tokenizer

# 1. Define system prompt template (you control this)
SYSTEM_PROMPT = """You are an expert assistant who solves tasks using tools.

Available tools:
{tool_descriptions}

Use format: <tool_call>{{"name": "tool_name", "args": {{...}}}}</tool_call>"""

# 2. Format tool descriptions
tool_descriptions = "\n".join([
    f"- {tool['name']}: {tool['description']}"
    for tool in task["tools"]
])
system_content = SYSTEM_PROMPT.format(tool_descriptions=tool_descriptions)

# 3. Build messages
messages = [
    {"role": "system", "content": system_content},
    {"role": "user", "content": task["ticket"]}
]

# 4. Use Renderer to apply Qwen's chat format
tokenizer = get_tokenizer("Qwen/Qwen2.5-1.5B-Instruct")
renderer = Qwen3Renderer(tokenizer)
model_input = renderer.build_generation_prompt(messages)
prompt_ids = model_input.tokens  # Already tokenized
```

**What the formatted prompt looks like (via Renderer):**
```
<|im_start|>system
You are an expert assistant who solves tasks using tools.

Available tools:
- search_flights: Search for available flights
- book_flight: Book a specific flight

Use format: <tool_call>{"name": "tool_name", "args": {...}}</tool_call><|im_end|>
<|im_start|>user
Book a flight from SF to NYC on March 15th<|im_end|>
<|im_start|>assistant
```

**How it works:**
- You manually format tool descriptions into system prompt
- Renderer applies model-specific chat template (Qwen format shown)
- Reference: `tinker_cookbook.renderers.Qwen3Renderer._render_message` (lines 333-358)
- Used by: Thinker, Verifiers

---

### Option C: Environment-Provided Template

**Where does the template come from?**
The environment or task definition provides the system prompt.

```python
# Reference: How Tau2Bench or Thinker datasets might work

# 1. Task includes pre-formatted system prompt
task = {
    "ticket": "Book a flight from SF to NYC on March 15th",
    "system_prompt": "You are a travel booking assistant...",  # Pre-defined
    "tools": [...]
}

# 2. Or environment provides system prompt
from tinker_cookbook.recipes.tool_use.search import SearchEnv

env = SearchEnv(
    problem=task["ticket"],
    answer=task["target"],
    tool_client=tool_client,
    renderer=renderer
)

# Environment's initial_observation includes formatted prompt
observation, stop_condition = await env.initial_observation()
prompt_ids = observation.tokens  # Already includes system + user message
```

**What the environment does internally:**
```python
# Reference: tinker-cookbook/.../search_env.py:122-127
class SearchEnv:
    async def initial_observation(self):
        # Environment builds messages with its own system prompt
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},  # Env-defined
            {"role": "user", "content": self.problem}
        ]
        return self.renderer.build_generation_prompt(messages), stop_condition
```

**How it works:**
- Environment encapsulates system prompt logic
- Cleaner for researchers (don't worry about prompts)
- Used by: Thinker's environments

---

### Comparison Table

| Approach | Template Source | Tool Schema Location | Formatting | Who Manages Prompt |
|----------|----------------|----------------------|------------|-------------------|
| **Option A: vLLM Native** | Tokenizer's Jinja2 file | `tools=...` param | Tokenizer | You call `apply_chat_template` |
| **Option B: Manual + Renderer** | You define SYSTEM_PROMPT | System message | Renderer class | You build messages |
| **Option C: Environment** | Environment class | Environment config | Renderer (inside env) | Environment |

**Recommendation:**
- **Option A** for production (if tokenizer supports tools)
- **Option B** for research/flexibility (Thinker's approach)
- **Option C** for clean experiment code (hide prompt details)

All three produce similar prompts, just at different abstraction levels.

## 4.3 Component 2: Generation and Parsing

### Generation (Forge)
```python
# Reference: apps/grpo/main.py:373
# Forge uses async Generator actor
response = await policy.generate.route(
    prompt,  # Can be string or token IDs
    sampling_params={
        "temperature": 0.7,
        "max_tokens": 512,
        "n": 1  # Single sample in rollout, multiple for GRPO groups
    }
)

# response is a Completion object
# Reference: forge/data_models/completion.py
response.token_ids     # List[int]
response.logprobs      # List[float]
response.text          # str
response.prompt_ids    # List[int]
```

### Parsing Tool Calls

**Option A: Regex-based (Thinker)**
```python
# Reference: tinker-cookbook/tinker_cookbook/renderers.py:394-430
import re
import json

def parse_tool_call(text):
    """Parse <tool_call>...</tool_call> tags."""
    match = re.search(r"<tool_call>(.*?)</tool_call>", text, re.DOTALL)
    if not match:
        return None

    try:
        tool_call = json.loads(match.group(1))
        return {
            "name": tool_call["name"],
            "args": tool_call["args"]
        }
    except json.JSONDecodeError:
        return None
```

**Option B: vLLM Native Parsing**
```python
# If using vLLM with enable_auto_tool_choice=true
# Reference: verl/verl/experimental/agent_loop/tool_agent_loop.py:99-101

# vLLM automatically populates tool_calls
if response.choices[0].message.tool_calls:
    tool_call = response.choices[0].message.tool_calls[0]
    # Already parsed!
else:
    # Final answer
    pass
```

**Clarification on `response.choices[0]`:**
- This is **OpenAI API format**, used when vLLM native tool calling is enabled
- Forge's internal Generator returns `Completion` object, not OpenAI format
- For Forge, use regex parsing on `response.text`

### Handling Multiple Tool Calls

**Example: Model calls multiple tools in one turn**
```python
# Model output: "Let me search for flights and hotels.
# <tool_call>{"name": "search_flights", "args": {"destination": "NYC"}}</tool_call>
# <tool_call>{"name": "search_hotels", "args": {"city": "NYC"}}</tool_call>"

def parse_all_tool_calls(text):
    """Parse multiple tool calls."""
    matches = re.findall(r"<tool_call>(.*?)</tool_call>", text, re.DOTALL)
    tool_calls = []
    for match in matches:
        try:
            tool_call = json.loads(match)
            tool_calls.append(tool_call)
        except json.JSONDecodeError:
            continue
    return tool_calls if tool_calls else None
```

### Sample-Level Concurrency

**Sequential (simple)**
```python
# Reference: apps/grpo/main.py:372-394
episodes = []
for task in tasks:
    episode = await play_task(task, policy, tokenizer, env)
    episodes.append(episode)
```

**Parallel (faster)**
```python
# Process all tasks concurrently
tasks_coroutines = [
    play_task(task, policy, tokenizer, env)
    for task in tasks
]
episodes = await asyncio.gather(*tasks_coroutines)
```

**Why parallel?**
- While Sample 1 waits for tool execution, Sample 2/3 continue generating
- 2-4x speedup for variable-length episodes
- **OpenEnv locking**: Each task gets separate env instance, no locks needed
  ```python
  # Each task creates new environment
  async def play_task(task, ...):
      env = OpenSpielEnv(base_url=server_url)  # Separate instance
      ...
      env.close()
  ```

## 4.4 Component 3: Tool Execution

### Tool Definition (Where is it used?)

**Tool schemas are used in two places:**

1. **Prompt formatting** (Section 4.2) - tells model what tools exist
2. **Tool execution** - maps tool name to actual function

**Definition Pattern (Thinker):**
```python
# Reference: tinker-cookbook/tinker_cookbook/recipes/tool_use/search/tools.py:362-373
from abc import ABC, abstractmethod

class ToolClientInterface(ABC):
    @abstractmethod
    def get_tool_schemas(self) -> list[dict]:
        """Returns OpenAI-compatible tool definitions."""
        ...

    @abstractmethod
    async def invoke(self, tool_call: dict) -> list[dict]:
        """Executes tool and returns result messages."""
        ...

# Concrete implementation
class SearchToolClient(ToolClientInterface):
    def get_tool_schemas(self):
        return [
            {
                "name": "search",
                "description": "Search Wikipedia",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query_list": {
                            "type": "array",
                            "items": {"type": "string"}
                        }
                    },
                    "required": ["query_list"]
                }
            }
        ]

    async def invoke(self, tool_call):
        if tool_call["name"] == "search":
            results = await self.search_wikipedia(tool_call["args"]["query_list"])
            return [{"role": "tool", "content": json.dumps(results)}]
```

**Usage in loop:**
```python
# 1. Get schemas for prompt
prompt = tokenizer.apply_chat_template(
    messages,
    tools=tool_client.get_tool_schemas(),  # <-- Used here
    add_generation_prompt=True
)

# 2. Execute tool
tool_call = parse_tool_call(response.text)
if tool_call:
    result_messages = await tool_client.invoke(tool_call)  # <-- Used here
    messages.extend(result_messages)
```

### Multiple Tool Execution

**Sequential:**
```python
for tool_call in tool_calls:
    result = await tool_client.invoke(tool_call)
    messages.extend(result)
```

**Parallel (faster for I/O-bound tools):**
```python
# Execute all tools concurrently
tasks = [tool_client.invoke(tc) for tc in tool_calls]
results = await asyncio.gather(*tasks)

for result in results:
    messages.extend(result)
```

**When parallel matters:**
- Good for: API calls, database queries, web search
- Not needed for: Fast local tools (< 10ms)

## 4.5 Component 4: Message History Management

### Messages in Multi-Turn

**Structure over turns:**
```python
# Turn 1
messages = [
    {"role": "user", "content": "Search for flights to NYC"}
]

# Model generates
messages.append({"role": "assistant", "content": "I'll search... <tool_call>...</tool_call>"})

# Tool executes
messages.append({"role": "tool", "content": '{"flights": [...]}'})

# Turn 2
# Model generates again (with all history)
messages.append({"role": "assistant", "content": "Based on results, I recommend..."})
```

### Storage Patterns

**Option A: Explicit List (Thinker)**
```python
# Reference: tinker-cookbook/tinker_cookbook/recipes/tool_use/search/search_env.py:118
class SearchEnv:
    def __init__(self, ...):
        self.past_messages: list[dict] = []

    async def step(self, action):
        # Parse model response
        message = renderer.parse_response(action)
        self.past_messages.append(message)

        # Execute tools if needed
        if "tool_calls" in message:
            tool_results = await tool_client.invoke(message["tool_calls"][0])
            self.past_messages.extend(tool_results)

        # Build next prompt with all history
        next_prompt = renderer.build_generation_prompt(self.past_messages)
        return next_prompt
```

**Option B: Concatenated Tokens (Forge/VERL)**
```python
# Reference: apps/grpo/main.py:376-398, verl/.../tool_agent_loop.py:68-74
# Store all tokens in single list
episode_tokens = []
episode_logprobs = []
response_mask = []  # Track what to train on

for turn in turns:
    # LLM output
    episode_tokens.extend(llm_response.token_ids)
    episode_logprobs.extend(llm_response.logprobs)
    response_mask.extend([1] * len(llm_response.token_ids))

    # Tool result
    if tool_call:
        tool_tokens = tokenizer.encode(tool_result, add_special_tokens=False)
        episode_tokens.extend(tool_tokens)
        episode_logprobs.extend([0.0] * len(tool_tokens))  # Dummy
        response_mask.extend([0] * len(tool_tokens))
```

**Does OpenEnv hold messages?**
- **No** - OpenEnv manages environment state (game state, task state), not messages
- Messages are maintained by your rollout loop
- Reference: `OpenEnv/examples/grpo_blackjack/grpo_utils.py:408-456` shows loop managing messages

## 4.6 Component 5: Episode Storage and Response Masking

### Why Masking Matters

```python
# Multi-turn episode tokens:
# Turn 1:
"Create a task for user_1"                     # LLM output - TRAIN
"<tool_call>create_task(...)</tool_call>"      # LLM output - TRAIN
'{"status": "success", "task_id": "123"}'      # Tool output - DON'T TRAIN
# Turn 2:
"Task created successfully!"                    # LLM output - TRAIN
```

**Without masking**: Model learns to predict tool results (impossible!)
**With masking**: Model only learns its own outputs

### Episode Structure (Forge)

**Reference: apps/grpo/main.py:44-75**
```python
from dataclasses import dataclass
import torch

@dataclass
class Episode:
    episode_id: str
    pad_id: int
    request_len: int        # Length of initial prompt
    response_len: int       # Length of all responses (all turns concatenated)
    target: Any | None      # Ground truth for evaluation

    # Processed data
    completion: Completion | None      # Contains token_ids, logprobs, text
    ref_logprobs: torch.Tensor | None  # From reference model
    reward: float | None               # From reward function
    advantage: float | None            # Computed with group

    @property
    def request_tensor(self) -> torch.Tensor:
        """Padded prompt tokens."""
        ...

    @property
    def response_tensor(self) -> torch.Tensor:
        """Padded response tokens."""
        ...
```

**What about response_mask?**
- Not stored in Episode (Forge's design choice)
- Computed during training from `completion.token_ids`
- Alternative: Add to Episode or Completion (see VERL approach)

### Building Episodes from Messages

**Converting messages → single Episode:**

```python
# Reference: Adapted from apps/grpo/main.py:376-394
def messages_to_episode(messages, tokenizer, reward, task_id):
    """Convert multi-turn messages to single Episode."""

    # 1. Extract initial prompt (everything up to first assistant message)
    first_assistant_idx = next(i for i, m in enumerate(messages) if m["role"] == "assistant")
    prompt_messages = messages[:first_assistant_idx]

    prompt = tokenizer.apply_chat_template(
        prompt_messages,
        add_generation_prompt=True,
        tokenize=False
    )
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)

    # 2. Concatenate all responses
    all_tokens = []
    all_logprobs = []

    for i in range(first_assistant_idx, len(messages)):
        message = messages[i]
        text = message["content"]

        if message["role"] == "assistant":
            # LLM output - has logprobs
            tokens = tokenizer.encode(text, add_special_tokens=False)
            all_tokens.extend(tokens)
            # Note: Need to store logprobs during generation
            all_logprobs.extend(message.get("logprobs", [0.0] * len(tokens)))
        elif message["role"] == "tool":
            # Tool output - dummy logprobs
            tokens = tokenizer.encode(text, add_special_tokens=False)
            all_tokens.extend(tokens)
            all_logprobs.extend([0.0] * len(tokens))

    # 3. Create Completion
    completion = Completion(
        prompt_ids=torch.tensor(prompt_ids),
        token_ids=torch.tensor(all_tokens),
        logprobs=torch.tensor(all_logprobs),
        text=tokenizer.decode(all_tokens),
        generator_version=0
    )

    # 4. Create Episode
    episode = Episode(
        episode_id=str(uuid.uuid4()),
        pad_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        request_len=len(prompt_ids),
        response_len=len(all_tokens),
        target=None,
        completion=completion,
        ref_logprobs=None,
        reward=reward,
        advantage=None
    )

    return episode

# Usage
episode = messages_to_episode(messages, tokenizer, reward=1.0, task_id="task_1")
```

**Building response_mask:**
```python
def build_response_mask(messages, first_assistant_idx):
    """Build mask: 1 for LLM output, 0 for tool output."""
    mask = []

    for i in range(first_assistant_idx, len(messages)):
        message = messages[i]
        tokens = tokenizer.encode(message["content"], add_special_tokens=False)

        if message["role"] == "assistant":
            mask.extend([1] * len(tokens))  # TRAIN
        elif message["role"] == "tool":
            mask.extend([0] * len(tokens))  # DON'T TRAIN

    return mask
```

**How to use masks in training:**
- Pass to loss function (see `apps/grpo/main.py:127-138` for GRPO loss)
- Multiply per-token loss by mask before averaging

## 4.7 Component 6: Reward Computation

### Sparse Rewards (Most Common)

**Pattern:**
```python
# Reference: tinker-cookbook/tinker_cookbook/recipes/tool_use/search/search_env.py:161-209
# All intermediate steps get 0 reward
for turn in range(max_turns):
    if done:
        break
    response = await generate(...)
    intermediate_reward = 0.0  # No reward yet

# Final step gets actual reward
final_reward = env.check_answer(final_response)  # 1.0 or 0.0
```

**Used by:**
- Tau2Bench: 1.0 for success, 0.0 for failure
- Thinker: `correct_answer` (1.0/0.0) + format penalty
- Forge GSM8K: `MathReward()` checks final answer

### Multiple Reward Signals (Thinker Pattern)

**Reference: tinker-cookbook/tinker_cookbook/recipes/tool_use/search/search_env.py:196-209**
```python
# Thinker: Separate reward components
def compute_reward(response, ground_truth):
    correct_format = float(check_format(response))     # 1.0 or 0.0
    correct_answer = float(check_answer(response, ground_truth))  # 1.0 or 0.0

    # Combine with weights
    format_coef = -1.0  # Penalty for bad format
    total_reward = format_coef * (correct_format - 1) + correct_answer
    return total_reward

# Example:
# - Good answer, good format: -1.0 * (1.0 - 1) + 1.0 = 1.0
# - Good answer, bad format: -1.0 * (0.0 - 1) + 1.0 = 2.0
# - Bad answer, good format: -1.0 * (1.0 - 1) + 0.0 = 0.0
# - Bad answer, bad format: -1.0 * (0.0 - 1) + 0.0 = 1.0
```

**Forge Pattern:**
```python
# Reference: apps/grpo/main.py:334-336
from forge.data.rewards import MathReward, ThinkingReward

reward_functions = [MathReward(), ThinkingReward()]

total_reward = sum(
    reward_fn(prompt, response, target)
    for reward_fn in reward_functions
)
avg_reward = total_reward / len(reward_functions)
```

**Key Difference:**
- **Thinker**: Combines rewards with explicit coefficients
- **Forge**: Averages multiple reward functions
- **Both**: Sparse (only at episode end)

### Reward Shaping (Optional)

**Reference: OpenEnv/examples/grpo_blackjack/grpo_utils.py:256-268**
```python
# Base reward from environment
base_reward = env.get_final_reward()  # +1 (win), -1 (loss), 0 (draw)

# Optional shaping
shaped_reward = base_reward
if base_reward > 0:
    shaped_reward = 2.0  # Amplify wins
elif base_reward == 0:
    shaped_reward = 0.5  # Draws better than losses
else:
    shaped_reward = -1.0  # Losses

# Use shaped_reward for training
```

**When to use:**
- Sparse rewards are too delayed
- Want to bias learning toward certain behaviors
- **Caution**: Can introduce bias, use carefully

### How Environment Knows Reward

**With Environment:**
```python
# Reference: tinker-cookbook/.../search_env.py:140-148
class SearchEnv:
    def __init__(self, problem, answer, ...):
        self.problem = problem
        self.answer = answer  # Ground truth stored

    def check_answer(self, response):
        model_answer = self._extract_answer(response)
        for gold_answer in self.answer:
            if normalize_answer(model_answer) == normalize_answer(gold_answer):
                return True
        return False

    async def step(self, action):
        ...
        if episode_done:
            reward = float(self.check_answer(action))
            return StepResult(reward=reward, episode_done=True, ...)
```

**Without Environment:**
```python
# You provide reward function
def compute_reward(response, target):
    # Your logic
    return 1.0 if check_correct(response, target) else 0.0

# In loop
reward = compute_reward(final_response, task["target"])
```

## 4.8 Component 7: Environment Integration

### Thinker's Environment API (Recommended)

**Reference: tinker-cookbook/tinker_cookbook/rl/types.py**
```python
from abc import ABC, abstractmethod
from dataclasses import dataclass

class Environment(ABC):
    @abstractmethod
    async def initial_observation(self) -> tuple[Observation, StopCondition]:
        """Start episode, return initial state."""
        ...

    @abstractmethod
    async def step(self, action: Action) -> StepResult:
        """Execute action, return result."""
        ...

@dataclass
class StepResult:
    reward: float
    episode_done: bool
    next_observation: Observation
    next_stop_condition: StopCondition
    metrics: dict = field(default_factory=dict)
```

**Why this is good:**
- Standard gym-like interface
- Clear separation: env manages state, you manage policy
- Easy to implement new environments
- Used by Thinker, similar to gym

**Example Implementation:**
```python
# Reference: tinker-cookbook/.../search_env.py:100-219
class SearchEnv(Environment):
    def __init__(self, problem, answer, tool_client, renderer, ...):
        self.problem = problem
        self.answer = answer
        self.tool_client = tool_client
        self.renderer = renderer
        self.past_messages = []

    async def initial_observation(self):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": self.problem}
        ]
        self.past_messages = messages
        prompt = self.renderer.build_generation_prompt(messages)
        return prompt, stop_condition

    async def step(self, action):
        # Parse response
        message, parse_success = self.renderer.parse_response(action)
        self.past_messages.append(message)

        # Execute tools if needed
        if "tool_calls" in message:
            tool_result = await self.tool_client.invoke(message["tool_calls"][0])
            self.past_messages.extend(tool_result)

            # Continue episode
            next_prompt = self.renderer.build_generation_prompt(self.past_messages)
            return StepResult(
                reward=0.0,
                episode_done=False,
                next_observation=next_prompt,
                ...
            )
        else:
            # Final answer
            correct = self.check_answer(message["content"])
            return StepResult(
                reward=float(correct),
                episode_done=True,
                next_observation=None,
                ...
            )
```

### OpenEnv vs Thinker ToolEnv vs No Env

| Feature | OpenEnv | Thinker ToolEnv | No Env |
|---------|---------|-----------------|--------|
| **API** | Docker HTTP | Python ABC | You implement |
| **Tools** | Env-specific | Tool client | You provide |
| **Setup** | Docker containers | `pip install` | Minimal |
| **State** | Env manages | Env manages | You manage |
| **Best for** | Complex envs (browsers, games) | Tool calling tasks | Simple tasks |
| **Example** | Tau2Bench airline tasks | Wikipedia search | Math reasoning |

**When to use each:**
- **OpenEnv**: Training on diverse, sandboxed environments (Tau2Bench)
- **Thinker ToolEnv**: Clean tool calling with Python functions
- **No Env**: Simple tasks, full control over loop

### Using Thinker's Env in Forge

```python
# Forge app using Thinker's environment
async def play_task(task, policy, renderer, env):
    # 1. Get initial observation
    observation, stop_condition = await env.initial_observation()

    done = False
    all_tokens = []
    all_logprobs = []

    while not done:
        # 2. Generate
        response = await policy.generate.route(observation.prompt)

        # 3. Step environment
        step_result = await env.step(response.token_ids)

        # 4. Collect tokens
        all_tokens.extend(response.token_ids)
        all_logprobs.extend(response.logprobs)

        # 5. Check if done
        done = step_result.episode_done
        observation = step_result.next_observation

    # 6. Create Episode with final reward
    reward = step_result.reward
    episode = Episode(...)  # As in section 4.7
    return episode
```

**Key Point**: Core RL loop stays env-agnostic. Environment is injected at app level.

---

## 4.9 Handling Multiple Environments (WebSearch + Coding, etc.)

### The Challenge

Tau2Bench has multiple domains (airline, retail, etc.) and you may want to train on a mix. Similarly, you might want to train on both websearch and coding tasks. Each domain/task type has:
- Different tools
- Different max_turns
- Different reward functions
- Different evaluation criteria

### Recommended Pattern: Tinker's `CompositeDataset`

**Location**: See full research in `/home/felipemello/forge/brainstorming_forge_tau/4_examples_APIs.md` section "Handling Multiple Environments"

#### Core Abstraction: `EnvGroupBuilder`

Every environment implements this interface:

```python
# Based on tinker_cookbook/rl/types.py:64-108

class EnvGroupBuilder(ABC):
    """
    Builds a group of environments. Used for:
    - GRPO groups (e.g., 8 copies for one problem)
    - Mixed environment training
    """

    @abstractmethod
    async def make_envs(self) -> Sequence[Env]:
        """Create a group of environments (e.g., 8 copies for GRPO)"""
        pass

    def logging_tags(self) -> list[str]:
        """Tags for logging (e.g., ['airline'], ['retail'])"""
        return []
```

#### Mixing Environments: `CompositeDataset`

```python
class CompositeDataset:
    """Mix multiple datasets at the batch level."""

    def __init__(self, datasets: List[RLDataset], groups_per_batch_list: List[int]):
        self.datasets = datasets
        self.groups_per_batch_list = groups_per_batch_list

    def get_batch(self, i_batch: int) -> tuple[List[EnvGroupBuilder], List[int]]:
        """
        Get a batch by sampling from each dataset.

        Returns:
            env_group_builders: List of all env group builders (mixed!)
            dataset_indices: Which dataset each builder came from
        """
        all_env_group_builders = []
        all_dataset_indices = []

        for dataset_idx, (dataset, groups_per_batch) in enumerate(
            zip(self.datasets, self.groups_per_batch_list)
        ):
            env_group_builders = dataset.get_batch(i_batch)
            all_env_group_builders.extend(env_group_builders)
            all_dataset_indices.extend([dataset_idx] * groups_per_batch)

        return all_env_group_builders, all_dataset_indices
```

#### Example: Airline + Retail Tasks

```python
# 1. Define environment builders for each domain
airline_env_builder = Tau2BenchEnvGroupBuilder(
    domain="airline",
    tools=[book_flight, cancel_reservation, ...],
    max_turns=10,
    dataset_name="airline"
)

retail_env_builder = Tau2BenchEnvGroupBuilder(
    domain="retail",
    tools=[search_products, add_to_cart, ...],
    max_turns=15,
    dataset_name="retail"
)

# 2. Create datasets
airline_dataset = Tau2BenchDataset(domain="airline")
retail_dataset = Tau2BenchDataset(domain="retail")

# 3. Mix with CompositeDataset
mixed_dataset = CompositeDataset(
    datasets=[airline_dataset, retail_dataset],
    groups_per_batch_list=[50, 50]  # 50 airline + 50 retail per batch
)

# 4. Use in Forge rollout
async def continuous_rollouts():
    while True:
        # Get mixed batch
        env_group_builders, dataset_indices = mixed_dataset.get_batch(batch_idx)

        # Each builder knows its own environment configuration!
        for builder in env_group_builders:
            # builder has:
            # - Its own tools (airline vs retail)
            # - Its own max_turns
            # - Its own reward function
            episodes = await play_task_with_env_builder(
                policy=policy,
                env_builder=builder,
            )

            # Logging automatically separates by domain (via builder.logging_tags())
```

#### Why This Works

- ✅ **Different tools** per environment (airline vs retail)
- ✅ **Different max_turns** per environment
- ✅ **Different rewards** per environment (domain-specific rubrics)
- ✅ **Unified training loop** (no special casing needed)
- ✅ **Separate metrics** (via logging_tags: ['airline'], ['retail'])
- ✅ **Flexible mixing ratios** (control via groups_per_batch_list)
- ✅ **Batch-level mixing**: Each batch contains groups from multiple datasets
- ✅ **Decentralized**: Each `EnvGroupBuilder` is self-contained

#### Simpler Alternative: Manual Routing

If you don't need the full flexibility, implement simple routing:

```python
# Map domain to environment configuration
task_to_env = {
    "airline": (airline_tools, airline_max_turns, airline_reward_fn),
    "retail": (retail_tools, retail_max_turns, retail_reward_fn),
}

async def play_task(task_sample, policy, tokenizer):
    domain = task_sample["domain"]
    tools, max_turns, reward_fn = task_to_env[domain]

    # Use domain-specific configuration
    episode = await multi_turn_rollout(
        task=task_sample,
        policy=policy,
        tools=tools,
        max_turns=max_turns,
    )

    episode.reward = reward_fn(episode)
    return episode
```

**Recommendation**: Start with manual routing for simplicity. Upgrade to `CompositeDataset` pattern if you need:
- Fine-grained control over mixing ratios
- Separate logging per domain
- Easy addition of new domains

---

**Next**: Part 5 shows complete architectural patterns for Forge + Tau2Bench.
