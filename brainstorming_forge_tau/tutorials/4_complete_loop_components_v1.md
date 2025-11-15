# Part 4: Complete Multi-Turn Tool Calling Loop (Components)

This part breaks down all 8 components needed for multi-turn tool calling.

## 4.0 Generator Options: Internal vs External vLLM

You have three options for running vLLM:

### Option A: Forge Generator (Internal vLLM) ✅ **Recommended**

**How it works:**
- vLLM engine runs **inside Forge** as a distributed actor
- Allocated to its own GPUs via Monarch process mesh
- Communication via **async actor calls** (not HTTP)
- This is what Forge currently does

```python
# apps/grpo/main.py
policy = Generator(
    model_path="Qwen/Qwen2.5-1.5B-Instruct",
    engine_args={...}
)

# Generate
response = await policy.generate.route(prompt)
```

**Pros:**
- Efficient (no HTTP overhead)
- Integrated with Forge's distributed system
- GPU allocation handled automatically

**Cons:**
- Less flexible for debugging
- Harder to inspect intermediate states

### Option B: External vLLM Server (Separate Process)

**How it works:**
- vLLM runs as independent HTTP server (separate process)
- Forge sends blocking or async HTTP requests
- Used by TRL examples

```python
# Start vLLM server separately:
# $ vllm serve Qwen/Qwen2.5-1.5B-Instruct --port 8000

# In your code:
import requests

response = requests.post(
    "http://localhost:8000/v1/completions",
    json={
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
        "prompt": prompt,
        "max_tokens": 512
    }
)
```

**Pros:**
- Easy to debug (inspect server logs)
- Can restart server without restarting training
- Separation of concerns

**Cons:**
- HTTP overhead
- Separate GPU allocation needed
- More complex setup

### Option C: Hybrid

Use external for debugging/exploration, internal for production training.

**All examples in this tutorial use Option A (Forge Generator).** We'll note where Option B could be used.

## 4.1 Overview: The Complete Loop

```python
async def play_task(task, policy, tokenizer, env, max_turns=10):
    """Complete multi-turn tool calling loop."""

    # 1. Episode Initialization
    env_result = env.reset(task=task)
    messages = [{"role": "user", "content": task}]
    done = False
    turn = 0

    # Storage for episode
    all_tokens = []
    all_logprobs = []
    response_mask = []

    while not done and turn < max_turns:
        # 2. Prompt Formatting
        prompt = tokenizer.apply_chat_template(
            messages,
            tools=env.get_tools(),  # Tool definitions
            add_generation_prompt=True
        )

        # 3. Generation & Parsing
        response = await policy.generate.route(prompt)
        tool_call = parse_tool_call(response.text)

        # 4. Tool Execution (if tool call)
        if tool_call:
            result = env.execute_tool(tool_call)
            messages.append({"role": "assistant", "tool_calls": [tool_call]})
            messages.append({"role": "tool", "content": result})

            # 5. Token Collection (concatenate)
            all_tokens.extend(response.token_ids)
            all_logprobs.extend(response.logprobs)
            response_mask.extend([1] * len(response.token_ids))  # Train on LLM output

            tool_tokens = tokenizer.encode(result)
            all_tokens.extend(tool_tokens)
            response_mask.extend([0] * len(tool_tokens))  # DON'T train on tool result
        else:
            # Final answer
            messages.append({"role": "assistant", "content": response.text})
            all_tokens.extend(response.token_ids)
            all_logprobs.extend(response.logprobs)
            response_mask.extend([1] * len(response.token_ids))
            done = True

        turn += 1

    # 6. Reward Computation
    reward = env.get_final_reward()

    # 7. Create Episode
    episode = Episode(
        token_ids=all_tokens,
        logprobs=all_logprobs,
        response_mask=response_mask,
        reward=reward
    )

    return episode
```

Let's break down each component.

## 4.2 Component 1: Episode Initialization

**Option A: From environment**
```python
env = OpenEnv(base_url="http://localhost:8001")
result = env.reset(task_id="create_task_1", domain="mock")

# result.observation contains initial state
messages = [{"role": "user", "content": result.observation.info_state}]
```

**Option B: From task data**
```python
task_data = load_task("tau2bench/mock/create_task_1.json")
messages = [
    {"role": "system", "content": format_system_prompt(task_data["tools"])},
    {"role": "user", "content": task_data["ticket"]}
]
```

**Pros/Cons:**
- **Option A**: Cleaner, environment handles state
- **Option B**: More control, can customize prompts

## 4.3 Component 2: Prompt Formatting with Tools

### Option A: Manual Chat Template

```python
def format_prompt(messages, tools):
    # Build system prompt
    tool_schemas = "\n".join([f"- {t['name']}: {t['description']}" for t in tools])
    system = f"You have access to:\n{tool_schemas}\nUse format: <function_call>{{...}}</function_call>"

    # Apply chat template
    full_messages = [{"role": "system", "content": system}] + messages
    return tokenizer.apply_chat_template(full_messages, add_generation_prompt=True)
```

### Option B: Renderer Pattern (Tinker) 🎯

**Clean abstraction for prompt formatting:**

```python
# tinker_cookbook/renderers.py
class Renderer:
    def build_generation_prompt(self, messages):
        """Convert messages to tokenized prompt."""
        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        tokens = self.tokenizer.encode(prompt_text)
        return ModelInput(prompt=prompt_text, tokens=tokens)

    def parse_response(self, tokens):
        """Parse model output to Message."""
        text = self.tokenizer.decode(tokens)

        # Check for tool calls
        if "<tool_call>" in text:
            tool_call = self._parse_tool_call(text)
            return Message(role="assistant", tool_calls=[tool_call])
        else:
            return Message(role="assistant", content=text)
```

**Why Tinker's approach is good:**
- Separation of concerns (rendering vs logic)
- Reusable across tasks
- Easy to test
- Handles tokenization details

### Option C: vLLM Native (Verifiers)

```python
# vLLM handles tool formatting automatically
prompt = tokenizer.apply_chat_template(
    messages,
    tools=tool_schemas,  # Pass tools to tokenizer
    add_generation_prompt=True
)
# vLLM formats tools based on model type
```

**When to use each:**
- **Manual**: Full control, debugging
- **Renderer** 🎯: Clean architecture, reusability
- **vLLM Native**: Model supports it, production-ready

## 4.4 Component 3: Generation, Parsing, and Concurrency

### Calling the Generator

**Forge Generator (async):**
```python
response = await policy.generate.route(
    prompt,
    sampling_params={
        "temperature": 0.7,
        "max_tokens": 512
    }
)
```

### Parsing Tool Calls

**Text parsing (regex):**
```python
def parse_tool_call(text):
    match = re.search(r'<function_call>(.*?)</function_call>', text)
    if match:
        return json.loads(match.group(1))
    return None
```

**Tag-based (Qwen example):**
```python
# tinker_cookbook/renderers.py
def parse_response(self, text):
    match = re.search(r"<tool_call>(.*?)</tool_call>", text, re.DOTALL)
    if match:
        try:
            tool_call = json.loads(match.group(1))
            return Message(role="assistant", tool_calls=[tool_call])
        except json.JSONDecodeError:
            return Message(role="assistant", content=text)
    return Message(role="assistant", content=text)
```

**Native (vLLM auto-parsing):**
```python
# response.choices[0] already has tool_calls populated by vLLM
if response.choices[0].message.tool_calls:
    tool_call = response.choices[0].message.tool_calls[0]
```

**Note on `response.choices[0]`:**
- `generate()` can return N samples when `n > 1`
- We typically use first sample (`[0]`) in rollout
- For GRPO, we generate multiple samples per prompt (group_size)

### vLLM Configuration Flags

**For Forge Generator (Option A):**
```yaml
# apps/tau2bench/grpo/config.yaml
policy:
  engine_args:
    model: "Qwen/Qwen2.5-1.5B-Instruct"

    # Tool calling support
    enable_auto_tool_choice: true  # vLLM parses tool calls automatically
    tool_call_parser: "hermes"     # Format: hermes/mistral/llama/internlm

    # Performance
    tensor_parallel_size: 1
    gpu_memory_utilization: 0.9
    enable_prefix_caching: true    # Helps with multi-turn!
```

**Flag meanings:**
- `enable_auto_tool_choice`: Enables native tool call parsing
- `tool_call_parser`: Specifies parser format (model-dependent)
- `async_engine`: Enables AsyncLLM engine
    # TODO: need to confirm if what we are doing is compatible with this
    # TODO: explain why this would be helpful at all

### Sample-Level Concurrency

**Sequential (simple):**
```python
episodes = []
for task in tasks:
    episode = await play_task(task, ...)
    episodes.append(episode)
```

**Parallel:**
```python
# Process all tasks concurrently
tasks_coroutines = [
    play_task(task, ...)
    for task in tasks
]
episodes = await asyncio.gather(*tasks_coroutines)
```

**Performance benefit:**
- While Sample 1 waits for tool execution, Sample 2/3/4 continue generating
- Can achieve 2-4x speedup with variable-length episodes

## 4.5 Component 4: Tool Execution

### Tool Definition Approaches

**Type-hinted Python functions (Verifiers)** 🎯:
```python
async def search_wiki(query: str) -> list[str]:
    """Search Wikipedia for articles.

    Args:
        query: Search query string

    Returns:
        List of article titles
    """
    return wikipedia.search(query)

# Auto-convert to schema
tool_schema = convert_func_to_oai_tool(search_wiki)
```

**Tinker's approach** 🎯:
```python
# tinker_cookbook/recipes/tool_use/search/tools.py
class ToolClientInterface(ABC):
    @abstractmethod
    def get_tool_schemas(self) -> list[dict]:
        """Returns tool definitions"""
        ...

    @abstractmethod
    async def invoke(self, tool_call: ToolCall) -> list[Message]:
        """Executes tool and returns results"""
        ...
```

**Manual schemas:**
```python
tools = [
    {
        "name": "create_task",
        "description": "Create a new task",
        "parameters": {
            "type": "object",
            "properties": {
                "user_id": {"type": "string"},
                "title": {"type": "string"}
            },
            "required": ["user_id", "title"]
        }
    }
]
```

### Execution Patterns

**Sequential:**
```python
for tool_call in tool_calls:
    result = await execute_tool(tool_call)
    results.append(result)
```

**Parallel:**
```python
# Execute all tools concurrently
tasks = [execute_tool(tc) for tc in tool_calls]
results = await asyncio.gather(*tasks)
```

**When parallel matters:**
- ✅ **Good for**: I/O-bound tools (API calls, database queries)
- ⚠️ **OK for**: Fast tools, debugging, simple cases (sequential is fine)

## 4.6 Component 5: Message History Management

### Explicit List Pattern (Tinker)

```python
# tinker_cookbook/recipes/tool_use/search/search_env.py
class SearchEnv:
    def __init__(self, ...):
        self.past_messages: list[Message] = []

    async def step(self, action):
        # Parse response
        message = self.renderer.parse_response(action)
        self.past_messages.append(message)

        # Execute tools if needed
        if "tool_calls" in message:
            tool_result = await execute_tool(...)
            self.past_messages.extend(tool_result)

        # Build next prompt
        next_prompt = self.renderer.build_generation_prompt(self.past_messages)
        return StepResult(next_observation=next_prompt, ...)
```

### Concatenated Storage (TRL, NeMo-RL)

```python
# TRL pattern: concatenate all tokens
episode_tokens = []
episode_logprobs = []

for turn in turns:
    response = generate(...)
    episode_tokens.extend(response.token_ids)  # Concatenate
    episode_logprobs.extend(response.logprobs)
```

### Token ID Storage in Messages (NeMo-RL)

```python
# RL/nemo_rl/experience/rollouts.py
messages = [
    {
        "role": "user",
        "content": "Task prompt",
        "token_ids": [101, 102, 103, ...]
    },
    {
        "role": "assistant",
        "content": "Tool call...",
        "token_ids": [345, 346, ...],
        "generation_logprobs": [-0.1, -0.2, ...]
    }
]
```

**Comparison:**

| Approach | Pros | Cons | Use When |
|----------|------|------|----------|
| Explicit list | Clean, debuggable | Requires conversion | Research, clean code |
| Concatenated | Simple, direct | Hard to debug | Simple prototypes |
| Token IDs in msgs | Preserves structure | More complex | Production, flexibility |

## 4.7 Component 6: Token Collection, Episode Storage, and Response Masking

### Why Masking Matters

**Problem**: Tool results are not model-generated, so we shouldn't train on them.

```python
# Multi-turn episode:
Turn 1: User: "Create task"
Turn 2: Model: create_task(user_id="user_1", ...)  # TRAIN on this
Turn 3: Tool: {"status": "success", "task_id": "task_123"}  # DON'T TRAIN on this
Turn 4: Model: "Task created!"  # TRAIN on this
```

**Without masking**: Model learns to predict tool results (impossible!)
**With masking**: Model only learns to predict its own outputs

### Token Collection Strategies

**Strategy A: Per-step Episodes** (simpler):
```python
# Each turn = separate Episode
episodes = []
for step in game_steps:
    episode = Episode(
        game_id=game_id,
        step_num=step_num,
        completion=step["response"],
        reward=final_game_reward  # Same reward for all steps
    )
    episodes.append(episode)
```

**Pros**: Simpler, matches Forge's current pattern
**Cons**: Can't share context between steps easily

**Strategy B: Concatenated Episodes** (full trajectory):
```python
# All turns = one Episode
all_tokens = []
all_logprobs = []
response_mask = []

for turn in turns:
    # LLM output
    all_tokens.extend(llm_tokens)
    all_logprobs.extend(llm_logprobs)
    response_mask.extend([1] * len(llm_tokens))  # TRAIN

    # Tool result
    all_tokens.extend(tool_tokens)
    response_mask.extend([0] * len(tool_tokens))  # IGNORE

episode = Episode(
    token_ids=all_tokens,
    logprobs=all_logprobs,
    response_mask=response_mask,
    reward=final_reward
)
```

**Pros**: Full trajectory, gradient flows through all turns
**Cons**: More complex

### Building the Response Mask

**During Rollout (VERL, NeMo-RL):**
```python
# verl/experimental/agent_loop/tool_agent_loop.py
response_mask = []

# LLM generates
agent_data.response_ids = output.token_ids
response_mask.extend([1] * len(agent_data.response_ids))  # TRAIN

# Tool executes
tool_result_ids = tokenizer.encode(tool_result)
response_mask.extend([0] * len(tool_result_ids))  # DON'T TRAIN
```

**During Processing (Verifiers, Tinker)**:

Tinker's trajectory→data conversion:

```python
# tinker_cookbook/rl/data_processing.py
def trajectory_to_data(traj: Trajectory):
    mask = []
    advantages = []

    for transition in traj.transitions:
        obs_len = len(transition.ob.tokens)  # Environment observation
        ac_len = len(transition.ac.tokens)   # LLM action

        # Build mask
        mask.extend([0.0] * obs_len)   # DON'T train on observations
        mask.extend([1.0] * ac_len)     # TRAIN on actions

        # Assign advantages
        advantages.extend([0] * obs_len)
        advantages.extend([traj_advantage] * ac_len)

    return Datum(
        model_input=input_tokens,
        loss_fn_inputs={
            "mask": mask,
            "advantages": advantages
        }
    )
```

**Why Tinker's approach is good:** 🎯
- Clean separation: rollout phase vs data processing phase
- Reusable across RL algorithms
- Easy to test and debug
- Explicit trajectory structure

### Episode Storage Patterns

**Forge-compatible Episode:**
```python
@dataclass
class Episode:
    episode_id: str

    # Token data
    token_ids: list[int]        # Concatenated all turns
    logprobs: list[float]       # Per-token logprobs
    response_mask: list[int]    # 1=train, 0=ignore

    # Metadata
    reward: float
    num_turns: int
    task_id: str

    # Optional: store messages for debugging
    messages: list[dict] = None
```

## 4.8 Component 7: Reward Computation

### Sparse Rewards (Tau2Bench, most RL)

```python
# All intermediate steps: reward = 0.0
for turn in range(max_turns):
    if done:
        break
    response = generate(...)
    env_result = env.step(response)
    intermediate_reward = 0.0  # No reward yet

# Final step: get actual reward
final_reward = env.get_final_reward()  # 0.0 or 1.0
```

### Dense Rewards (per-step shaping)

```python
# OpenEnv/examples/grpo_blackjack/grpo_utils.py
final_game_reward = result.reward  # +1, -1, or 0

# Optional: reward shaping
shaped_reward = final_game_reward
if final_game_reward > 0:
    shaped_reward += 0.1 * num_correct_actions  # Bonus for good actions
```

### Multiple Reward Signals (TRL pattern)

```python
# trl/examples/scripts/openenv/wordle.py
def reward_correct(completions, **kwargs):
    return kwargs.get("correct_reward", [0.0] * len(completions))

def reward_greens(completions, **kwargs):
    return kwargs.get("green_reward", [0.0] * len(completions))

# In trainer
trainer = GRPOTrainer(
    reward_funcs=[reward_correct, reward_greens],
    reward_weights=[1.0, 0.5]  # Weight each signal
)

# Total reward = 1.0 * correct + 0.5 * greens
```

## 4.9 Component 8: Environment Integration

### OpenEnv vs ToolEnv Comparison

| Feature | OpenEnv | ToolEnv (Verifiers) |
|---------|---------|---------------------|
| **Purpose** | General environments | Tool calling tasks |
| **API** | Docker HTTP | Python functions |
| **Tools** | Environment-specific | Type-hinted functions |
| **Setup** | Docker containers | pip install |
| **Use for** | Training (flexible) | Evaluation (clean) |

### Tinker's Environment API 🎯

```python
# tinker_cookbook/rl/environments.py
class Environment(ABC):
    @abstractmethod
    async def initial_observation(self) -> tuple[Observation, StopCondition]:
        """Start episode, return initial state"""
        ...

    @abstractmethod
    async def step(self, action: Action) -> StepResult:
        """Execute action, return result"""
        ...

@dataclass
class StepResult:
    reward: float
    episode_done: bool
    next_observation: Observation
    metrics: dict = field(default_factory=dict)
```

**Why Tinker's API is good:** 🎯
- Standard gym-like interface
- Clear data structures
- Easy to implement new environments
- Separation of concerns

### When to Use Each

**Use OpenEnv when:**
- Training on diverse tasks
- Need sandboxed execution
- Want flexibility

**Use ToolEnv when:**
- Evaluating on specific benchmarks
- Tools are Python functions
- Want clean, simple setup

**Note**: Core functions stay env-agnostic. Environment is injected at app level.

---

**Next**: Part 5 shows complete architectural patterns for Forge + Tau2Bench + OpenEnv.
