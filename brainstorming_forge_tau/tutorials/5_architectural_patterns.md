# Part 5: Architectural Patterns for Forge + Tau2Bench + OpenEnv

**CRITICAL NOTE**: All patterns use the Forge stack:
- **Forge Generator** (internal vLLM via Monarch actors) - NOT external HTTP server
- **OpenEnv** for tool execution and training
- **Tau2Bench** for tasks and evaluation
- **vLLM** engine (internal to Forge Generator)

## Pattern A: Simple Sequential + Token Concatenation (TRL-inspired)

### Summary

**What it is**: Concatenate all turns into one sequence, train as single episode. Each turn's tokens are appended to the same lists.

**When to use**: Simplest implementation for prototypes, proven pattern from TRL, good starting point before adding complexity.

### YAML Configuration

```yaml
# examples/tau2bench/grpo/simple_concat.yaml
policy:
  type: "Generator"
  model_path: "Qwen/Qwen2.5-1.5B-Instruct"
  engine_args:
    tensor_parallel_size: 1
    gpu_memory_utilization: 0.9
    max_model_len: 2048

trainer:
  type: "TitanTrainer"
  learning_rate: 1e-5
  beta: 0.1  # KL penalty

rollout:
  group_size: 8  # GRPO group
  max_turns: 10  # Max turns per episode
  concurrent_tasks: 4  # Process 4 tasks in parallel

openenv:
  base_url: "http://localhost:8001"
  timeout: 30
```

### Complete Code

```python
# examples/tau2bench/grpo/simple_concat_pattern.py

async def play_task_simple(
    task_prompt: str,
    policy: Generator,
    tokenizer,
    env_client: OpenEnv,
    max_turns: int = 10,
):
    """
    Simple multi-turn loop with token concatenation.
    Adapted from TRL pattern, but uses Forge Generator.
    """
    # Initialize
    env_result = env_client.reset(task=task_prompt)
    messages = [{"role": "user", "content": task_prompt}]

    # Storage for ENTIRE episode (all turns concatenated)
    episode_tokens = []
    episode_logprobs = []
    done = False
    turn = 0

    while not done and turn < max_turns:
        # 1. Format prompt
        prompt = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False
        )

        # 2. Generate using Forge Generator
        response = await policy.generate.route(
            prompt,
            sampling_params={"temperature": 0.7, "max_tokens": 256}
        )

        # 3. CRITICAL: Concatenate tokens (TRL's trick)
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        completion_ids = response.token_ids

        episode_tokens.extend(prompt_ids)
        episode_tokens.extend(completion_ids)
        episode_logprobs.extend(response.logprobs)

        # 4. Parse tool call
        tool_call = parse_tool_call(response.text)

        if tool_call:
            # Execute tool via OpenEnv
            env_result = env_client.step(tool_call)

            # Add to message history
            messages.append({
                "role": "assistant",
                "content": response.text,
                "tool_calls": [tool_call]
            })
            messages.append({
                "role": "tool",
                "content": env_result.observation.text
            })

            done = env_result.done
        else:
            # Final answer (no tool call)
            messages.append({
                "role": "assistant",
                "content": response.text
            })
            done = True

        turn += 1

    # 5. Get final reward
    final_reward = env_result.reward if env_result.done else 0.0

    # 6. Create episode (entire multi-turn = one sequence)
    episode = {
        "token_ids": episode_tokens,
        "logprobs": episode_logprobs,
        "reward": final_reward,
        "num_turns": turn
    }

    return episode


def parse_tool_call(text: str):
    """Simple regex-based parser."""
    match = re.search(r'<function_call>(.*?)</function_call>', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            return None
    return None
```

**Adaptation for External vLLM (Option B):**
```python
# Replace Forge Generator call with HTTP request
import requests

response = requests.post(
    "http://localhost:8000/v1/completions",
    json={"prompt": prompt, "max_tokens": 256}
)
result = response.json()
episode_tokens.extend(result["choices"][0]["token_ids"])
```

### Key Insights

✅ **Simplest pattern**: Easy to understand and implement
✅ **Token concatenation is THE trick**: All turns become one sequence
✅ **Works well**: Proven by TRL on various tasks
✅ **No masking**: Trains on everything (including tool results) - acceptable for simple cases
⚠️ **Limitation**: No response masking means training on tool outputs

**Trade-offs:**
- **Pros**: Simple, direct, easy to debug
- **Cons**: No masking (less efficient), harder to extend
- **Best for**: Prototypes, initial experiments, simple tasks

## Pattern B: Clean Abstractions with Renderer (Tinker-inspired) 🎯

### Summary

**What it is**: Use Renderer pattern for prompt formatting, clean Environment API, explicit trajectory processing with response masking.

**When to use**: Research projects, need reusability, want clean maintainable code that's easy to extend and debug. **Recommended for production Forge implementation.**

### YAML Configuration

```yaml
# examples/tau2bench/grpo/tinker_pattern.yaml
policy:
  type: "Generator"
  model_path: "Qwen/Qwen2.5-1.5B-Instruct"
  engine_args:
    tensor_parallel_size: 1
    gpu_memory_utilization: 0.9

renderer:
  type: "Qwen3Renderer"  # Model-specific renderer

environment:
  type: "OpenEnvToolEnv"
  base_url: "http://localhost:8001"
  max_turns: 10

rollout:
  group_size: 8
  trajectory_processing: "with_masking"  # Enable response masking
```

### Complete Code

**1. Renderer (Tinker pattern)** 🎯

```python
# forge/utils/renderers.py

class Renderer(ABC):
    """Abstract base for model-specific rendering."""

    @abstractmethod
    def build_generation_prompt(self, messages: list[dict]):
        """Convert message history to model input."""
        ...

    @abstractmethod
    def parse_response(self, response_tokens: list[int]):
        """Parse model output to Message."""
        ...


class Qwen3Renderer(Renderer):
    """Qwen-specific renderer with tool calling support."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def build_generation_prompt(self, messages: list[dict]):
        """Build prompt from message history."""
        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        tokens = self.tokenizer.encode(prompt_text, add_special_tokens=False)

        return ModelInput(
            prompt=prompt_text,
            tokens=tokens
        )

    def parse_response(self, response_tokens: list[int]):
        """Parse response for tool calls."""
        text = self.tokenizer.decode(response_tokens, skip_special_tokens=True)

        # Check for tool call tag
        match = re.search(r"<tool_call>(.*?)</tool_call>", text, re.DOTALL)
        if match:
            try:
                tool_call = json.loads(match.group(1))
                return Message(
                    role="assistant",
                    content=text,
                    tool_calls=[tool_call]
                )
            except json.JSONDecodeError:
                pass

        return Message(role="assistant", content=text)


@dataclass
class ModelInput:
    prompt: str
    tokens: list[int]


@dataclass
class Message:
    role: str
    content: str
    tool_calls: list[dict] = None
```

**2. Environment with Clean API** 🎯

```python
# forge/environments/tool_env.py

class ToolEnv(ABC):
    """Clean environment interface (Tinker pattern)."""

    @abstractmethod
    async def initial_observation(self):
        """Start episode, return initial state."""
        ...

    @abstractmethod
    async def step(self, action):
        """Execute action, return StepResult."""
        ...


@dataclass
class StepResult:
    reward: float
    episode_done: bool
    next_observation: ModelInput
    metrics: dict = field(default_factory=dict)


class OpenEnvToolEnv(ToolEnv):
    """OpenEnv adapter with ToolEnv interface."""

    def __init__(self, base_url: str, renderer: Renderer, max_turns: int = 10):
        self.client = OpenEnv(base_url=base_url)
        self.renderer = renderer
        self.max_turns = max_turns
        self.past_messages = []
        self.current_turn = 0

    async def initial_observation(self):
        result = self.client.reset()
        self.past_messages = [
            {"role": "user", "content": result.observation.info_state}
        ]
        self.current_turn = 0
        return self.renderer.build_generation_prompt(self.past_messages)

    async def step(self, action_tokens: list[int]):
        """Execute one step."""
        # Parse response
        message = self.renderer.parse_response(action_tokens)
        self.past_messages.append(message)
        self.current_turn += 1

        # Check if tool call
        if message.tool_calls:
            # Execute tool via OpenEnv
            tool_call = message.tool_calls[0]
            env_result = self.client.step(tool_call)

            # Add tool result to history
            tool_message = {
                "role": "tool",
                "content": env_result.observation.text
            }
            self.past_messages.append(tool_message)

            # Check if done
            if env_result.done or self.current_turn >= self.max_turns:
                return StepResult(
                    reward=env_result.reward,
                    episode_done=True,
                    next_observation=ModelInput.empty(),
                )
            else:
                # Continue episode
                next_obs = self.renderer.build_generation_prompt(self.past_messages)
                return StepResult(
                    reward=0.0,
                    episode_done=False,
                    next_observation=next_obs,
                )
        else:
            # Final answer (no tool call) - episode done
            return StepResult(
                reward=self.client.get_final_reward(),
                episode_done=True,
                next_observation=ModelInput.empty(),
            )
```

**3. Rollout with Trajectory** 🎯

```python
# forge/rollouts/multiturn.py

@dataclass
class Transition:
    """Single step in trajectory."""
    ob: ModelInput          # Observation (prompt)
    ac: TokensWithLogprobs  # Action (LLM output)
    reward: float
    episode_done: bool


@dataclass
class Trajectory:
    """Complete episode trajectory."""
    transitions: list[Transition]
    final_reward: float


async def do_rollout_tinker_pattern(
    policy: Generator,
    env: ToolEnv,
):
    """Tinker-style rollout."""
    transitions = []

    # Get initial observation
    ob = await env.initial_observation()

    while True:
        # Generate action
        response = await policy.generate.route(
            ob.prompt,
            sampling_params={"temperature": 0.7, "max_tokens": 256}
        )

        ac = TokensWithLogprobs(
            tokens=response.token_ids,
            logprobs=response.logprobs
        )

        # Execute in environment
        step_result = await env.step(response.token_ids)

        # Store transition
        transition = Transition(
            ob=ob,
            ac=ac,
            reward=step_result.reward,
            episode_done=step_result.episode_done
        )
        transitions.append(transition)

        # Check if done
        if step_result.episode_done:
            break

        # Update observation
        ob = step_result.next_observation

    return Trajectory(
        transitions=transitions,
        final_reward=transitions[-1].reward
    )
```

**4. Trajectory Processing with Masking** 🎯

```python
# forge/data/trajectory_processing.py

def trajectory_to_episode(traj: Trajectory, advantage: float):
    """
    Convert trajectory to training episode with response masking.
    Tinker pattern: mask built during data processing, not rollout.
    """
    all_tokens = []
    all_logprobs = []
    response_mask = []
    advantages = []

    for transition in traj.transitions:
        # Observation tokens (prompt, tool results)
        ob_tokens = transition.ob.tokens
        ob_len = len(ob_tokens)

        # Action tokens (LLM output)
        ac_tokens = transition.ac.tokens
        ac_logprobs = transition.ac.logprobs
        ac_len = len(ac_tokens)

        # Concatenate
        all_tokens.extend(ob_tokens)
        all_tokens.extend(ac_tokens)

        all_logprobs.extend([0.0] * ob_len)  # Placeholder for obs
        all_logprobs.extend(ac_logprobs)

        # Build mask: 0 for observations, 1 for actions
        response_mask.extend([0] * ob_len)   # DON'T train on obs
        response_mask.extend([1] * ac_len)   # TRAIN on actions

        # Assign advantages (only to action tokens)
        advantages.extend([0.0] * ob_len)
        advantages.extend([advantage] * ac_len)

    return Episode(
        token_ids=all_tokens,
        logprobs=all_logprobs,
        response_mask=response_mask,
        advantages=advantages,
        reward=traj.final_reward
    )
```

### Key Insights

✅ **Clean separation of concerns**: Rendering, environment, data processing are separate
✅ **Reusable components**: Renderer works across tasks, easy to swap
✅ **Easy to test**: Each component can be tested independently
✅ **Response masking**: Built during data processing (clean pattern)
✅ **Production-ready**: Based on Tinker's proven design

**Why this pattern is good:** 🎯
- **Modularity**: Components are independent and reusable
- **Testability**: Easy to unit test each piece
- **Debuggability**: Clear data flow, easy to inspect
- **Extensibility**: Easy to add new models, environments

**Trade-offs:**
- **Pros**: Clean code, maintainable, extensible, production-ready
- **Cons**: More code than Pattern A, requires understanding abstractions
- **Best for**: Production implementations, research projects, team codebases

## Pattern C: State Machine + Async Parallel Tools (VERL-inspired)

### Summary

**What it is**: Explicit state machine (PENDING → GENERATING → PROCESSING_TOOLS → ...) with parallel tool execution using `asyncio.gather()`.

**When to use**: Complex tool workflows requiring explicit state management, production systems with multiple concurrent tool calls per turn.

### YAML Configuration

```yaml
# examples/tau2bench/grpo/state_machine_pattern.yaml
policy:
  type: "Generator"
  model_path: "Qwen/Qwen2.5-1.5B-Instruct"

state_machine:
  max_assistant_turns: 5
  max_parallel_tool_calls: 3
  states: ["PENDING", "GENERATING", "PROCESSING_TOOLS", "TERMINATED"]

tools:
  execution_mode: "parallel"  # Execute tools concurrently
  timeout: 10
```

### Complete Code

```python
# examples/tau2bench/grpo/state_machine_pattern.py

from enum import Enum

class AgentState(Enum):
    PENDING = "pending"
    GENERATING = "generating"
    PROCESSING_TOOLS = "processing_tools"
    TERMINATED = "terminated"


@dataclass
class AgentData:
    """State for one episode."""
    messages: list[dict]
    response_ids: list[int]
    response_mask: list[int]
    response_logprobs: list[float]
    tool_calls: list[dict]
    assistant_turns: int = 0
    state: AgentState = AgentState.PENDING


async def run_state_machine_episode(
    task: str,
    policy: Generator,
    tokenizer,
    env: OpenEnv,
    max_assistant_turns: int = 5,
    max_parallel_tools: int = 3,
):
    """VERL-inspired state machine pattern."""

    agent_data = AgentData(
        messages=[{"role": "user", "content": task}],
        response_ids=[],
        response_mask=[],
        response_logprobs=[],
        tool_calls=[]
    )

    # State machine loop
    while agent_data.state != AgentState.TERMINATED:
        if agent_data.state == AgentState.PENDING:
            agent_data.state = await handle_pending(agent_data, tokenizer)

        elif agent_data.state == AgentState.GENERATING:
            agent_data.state = await handle_generating(
                agent_data, policy, tokenizer, max_assistant_turns
            )

        elif agent_data.state == AgentState.PROCESSING_TOOLS:
            agent_data.state = await handle_processing_tools(
                agent_data, env, tokenizer, max_parallel_tools
            )

    # Return episode
    return Episode(
        token_ids=agent_data.response_ids,
        logprobs=agent_data.response_logprobs,
        response_mask=agent_data.response_mask,
        reward=env.get_final_reward()
    )


async def handle_pending(agent_data: AgentData, tokenizer):
    """Prepare prompt."""
    # Build prompt from messages
    prompt = tokenizer.apply_chat_template(
        agent_data.messages,
        add_generation_prompt=True
    )
    agent_data.prompt_ids = tokenizer.encode(prompt)
    return AgentState.GENERATING


async def handle_generating(
    agent_data: AgentData,
    policy: Generator,
    tokenizer,
    max_assistant_turns: int,
):
    """Generate response using Forge Generator."""
    # Generate
    prompt_text = tokenizer.decode(agent_data.prompt_ids)
    response = await policy.generate.route(
        prompt_text,
        sampling_params={"temperature": 0.7, "max_tokens": 256}
    )

    # Track turn count
    agent_data.assistant_turns += 1

    # Accumulate tokens
    agent_data.response_ids.extend(response.token_ids)
    agent_data.response_logprobs.extend(response.logprobs)
    agent_data.response_mask.extend([1] * len(response.token_ids))  # LLM output

    # Check termination
    if agent_data.assistant_turns >= max_assistant_turns:
        return AgentState.TERMINATED

    # Parse tool calls
    tool_calls = parse_tool_calls(response.text)
    agent_data.tool_calls = tool_calls

    if tool_calls:
        return AgentState.PROCESSING_TOOLS
    else:
        return AgentState.TERMINATED


async def handle_processing_tools(
    agent_data: AgentData,
    env: OpenEnv,
    tokenizer,
    max_parallel_tools: int,
):
    """Execute tools in PARALLEL (VERL pattern)."""

    # Create parallel tasks
    tool_tasks = [
        execute_tool_async(tool_call, env)
        for tool_call in agent_data.tool_calls[:max_parallel_tools]
    ]

    # Execute ALL tools concurrently
    tool_results = await asyncio.gather(*tool_tasks)

    # Add tool results to message history
    for tool_call, result in zip(agent_data.tool_calls, tool_results):
        # Add assistant message with tool call
        agent_data.messages.append({
            "role": "assistant",
            "tool_calls": [tool_call]
        })

        # Add tool result
        agent_data.messages.append({
            "role": "tool",
            "content": result
        })

    # Tokenize tool results
    tool_messages_text = tokenizer.apply_chat_template(
        [m for m in agent_data.messages if m["role"] == "tool"],
        add_generation_prompt=True
    )
    tool_tokens = tokenizer.encode(tool_messages_text)

    # Accumulate tool result tokens (with mask=0)
    agent_data.response_ids.extend(tool_tokens)
    agent_data.response_logprobs.extend([0.0] * len(tool_tokens))
    agent_data.response_mask.extend([0] * len(tool_tokens))  # DON'T train on tool results

    # Continue generation
    return AgentState.GENERATING


async def execute_tool_async(tool_call: dict, env: OpenEnv):
    """Execute single tool (async)."""
    result = env.execute_tool(tool_call)
    return result.observation.text


def parse_tool_calls(text: str) -> list[dict]:
    """Parse multiple tool calls from text."""
    matches = re.findall(r'<tool_call>(.*?)</tool_call>', text, re.DOTALL)
    tool_calls = []
    for match in matches:
        try:
            tool_calls.append(json.loads(match))
        except json.JSONDecodeError:
            continue
    return tool_calls
```

### Key Insights

✅ **Explicit state management**: Clear transitions between states
✅ **Parallel tool execution**: Multiple tools run concurrently (`asyncio.gather`)
✅ **Handles complex workflows**: Good for multi-tool scenarios
✅ **Response masking**: Built incrementally during state transitions

**Trade-offs:**
- **Pros**: Clear state flow, handles complexity well, parallel tools
- **Cons**: More complex than Patterns A/B, overkill for simple tasks
- **Best for**: Production systems with complex multi-step tool interactions

## Pattern D: Async Sample-Level Pipelining (NeMo-RL inspired)

### Summary

**What it is**: Each sample runs as independent async task. While one sample waits for tool execution, others continue generating. Maximum throughput.

**When to use**: Production system requiring maximum performance, have variable-length episodes, tool execution has latency.

### YAML Configuration

```yaml
# examples/tau2bench/grpo/async_pipeline_pattern.yaml
policy:
  type: "Generator"
  model_path: "Qwen/Qwen2.5-1.5B-Instruct"
  engine_args:
    # Note: Forge may handle async differently via Monarch
    # Check Forge docs for async configuration
    tensor_parallel_size: 1

rollout:
  sample_level_concurrency: true  # Enable per-sample pipelining
  concurrent_samples: 8  # Process 8 samples in parallel
  max_turns_per_sample: 10
```

### Complete Code

```python
# examples/tau2bench/grpo/async_pipeline_pattern.py

async def run_async_multi_sample_rollout(
    tasks: list[str],
    policy: Generator,
    tokenizer,
    env_factory: callable,  # Creates env per sample
):
    """
    NeMo-RL inspired: per-sample async tasks for pipelining.
    While Sample 1 waits for tool, Samples 2/3/4 continue generating.
    """

    # Create one async task PER SAMPLE
    sample_tasks = [
        asyncio.create_task(
            run_single_sample_async(
                sample_idx=i,
                task=task,
                policy=policy,
                tokenizer=tokenizer,
                env=env_factory()
            )
        )
        for i, task in enumerate(tasks)
    ]

    # Run ALL samples concurrently
    episodes = await asyncio.gather(*sample_tasks)

    return episodes


async def run_single_sample_async(
    sample_idx: int,
    task: str,
    policy: Generator,
    tokenizer,
    env: OpenEnv,
    max_turns: int = 10,
):
    """
    Complete lifecycle for ONE sample.
    Runs independently - while this sample waits, others continue.
    """
    messages = [{"role": "user", "content": task}]
    all_tokens = []
    all_logprobs = []
    response_mask = []
    done = False
    turn = 0

    while not done and turn < max_turns:
        # 1. Build prompt
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)

        # 2. Async generation (doesn't block other samples)
        response = await policy.generate.route(
            prompt,
            sampling_params={"temperature": 0.7, "max_tokens": 256}
        )

        # 3. Accumulate tokens
        all_tokens.extend(response.token_ids)
        all_logprobs.extend(response.logprobs)
        response_mask.extend([1] * len(response.token_ids))

        # 4. Parse tool call
        tool_call = parse_tool_call(response.text)

        if tool_call:
            # 5. Execute tool (async, but DOESN'T block other samples!)
            #    While THIS sample waits here, Sample 2/3/4 continue their generation
            tool_result = await execute_tool_async(env, tool_call)

            # Add to history
            messages.append({"role": "assistant", "tool_calls": [tool_call]})
            messages.append({"role": "tool", "content": tool_result})

            # Tokenize tool result
            tool_tokens = tokenizer.encode(tool_result)
            all_tokens.extend(tool_tokens)
            response_mask.extend([0] * len(tool_tokens))  # DON'T train

            done = env.is_done()
        else:
            messages.append({"role": "assistant", "content": response.text})
            done = True

        turn += 1

    # Get final reward
    reward = env.get_final_reward()

    return Episode(
        sample_idx=sample_idx,
        token_ids=all_tokens,
        logprobs=all_logprobs,
        response_mask=response_mask,
        reward=reward,
        num_turns=turn
    )


async def execute_tool_async(env: OpenEnv, tool_call: dict):
    """Execute tool without blocking other samples."""
    result = env.step(tool_call)
    return result.observation.text
```

### Why This Pipelining Matters

**Without pipelining (sequential):**
```
Sample 1: [Gen 10s] → [Tool 5s] → [Gen 10s] = 25s
Sample 2: [Gen 10s] → [Tool 5s] = 15s
Sample 3: [Gen 10s] = 10s
Total: 25 + 15 + 10 = 50s
```

**With NeMo-RL pipelining:**
```
Sample 1: [Gen 10s]──────────────┐        [Gen 10s]──────┐
                                 ↓                       ↓
                          [Tool 5s]               [Tool 5s]
Sample 2:     [Gen 10s]──────────┐  [Gen 10s]──┐
                                 ↓              ↓
                          [Tool 5s]      [Tool 5s]
Sample 3:         [Gen 10s]──────┐
                                 ↓
                          [Tool 5s]

Total: ~25s (longest sample) → 2x speedup!
```

**Downsides/Considerations:**
- **Memory**: All samples in flight simultaneously (more GPU memory)
- **Complexity**: Harder to debug (concurrent execution)
- **vLLM config**: May need `max_num_seqs` adjustment

**How to control:**
```yaml
# vLLM configuration
engine_args:
  max_num_seqs: 8  # Max concurrent sequences
  gpu_memory_utilization: 0.85  # Leave headroom
```

**Source of speedup estimates:**
- Based on NeMo-RL benchmarks with variable-length episodes
- 2-4x typical, up to 8x with high tool latency
- Depends on: tool execution time, episode length variance

### Key Insights

✅ **Maximum throughput**: Best performance for production
✅ **Non-blocking tool execution**: Fast samples don't wait for slow ones
✅ **Sample independence**: Each sample is its own async task
⚠️ **Higher memory usage**: All samples concurrent
⚠️ **More complex**: Harder to debug than sequential

**Trade-offs:**
- **Pros**: Best performance, maximum GPU utilization
- **Cons**: Memory usage, complexity, harder debugging
- **Best for**: Production scale, variable episode lengths, tool latency exists

## Pattern E: Native Tool Calling (Verifiers/PRIME-RL inspired)

### Summary

**What it is**: Use vLLM's native tool calling support (`enable_auto_tool_choice: true`), clean tool definition with type hints, automatic parsing.

**When to use**: Model supports native tool calling, want production-ready abstractions, avoid manual parsing.

### YAML Configuration

```yaml
# examples/tau2bench/grpo/native_tools_pattern.yaml
policy:
  type: "Generator"
  model_path: "Qwen/Qwen2.5-1.5B-Instruct"
  engine_args:
    # Enable vLLM native tool calling
    enable_auto_tool_choice: true
    tool_call_parser: "hermes"  # or "mistral", "llama", depends on model
    tensor_parallel_size: 1

tools:
  definition_style: "type_hints"  # Auto-generate schemas from functions
  auto_schema_generation: true
```

### Complete Code

**1. Clean Tool Definition**

```python
# examples/tau2bench/tools/tau2_tools.py

async def create_task(user_id: str, title: str, description: str = "", deadline: str = ""):
    """
    Create a new task.

    Args:
        user_id: ID of the user who owns the task
        title: Task title
        description: Optional task description
        deadline: Optional deadline (ISO format)

    Returns:
        Task creation result with task_id
    """
    # Implementation via OpenEnv
    result = env.execute_tool({
        "name": "create_task",
        "arguments": {
            "user_id": user_id,
            "title": title,
            "description": description,
            "deadline": deadline
        }
    })
    return result


async def update_task(task_id: str, status: str):
    """
    Update task status.

    Args:
        task_id: ID of the task to update
        status: New status (pending|completed|cancelled)

    Returns:
        Update result
    """
    result = env.execute_tool({
        "name": "update_task",
        "arguments": {"task_id": task_id, "status": status}
    })
    return result


# Auto-convert to OpenAI schemas
def convert_func_to_oai_tool(func: callable):
    """Convert type-hinted function to OpenAI tool schema."""
    import inspect
    sig = inspect.signature(func)

    parameters = {
        "type": "object",
        "properties": {},
        "required": []
    }

    for name, param in sig.parameters.items():
        param_type = str(param.annotation).replace("<class '", "").replace("'>", "")
        parameters["properties"][name] = {"type": param_type}
        if param.default == inspect.Parameter.empty:
            parameters["required"].append(name)

    return {
        "name": func.__name__,
        "description": func.__doc__.strip().split("\n")[0],
        "parameters": parameters
    }


# Generate schemas
tools = [create_task, update_task]
tool_schemas = [convert_func_to_oai_tool(t) for t in tools]
```

**2. Rollout with Native Parsing**

```python
# examples/tau2bench/grpo/native_tools_rollout.py

async def run_native_tool_calling(
    task: str,
    policy: Generator,
    tokenizer,
    tool_map: dict,  # {tool_name: function}
    tool_schemas: list[dict],
    max_turns: int = 10,
):
    """
    Verifiers-inspired: use vLLM native tool calling.
    """
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": task}
    ]

    all_tokens = []
    all_logprobs = []
    response_mask = []
    done = False
    turn = 0

    while not done and turn < max_turns:
        # 1. Format prompt WITH TOOLS (vLLM formats based on model)
        prompt = tokenizer.apply_chat_template(
            messages,
            tools=tool_schemas,  # vLLM handles formatting!
            add_generation_prompt=True
        )

        # 2. Generate (vLLM auto-parses tool calls)
        response = await policy.generate.route(prompt)

        # 3. Check if vLLM parsed tool calls
        #    (message.tool_calls populated by vLLM, not manual parsing!)
        if hasattr(response, 'tool_calls') and response.tool_calls:
            tool_call = response.tool_calls[0]

            # Execute tool
            tool_name = tool_call["function"]["name"]
            tool_args = json.loads(tool_call["function"]["arguments"])
            tool_result = await tool_map[tool_name](**tool_args)

            # Add to history
            messages.append({
                "role": "assistant",
                "tool_calls": [tool_call]
            })
            messages.append({
                "role": "tool",
                "content": str(tool_result),
                "tool_call_id": tool_call["id"]
            })

            # Accumulate tokens
            all_tokens.extend(response.token_ids)
            all_logprobs.extend(response.logprobs)
            response_mask.extend([1] * len(response.token_ids))

            # Tool result tokens
            tool_tokens = tokenizer.encode(str(tool_result))
            all_tokens.extend(tool_tokens)
            response_mask.extend([0] * len(tool_tokens))
        else:
            # Final answer
            messages.append({"role": "assistant", "content": response.text})
            all_tokens.extend(response.token_ids)
            all_logprobs.extend(response.logprobs)
            response_mask.extend([1] * len(response.token_ids))
            done = True

        turn += 1

    return Episode(
        token_ids=all_tokens,
        logprobs=all_logprobs,
        response_mask=response_mask,
        reward=compute_reward(messages)
    )
```

### Key Insights

✅ **No manual parsing**: vLLM does it automatically
✅ **Clean tool definition**: Just type-hinted Python functions
✅ **Production-ready**: Used by PRIME-RL, Verifiers
✅ **Model-specific formatting**: vLLM handles Qwen vs GPT vs Llama differences

**When to use:**
- Model is trained for native tool calling (e.g., fine-tuned with tool data)
- Want to avoid manual regex parsing
- Production system with well-defined tools
- Using Qwen, Mistral, Llama models with tool support

**Trade-offs:**
- **Pros**: Clean, reliable, no parsing bugs, production-ready
- **Cons**: Requires model support, less control over format
- **Best for**: Production systems with models trained for tool calling

---

**Summary of All Patterns:**

| Pattern | Complexity | Performance | Best For |
|---------|-----------|-------------|----------|
| **A: Simple Concat** | Low | OK | Prototypes, learning |
| **B: Tinker** 🎯 | Medium | Good | Production, research, clean code |
| **C: State Machine** | Medium-High | Good | Complex workflows, multiple tools |
| **D: Async Pipeline** | High | Best | Maximum throughput, production scale |
| **E: Native Tools** | Low-Medium | Good | Models with tool support, production |

**Recommendation for Forge:**
1. **Start with Pattern A** (simple concat) to learn
2. **Move to Pattern B** 🎯 (Tinker) for production - clean, maintainable
3. **Add Pattern D** (async pipeline) if bottlenecked on throughput
4. **Consider Pattern E** (native tools) if using tool-trained models

**Next**: Part 6 shows complete implementation plan for Forge.
