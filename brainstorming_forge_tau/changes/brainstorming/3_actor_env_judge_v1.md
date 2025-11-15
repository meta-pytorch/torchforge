My initial prompt:

```
you are given '/home/felipemello/forge/brainstorming_forge_tau/1_requirements_and_context.md''/home/felipemello/forge/brainstor
ming_forge_tau/4_examples_APIs.md' '/home/felipemello/forge/brainstorming_forge_tau/tutorials/3_forge_current_state.md''/home/fel
ipemello/forge/brainstorming_forge_tau/tutorials/4_forge_ideal_state.md'

I want you to explore 3 things
1. What happens if i need multiple envs for the same task, e.g. search the web AND code? In the 4_forge_ideal_state.md, there is
some basic map, but the way its structure only allows 1 env per task. Please reserach how the other frameworks handle this. Do it
 for all frameworks expect blackjack
2 Further more, what if my env needs to be an actor? For example, what if my coding env needs gpu access? Or what if i want to
create a stack of envs on 100 cpus = 100 envs, for example? It seems reasonable to leverage Forge + Monarch actor to do all of
the routing / async calls. Then should Forge have a wrapper for OpenEnv envs?
3. Envs are responsible for returning rewards. Its commmon to have llm as a judge. OpenEnv doesnt have an example for that,
afaik. Might be worth investigate their RFCs. How could we have llm as a judge using open env? The case where it just calls an
API is trivial. But what if my model is hosted locally, as an actor?


Each one of these can result in a very long research, however, the design on all 3 are related.

Here is my hint:
For 1, search how other libraries do it
For 2, take a good look at Forge APIs, starting from /home/felipemello/forge/apps/grpo/main.py, and also understand well OpenEnv environments. They have one for coding /home/felipemello/forge/OpenEnv/examples/coding_env_inference.py. Think about what would change if we had to execute this on GPU. Perhaps its also worth checking verifiers at least? Maybe the other frameworks too
For 3, definetely worth checking how other frameworks do llm as a judge, but now you also have a good understanding of Forge actors.

however, you **MUST** do it phased, i.e. research about a topic and update the doc, ONLY THEN, research about the next topic and
update the doc, etc. I DO NOT want you to do all of the writing at once.

if you have questions during the process, you can ask me or have a "open questions" at the end of the doc
```

----------

# Research: Actors, Environments, and LLM-as-a-Judge for Forge Multi-Turn RL

This document presents research on three interrelated design questions for implementing multi-turn tool calling in Forge:

1. **Multiple environments per task** (e.g., websearch AND coding)
2. **Environments as actors** (GPU access, distributed execution)
3. **LLM-as-a-judge for rewards** (local models as actors)

---

## 1. Multiple Environments Per Task

### Research Question
The current design in `4_forge_ideal_state.md` shows a basic 1:1 mapping between tasks and environments. However, real-world scenarios may require:
- **Single task, multiple tool domains**: e.g., "Research X and write code to analyze it" requires both websearch AND coding tools
- **Mixed training batches**: Training on websearch tasks AND coding tasks simultaneously for curriculum learning
- **Task-specific routing**: Different max_turns, tools, and reward functions per environment type

### How Other Frameworks Handle This

#### Framework 1: Tinker-Cookbook (Meta) - `CompositeDataset` Pattern P **RECOMMENDED**

**Location**: `tinker-cookbook/distillation/datasets.py:45-84`

**Core Abstraction**: `EnvGroupBuilder`

Every environment type implements a common interface:

```python
# tinker_cookbook/rl/types.py:64-108

class EnvGroupBuilder(ABC):
    """
    Builds a group of environments. Enables:
    - Multi-agent environments
    - GRPO groups (e.g., 8 copies for one problem)
    - Task-specific configurations
    """

    @abstractmethod
    async def make_envs(self) -> Sequence[Env]:
        """Create a group of environments (e.g., 8 copies for GRPO)"""
        pass

    async def compute_group_rewards(
        self, trajectory_group: list[Trajectory], env_group: Sequence[Env]
    ) -> list[tuple[float, Metrics]]:
        """Compute final reward looking at whole group (optional)"""
        return [(0.0, {}) for _ in trajectory_group]

    def logging_tags(self) -> list[str]:
        """Tags for logging (e.g., ['websearch'], ['coding'])"""
        return []
```

**Mixing Multiple Environment Types**: `CompositeDataset`

```python
# tinker_cookbook/distillation/datasets.py:45-84

class CompositeDataset:
    """Wraps multiple datasets and samples from each according to their groups_per_batch."""

    def __init__(self, datasets: List[RLDataset], groups_per_batch_list: List[int]):
        self.datasets = datasets
        self.groups_per_batch_list = groups_per_batch_list
        self.length = min(len(dataset) for dataset in datasets)

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

**Usage Example**:

```python
# Define two different environment types
websearch_dataset = WebSearchDataset(...)  # Returns EnvGroupBuilder for search tasks
coding_dataset = CodingDataset(...)        # Returns EnvGroupBuilder for coding tasks

# Mix them with explicit control over ratios
mixed_dataset = CompositeDataset(
    datasets=[websearch_dataset, coding_dataset],
    groups_per_batch_list=[50, 50]  # 50 websearch + 50 coding groups per batch
)

# Training loop handles both types transparently
for i_batch in range(num_batches):
    env_group_builders, dataset_indices = mixed_dataset.get_batch(i_batch)
    # env_group_builders contains 100 items: 50 websearch + 50 coding
    # Each builder knows its own tools, max_turns, reward function!
```

**Key advantages**:
-  **Decentralized design**: Each `EnvGroupBuilder` is self-contained
-  **Batch-level mixing**: Control exact ratios via `groups_per_batch_list`
-  **Separate logging**: Each builder has `logging_tags()` for domain-specific metrics
-  **Flexible**: Can easily add new environment types without changing training loop

---

#### Framework 2: Verifiers (Prime Intellect) - `EnvGroup` Pattern

**Location**: `verifiers/verifiers/envs/env_group.py`

**Core Abstraction**: `EnvGroup` as a Composite Environment

```python
class EnvGroup(Environment):
    """
    Environment group that acts as a mixture of multiple environments.
    Routes operations to appropriate sub-environments based on the 'task' column.
    """

    def __init__(
        self,
        envs: list[Environment],
        env_names: list[str] | None = None,
        **kwargs
    ):
        self.envs = envs
        self.env_names = env_names or [f"env_{i}" for i in range(len(envs))]

        # Create mapping for quick lookup
        self.env_map = {name: env for name, env in zip(self.env_names, self.envs)}

        # Concatenate datasets with task labels
        datasets = []
        for env, name in zip(self.envs, self.env_names):
            env_dataset = env.get_dataset().map(lambda x: {**x, "task": name})
            datasets.append(env_dataset)

        # Combine all datasets
        self.dataset = concatenate_datasets(datasets)
```

**Routing Logic**:

```python
async def rollout(self, client, model, prompt, task, ...):
    # Route to appropriate environment based on task field
    env = self.env_map[task]

    # Set tools for this task's environment
    if hasattr(env, "oai_tools") and env.oai_tools:
        info["oai_tools"] = env.oai_tools  # Different tools per env!

    # Execute rollout with task-specific environment
    completion, state = await env.rollout(client, model, prompt, ...)
```

**Custom Rubric for Mixed Rewards**:

```python
class EnvGroupRubric(Rubric):
    """Routes scoring to appropriate environment rubrics."""

    def __init__(self, env_map: Mapping[str, Environment]):
        self.env_map = env_map

        # Collect ALL unique reward function names across environments
        all_names_set = set()
        for env in env_map.values():
            all_names_set.update(env.rubric.get_reward_func_names())
        self.all_reward_names = sorted(list(all_names_set))

    async def score_rollout(self, prompt, completion, task, ...):
        # Initialize ALL reward names to 0.0
        metrics = {name: 0.0 for name in self.all_reward_names}

        # Get environment for this task
        env = self.env_map.get(task)

        # Score with environment's rubric
        env_results = await env.rubric.score_rollout(...)

        # Update only the relevant metrics
        for reward_name, score in env_results.metrics.items():
            if reward_name in metrics:
                metrics[reward_name] = score

        return RolloutScore(reward=env_results.reward, metrics=metrics)
```

**Usage Example**:

```python
# Define environments
websearch_env = vf.ToolEnv(
    dataset=websearch_dataset,
    tools=[search_pages, view_sections],
    max_turns=10
)

coding_env = vf.ToolEnv(
    dataset=coding_dataset,
    tools=[execute_code, debug_code],
    max_turns=15
)

# Combine into EnvGroup
env = EnvGroup(
    envs=[websearch_env, coding_env],
    env_names=["websearch", "coding"]
)

# Training: samples automatically routed to correct environment
generate_outputs = await env.generate(
    inputs=mixed_dataset,  # Has both "websearch" and "coding" task fields
    client=client,
    model=model_name
)
```

**Key advantages**:
-  **Centralized routing**: `EnvGroup` owns all sub-environments
-  **Sample-level routing**: Automatic based on `task` field in dataset
-  **Unified reward tracking**: All environments' metrics tracked in single dict
-  **Simple API**: Just pass task name, routing happens internally

---

#### Framework 3: NeMo-RL (Thinking Machines) - Dict-based Routing

**Location**: `RL/nemo_rl/experience/rollouts.py:226-275`

**Core Pattern**: Explicit `task_to_env` dictionary passed through rollout functions

```python
def calculate_rewards(
    batch: BatchedDataDict[DatumSpec],
    task_to_env: dict[str, EnvironmentInterface],
) -> EnvironmentReturn:
    """Calculate rewards for generated responses.

    Args:
        batch: Contains message_log with generated responses
        task_to_env: Dictionary mapping task names to environments
    """
    # Extract task names from batch
    task_names = batch["task_name"]

    # Group messages by task type
    task_groups: dict[str, list[tuple[int, LLMMessageLogType]]] = {}
    for i, task_name in enumerate(task_names):
        if task_name not in task_groups:
            task_groups[task_name] = []
        task_groups[task_name].append((i, messages[i]))

    # Calculate rewards for each task group concurrently
    futures = []
    future_to_indices = {}
    for task_name, group in task_groups.items():
        if task_name not in task_to_env:
            raise ValueError(f"No environment found for task type: {task_name}")

        # Extract messages for this group
        indices = [idx for idx, _ in group]
        group_messages = [msg for _, msg in group]

        # Submit to environment (Ray actor call)
        future = task_to_env[task_name].step.remote(group_messages, env_info)
        futures.append(future)
        future_to_indices[future] = indices

    # Wait for all environments to complete
    results = ray.get(futures)

    # Merge results back into batch order
    # ... (details omitted)
```

**Usage in Rollout**:

```python
async def run_async_multi_turn_rollout(
    policy_generation,
    input_batch,
    tokenizer,
    task_to_env: dict[str, EnvironmentInterface],  # Explicit dict
    max_seq_len,
    max_rollout_turns,
):
    # Each sample has a task_name field
    for i in range(batch_size):
        sample_state = {
            "message_log": input_batch["message_log"][i],
            "task_name": input_batch["task_name"][i],  # Used for routing
            ...
        }

    # During reward calculation
    env_output = calculate_rewards(active_batch, task_to_env)
```

**Setup**:

```python
# In main training script
task_to_env = {
    "websearch": WebSearchEnvironment(...),
    "coding": CodeEnvironment(...),
    "math": MathEnvironment(...),
}

# Pass to all rollout functions
rollout_output = run_async_multi_turn_rollout(
    policy, batch, tokenizer,
    task_to_env=task_to_env,  # Explicit parameter
    ...
)
```

**Key advantages**:
-  **Explicit and simple**: Just a dict, no magic
-  **Ray actor support**: Environments can be distributed actors
-  **Concurrent execution**: Groups tasks by type, processes in parallel
-  **Full control**: You manage the task_to_env mapping

**Limitations**:
- � Manual setup required (no helper classes like CompositeDataset)
- � Must ensure dataset has `task_name` field
- � No built-in batch mixing logic

---

#### Framework 4: VERL - Separate Config Files (Manual)

**Location**: `verl/examples/sglang_multiturn/config/tool_config/`

VERL uses **separate YAML files** for different tool configurations, but does NOT have built-in multi-environment support.

```yaml
# gsm8k_tool_config.yaml
tools:
  - class_name: "verl.tools.gsm8k_tool.Gsm8kTool"
    tool_schema:
      type: "function"
      function:
        name: "calc_gsm8k_reward"

# sandbox_fusion_tool_config.yaml
tools:
  - class_name: "verl.tools.sandbox_fusion_tools.SandboxFusionTool"
    tool_schema:
      type: "function"
      function:
        name: "code_interpreter"
```

**Approach**: Run separate training jobs with different configs OR manually load tools based on task.

**Limitation**: Not designed for mixed datasets out-of-the-box.

---

### Framework Comparison Table

| Framework | Multi-Env Support | Routing Method | Tools Per Env | Batch Mixing | Best For |
|-----------|------------------|----------------|---------------|--------------|----------|
| **Tinker (Meta)** |  Built-in `CompositeDataset` | Batch-level mixing |  Different tools |  Explicit ratios | **Production multi-env** |
| **Verifiers (Prime)** |  Built-in `EnvGroup` | `task` field in dataset |  Different tools |  Automatic | **Production multi-env** |
| **NeMo-RL** | � Manual dict | Dict lookup |  Different tools | � Manual | Custom routing logic |
| **VERL** | L No built-in | Separate configs | Config-based | L | Single env per job |

---

### Recommendation for Forge

**Use Tinker's `CompositeDataset` pattern** as the foundation, with inspiration from Verifiers' centralized routing:

```python
# 1. Define EnvGroupBuilder abstraction (similar to Tinker)
class EnvGroupBuilder(ABC):
    """Base class for creating groups of environments."""

    @abstractmethod
    async def make_envs(self, group_size: int) -> list[Environment]:
        """Create group_size environments for this task."""
        pass

    def logging_tags(self) -> list[str]:
        """Tags for separating metrics by environment type."""
        return []

# 2. Implement for different environment types
class WebSearchEnvBuilder(EnvGroupBuilder):
    def __init__(self, task_data, tools, max_turns=10):
        self.task_data = task_data
        self.tools = tools
        self.max_turns = max_turns

    async def make_envs(self, group_size: int):
        return [
            WebSearchEnv(self.task_data, self.tools, self.max_turns)
            for _ in range(group_size)
        ]

    def logging_tags(self):
        return ["websearch"]

class CodingEnvBuilder(EnvGroupBuilder):
    def __init__(self, task_data, tools, max_turns=15):
        self.task_data = task_data
        self.tools = tools
        self.max_turns = max_turns

    async def make_envs(self, group_size: int):
        return [
            CodingEnv(self.task_data, self.tools, self.max_turns)
            for _ in range(group_size)
        ]

    def logging_tags(self):
        return ["coding"]

# 3. Use CompositeDataset for mixing
mixed_dataset = CompositeDataset(
    datasets=[
        WebSearchDataset(...),  # Returns WebSearchEnvBuilder per sample
        CodingDataset(...),     # Returns CodingEnvBuilder per sample
    ],
    groups_per_batch_list=[50, 50]  # 50 of each per batch
)

# 4. In Forge rollout loop
async def continuous_rollouts():
    while True:
        env_group_builders, dataset_indices = mixed_dataset.get_batch(batch_idx)

        # Each builder knows its own type!
        for builder in env_group_builders:
            # Create environments (e.g., 8 for GRPO)
            envs = await builder.make_envs(group_size=8)

            # Play episodes with appropriate tools/config
            episodes = await play_episodes_with_envs(
                policy=policy,
                envs=envs,
                builder=builder  # Has logging_tags for metrics
            )
```

**Why this approach**:
-  **Different tools per environment**: Each builder configures its own tools
-  **Different max_turns**: WebSearch uses 10, Coding uses 15
-  **Flexible mixing ratios**: Control with `groups_per_batch_list`
-  **Separate metrics**: Each builder's `logging_tags()` enables domain-specific tracking
-  **Unified training loop**: No special casing needed
-  **Extensible**: Add new environment types without changing core logic

---

## References - Topic 1

### Tinker-Cookbook (Meta)
- `tinker-cookbook/tinker_cookbook/rl/types.py:64-108` - `EnvGroupBuilder` interface
- `tinker-cookbook/distillation/datasets.py:45-84` - `CompositeDataset` implementation
- `tinker-cookbook/distillation/train_on_policy.py` - Usage in training loop

### Verifiers (Prime Intellect)
- `verifiers/verifiers/envs/env_group.py` - `EnvGroup` and `EnvGroupRubric`
- `verifiers/tests/test_env_group.py` - Usage examples
- `verifiers/environments/math_group/math_group.py` - Concrete implementation

### NeMo-RL (Thinking Machines)
- `RL/nemo_rl/experience/rollouts.py:226-275` - `calculate_rewards` with task routing
- `RL/nemo_rl/experience/rollouts.py:780-880` - `run_async_multi_turn_rollout`
- `RL/nemo_rl/environments/interfaces.py` - `EnvironmentInterface`

### VERL
- `verl/examples/sglang_multiturn/config/tool_config/` - Tool configuration YAMLs
- `verl/verl/tools/utils/tool_registry.py` - Tool registry pattern

---

## 2. Environments as Actors (GPU Access & Distributed Execution)

### Research Question
What if an environment needs computational resources like GPUs? For example:
- **Coding environment with GPU**: Execute ML code that requires CUDA
- **Scaling to 100s of environments**: Need distributed execution across multiple CPUs/GPUs
- **LLM-based judging**: Reward functions that call local LLMs (covered in Topic 3)

Should Forge wrap OpenEnv with actors? How do other frameworks handle this?

### Forge Actor System (Monarch)

**How Forge actors work**:

Forge uses **Monarch** for distributed actor communication, not Ray. Key components:

```python
# src/forge/actors/generator.py:71-80

@dataclass
class Generator(ForgeActor):
    """Instance of a vLLM-based generator.

    This class manually recreates a vLLM engine that mirrors AsyncLLMEngine in v1.
    All communications are controlled via Monarch's proc meshes.

    Args:
        engine_args (EngineArgs): vLLM engine arguments
        sampling_params (SamplingParams): Sampling parameters
```

**Key pattern**: All Forge actors inherit from `ForgeActor` and use `@endpoint` decorators:

```python
from monarch.actor import endpoint
from forge.controller import ForgeActor

@dataclass
class Generator(ForgeActor):

    @endpoint(async_mode=True)
    async def generate(self, prompt: str, n: int = 1):
        """Async endpoint callable from other actors."""
        # Implementation...

# Usage from apps/grpo/main.py:
responses = await policy.generate.route(prompt, n=8)
```

**Important differences from Ray**:
- ✅ **Monarch proc meshes**: Not Ray actors
- ✅ **Route-based communication**: `.route()` instead of `.remote()`
- ✅ **Process mesh coordination**: Actors coordinate via shared process meshes

### OpenEnv Execution Model (Docker + HTTP)

**How OpenEnv currently works** (`OpenEnv/examples/coding_env_inference.py`):

```python
from envs.coding_env import CodingEnv, CodeAction

# 1. Launch Docker container with HTTP server
env = CodingEnv.from_docker_image(
    "coding-env:latest",
    ports={8000: 8000},  # Expose HTTP API
)

# 2. Call via HTTP (blocking)
result = env.step(CodeAction(code="print('hello')"))

# 3. Docker container handles execution internally
# - Sandboxed Python environment
# - No GPU access by default
# - Synchronous HTTP calls
```

**Key characteristics**:
- ✅ **Isolated execution**: Docker provides sandboxing
- ✅ **Language-agnostic**: Any Docker image works
- ❌ **No GPU support out-of-the-box**: Would need `--gpus all` in Docker
- ❌ **Synchronous**: Blocking HTTP calls
- ❌ **Not distributed**: Each Docker container runs on same host

### NeMo-RL Approach: Ray Actors for Environments ⭐ **RECOMMENDED for GPU**

**Location**: `RL/nemo_rl/environments/code_environment.py:49-261`

**Key Pattern**: Environments are Ray actors with worker pools

```python
# 1. Define worker as Ray remote class
@ray.remote
class CodeExecutionWorker:
    """Helper class to process individual code execution steps."""

    def __init__(self):
        # Create sandbox for code execution
        self.sandbox = {"__builtins__": ...}

    def execute_code(self, code: str):
        # Execute code in sandbox
        result = exec(code, self.sandbox)
        return result

# 2. Environment is also a Ray actor that manages workers
@ray.remote(max_restarts=-1, max_task_retries=-1)
class CodeEnvironment(EnvironmentInterface):
    """Main environment that coordinates workers."""

    def __init__(self, config: CodeEnvConfig):
        self.num_workers = config["num_workers"]

        # Create pool of Ray workers
        self.workers = [
            CodeExecutionWorker.remote()
            for _ in range(self.num_workers)
        ]

    def step(self, message_logs, env_info):
        # Batch work across workers
        chunked_work = chunk_list_to_workers(message_logs, self.num_workers)

        # Execute in parallel
        futures = [
            self.workers[i].execute_code.remote(chunk)
            for i, chunk in enumerate(chunked_work)
        ]

        # Wait for results
        results = ray.get(futures)
        return merge_results(results)

    def shutdown(self):
        for worker in self.workers:
            ray.kill(worker)
```

**Usage in training** (`RL/nemo_rl/experience/rollouts.py:260-274`):

```python
# Setup: Create environments as Ray actors
task_to_env = {
    "coding": CodeEnvironment.remote(config),  # Ray actor!
    "math": MathEnvironment.remote(config),
}

# During rollout: Call actor methods
env = task_to_env[task_name]
future = env.step.remote(messages, env_info)  # Async Ray call
results = ray.get(future)  # Wait for completion
```

**Key advantages**:
- ✅ **Parallel execution**: Worker pool distributes work
- ✅ **Non-blocking**: Ray futures enable async execution
- ✅ **Resource isolation**: Each actor can have dedicated resources
- ✅ **Fault tolerance**: `max_restarts=-1` handles crashes

### GPU-Enabled Environments (NeMo-RL Reward Model Example)

**Location**: `RL/nemo_rl/environments/reward_model_environment.py:71-180`

**Pattern**: Ray actor with GPU allocation via virtual cluster

```python
@ray.remote
class RewardModelEnvironment(EnvironmentInterface):
    """Environment that uses GPU for reward computation."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

        # Create Ray virtual cluster with GPU allocation
        self.virtual_cluster = RayVirtualCluster(
            name="grpo_reward_model_cluster",
            bundle_ct_per_node_list=[
                config["resources"]["gpus_per_node"]
            ] * config["resources"]["num_nodes"],
            use_gpus=True,  # <-- Enable GPU allocation
            num_gpus_per_node=config["resources"]["gpus_per_node"],
            max_colocated_worker_groups=1,
        )

        # Initialize LLM policy on GPUs
        self.reward_model_policy = Policy(
            cluster=self.virtual_cluster,  # Uses GPUs
            config=self.config,
            tokenizer=self.tokenizer,
            weights_path=checkpoint_path,
        )

    def step(self, message_logs, env_info):
        # Run inference on GPUs
        batch = self.preprocess_data(message_logs)
        scores = self.reward_model_policy.forward(batch)
        return EnvironmentReturn(rewards=scores, ...)
```

**Resource configuration**:

```python
config = {
    "resources": {
        "num_nodes": 2,
        "gpus_per_node": 4,  # 8 total GPUs
    },
    "model_name": "Skywork/Skywork-Reward-V2-Qwen3-0.6B",
    "precision": "bfloat16",
}

env = RewardModelEnvironment.remote(config)
```

**Key insights**:
- ✅ **GPU allocation**: Virtual cluster manages GPU resources
- ✅ **Multi-node support**: Can span multiple machines
- ✅ **LLM-as-a-judge**: Reward model runs as environment (see Topic 3)

### Verifiers Approach: CPU-Only Async

Verifiers does NOT use actors for environments. All execution is CPU-based async:

```python
# verifiers/envs/tool_env.py
class ToolEnv(MultiTurnEnv):
    async def env_response(self, messages, state):
        """Execute tools (CPU-bound, async I/O)."""
        tool_messages = []
        for tool_call in messages[-1]["tool_calls"]:
            # Execute tool (async Python function)
            result = await self.tool_map[tool_name](**tool_args)
            tool_messages.append({...})
        return tool_messages, state
```

**No GPU support**: Tools are Python functions, no GPU access needed.

### When to Use Actors for Environments

| Use Case | Solution | Framework Example |
|----------|----------|-------------------|
| **Simple tools (API calls, DB queries)** | No actors, async functions | Verifiers `ToolEnv` |
| **CPU-intensive (code exec, search)** | Ray/Monarch actors with worker pools | NeMo-RL `CodeEnvironment` |
| **GPU-required (LLM judge, model exec)** | Ray actors with GPU allocation | NeMo-RL `RewardModelEnvironment` |
| **Sandboxed execution** | OpenEnv Docker containers | OpenEnv `CodingEnv` |
| **Distributed at scale (100+ envs)** | Ray actors across multiple nodes | NeMo-RL with Ray cluster |

### Recommendation for Forge

**Hybrid Approach**: Support both OpenEnv (Docker) AND Monarch actors (for GPU)

#### Option 1: OpenEnv with Docker (Current, CPU-only)

```python
# Good for: Sandboxed execution, language-agnostic tools
# Limited by: No GPU, synchronous HTTP

from openenv import CodingEnv

env = CodingEnv.from_docker_image("coding-env:latest")
result = env.step(CodeAction(code="..."))
```

#### Option 2: Forge Actors for GPU Environments (NEW)

```python
# Good for: GPU access, async execution, distributed
# Limited by: Requires Forge/Monarch infrastructure

from forge.controller import ForgeActor
from monarch.actor import endpoint

@dataclass
class GPUCodingEnv(ForgeActor):
    """Coding environment with GPU support."""

    config: dict

    def __post_init__(self):
        # Initialize GPU resources
        self.device = torch.device("cuda")
        # Load ML model for code analysis
        self.model = load_model().to(self.device)

    @endpoint(async_mode=True)
    async def execute_code(self, code: str, context: dict):
        """Execute code with GPU-accelerated analysis."""
        # Run code in sandbox
        result = exec_in_sandbox(code)

        # Analyze with GPU model
        analysis = self.model(result)  # GPU inference

        return {
            "output": result,
            "analysis": analysis,
            "device": str(self.device)
        }

# Usage:
gpu_env = GPUCodingEnv(config={"device": "cuda:0"})
result = await gpu_env.execute_code.route(code="...")
```

#### Option 3: Wrapper Pattern (Forge Actor → OpenEnv)

```python
# Good for: Leverage OpenEnv ecosy stem + Forge async
# Limited by: Still no GPU in OpenEnv

@dataclass
class ForgeOpenEnvWrapper(ForgeActor):
    """Forge actor that wraps OpenEnv for async routing."""

    env_image: str

    def __post_init__(self):
        from envs.coding_env import CodingEnv
        self.env = CodingEnv.from_docker_image(self.env_image)

    @endpoint(async_mode=True)
    async def step(self, action):
        # Run OpenEnv in thread pool (blocking → async)
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            self.env.step,
            action
        )
        return result

    @endpoint(async_mode=False)
    def close(self):
        self.env.close()

# Usage:
env_actor = ForgeOpenEnvWrapper(env_image="coding-env:latest")
result = await env_actor.step.route(CodeAction(code="..."))
```

### Proposed Design for Forge

**1. Create `Environment` interface** (similar to NeMo-RL):

```python
from abc import ABC, abstractmethod
from forge.controller import ForgeActor

class Environment(ABC):
    """Base class for all Forge environments."""

    @abstractmethod
    async def reset(self) -> dict:
        """Reset environment, return initial observation."""
        pass

    @abstractmethod
    async def step(self, action: Any) -> dict:
        """Execute action, return observation, reward, done."""
        pass

    async def close(self):
        """Cleanup resources."""
        pass

# 2. CPU-based implementation (wraps OpenEnv)
class OpenEnvEnvironment(Environment):
    def __init__(self, docker_image: str):
        from envs import create_env_from_image
        self.env = create_env_from_image(docker_image)

    async def step(self, action):
        # Wrap sync call in async
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.env.step, action)

# 3. GPU-based implementation (Forge actor)
@dataclass
class GPUEnvironment(Environment, ForgeActor):
    config: dict

    def __post_init__(self):
        self.device = torch.device(self.config["device"])
        # Initialize GPU resources

    @endpoint(async_mode=True)
    async def step(self, action):
        # GPU computation here
        pass
```

**2. Environment factory** (route based on config):

```python
def create_environment(env_type: str, config: dict) -> Environment:
    if config.get("requires_gpu", False):
        return GPUEnvironment(config)
    elif config.get("use_docker", True):
        return OpenEnvEnvironment(config["docker_image"])
    else:
        return LocalEnvironment(config)

# Usage:
env = create_environment(
    "coding",
    config={
        "requires_gpu": True,
        "device": "cuda:0",
        "model": "codellama"
    }
)
```

### Key Takeaways

1. **OpenEnv is great for CPU sandboxing** but lacks GPU support
2. **Ray actors enable GPU environments** (see NeMo-RL reward model)
3. **Forge has Monarch actors** (not Ray), need to adapt patterns
4. **Worker pools enable parallelism** (distribute work across CPUs/GPUs)
5. **Environment abstraction enables flexibility** (swap OpenEnv ↔ GPU actor)

---

## References - Topic 2

### Forge (Monarch Actors)
- `src/forge/actors/generator.py:71-80` - Generator as ForgeActor
- `apps/grpo/main.py:82-98` - Actor usage with `.route()`
- `forge/controller/actor.py` - `ForgeActor` base class
- Monarch documentation (proc meshes, @endpoint)

### OpenEnv
- `OpenEnv/examples/coding_env_inference.py` - Docker-based execution
- `OpenEnv/src/core/http_env_client.py` - HTTP client interface
- `OpenEnv/src/envs/coding_env/` - Coding environment implementation

### NeMo-RL (Ray Actors)
- `RL/nemo_rl/environments/code_environment.py:49-261` - Ray actor with workers
- `RL/nemo_rl/environments/reward_model_environment.py:71-180` - GPU environment
- `RL/nemo_rl/experience/rollouts.py:226-275` - Environment routing
- `RL/nemo_rl/distributed/virtual_cluster.py` - RayVirtualCluster

### Verifiers
- `verifiers/envs/tool_env.py` - Async CPU-only execution
- No actor-based environments

---

## 3. LLM-as-a-Judge for Rewards

### Research Question
Rewards often require LLM-based judging (e.g., "Was this answer helpful?"). Key challenges:
- **API-based judge**: Simple case (OpenAI API, async calls)
- **Local model as judge**: Model hosted as actor with GPU (more complex)
- **Where does judging happen**: Environment or separate reward function?

How do other frameworks handle LLM-as-a-judge, especially when the judge is hosted locally as an actor?

### OpenEnv Pattern: Environment Returns Rewards

**Key insight from OpenEnv**: Environments are responsible for rewards via `.step()`.

```python
# OpenEnv core interface (src/core/client_types.py)

@dataclass
class StepResult:
    """Result from environment.step()"""
    observation: Observation
    reward: float | None  # <-- Environment computes this!
    done: bool
    info: dict

# Example usage
result = env.step(action)
print(f"Reward: {result.reward}")  # Environment already computed it
```

**Where reward logic lives**:
- **Simple envs**: Reward computed inside Docker container
- **Complex envs**: Could call LLM API inside environment

**Limitation**: OpenEnv examples don't show LLM-as-a-judge patterns. All examples use rule-based rewards (e.g., poker chips, game scores).

### Verifiers Pattern: Separate Rubric with API-Based Judge ⭐ **RECOMMENDED for API**

**Location**: `verifiers/verifiers/rubrics/judge_rubric.py:31-145`

**Core Abstraction**: `JudgeRubric` separates reward computation from environment

```python
from openai import AsyncOpenAI
from verifiers.rubrics.rubric import Rubric

class JudgeRubric(Rubric):
    """Uses an LLM to judge if response matches ground truth."""

    def __init__(
        self,
        judge_client: AsyncOpenAI | None = None,
        judge_model: str = "gpt-4.1-nano",  # API model
        judge_sampling_args: dict[str, Any] | None = None,
        judge_prompt: str = DEFAULT_JUDGE_PROMPT,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.judge_client = judge_client or AsyncOpenAI()
        self.judge_model = judge_model
        self.judge_prompt = judge_prompt
        self.judge_sampling_args = judge_sampling_args or {}

    async def judge(
        self,
        prompt: Messages,
        completion: Messages,
        answer: str,  # Ground truth
        state: State,
        **kwargs,
    ) -> str:
        """Call LLM API to judge correctness."""
        # Extract question and response
        question = prompt[-1]["content"]
        response = self.parser.parse_answer(completion)

        # Format judge prompt
        judge_prompt = self.judge_prompt.format(
            question=question,
            answer=answer,
            response=response
        )

        # Check cache (avoid redundant API calls)
        cached = state.get("judge_response", {})
        if judge_prompt in cached:
            return cached[judge_prompt]

        # Call LLM API asynchronously
        judge_response = await self.judge_client.chat.completions.create(
            model=self.judge_model,
            messages=[{"role": "user", "content": judge_prompt}],
            **self.judge_sampling_args,
        )
        judge_response = str(judge_response.choices[0].message.content)

        # Cache result
        cached[judge_prompt] = judge_response
        state["judge_response"] = cached
        return judge_response

    async def score_rollout(self, prompt, completion, answer, state, ...):
        """Convert judge output to numeric reward."""
        judge_output = await self.judge(prompt, completion, answer, state)

        # Parse yes/no to 1.0/0.0
        reward = 1.0 if "yes" in judge_output.lower() else 0.0

        return RolloutScore(reward=reward, metrics={...})
```

**Default judge prompt**:

```python
DEFAULT_JUDGE_PROMPT = """Given a ground truth answer \
and a response, determine if the response is correct.

Question:
```
{question}
```

Ground truth answer:
```
{answer}
```

Response:
```
{response}
```

Respond either "yes" or "no" only."""
```

**Usage**:

```python
import verifiers as vf
from verifiers.rubrics import JudgeRubric

# Create environment with LLM judge
env = vf.ToolEnv(
    dataset=my_dataset,
    tools=[search_tool, calculator],
    rubric=JudgeRubric(
        judge_model="gpt-4.1-mini",
        judge_client=AsyncOpenAI(api_key=...),
        judge_sampling_args={"temperature": 0.0, "max_tokens": 10}
    )
)

# During rollout, rubric automatically calls judge
outputs = await env.generate(inputs=batch, client=client, model=model)
# outputs.rewards computed via LLM judge!
```

**Key advantages**:
- ✅ **Separation of concerns**: Rubric (reward) separate from Environment (tools)
- ✅ **Async API calls**: Non-blocking, can handle many concurrent requests
- ✅ **Caching**: Avoid redundant API calls for same prompt
- ✅ **Error handling**: Graceful handling of rate limits, timeouts, API errors
- ✅ **Flexible**: Easy to swap judge models or prompts

**Limitations**:
- ⚠️ **API-only**: Requires OpenAI-compatible API (can't use local actor model)
- ⚠️ **Latency**: API calls add latency to rollout

### NeMo-RL Pattern: Reward Model as Environment Actor ⭐ **RECOMMENDED for Local GPU**

**Location**: `RL/nemo_rl/environments/reward_model_environment.py:71-256`

**Key Pattern**: Reward model IS the environment, runs as Ray actor with GPUs

```python
@ray.remote
class RewardModelEnvironment(EnvironmentInterface):
    """Environment = Reward model with GPU."""

    def __init__(self, config: Dict[str, Any]):
        # Create Ray virtual cluster with GPUs
        self.virtual_cluster = RayVirtualCluster(
            bundle_ct_per_node_list=[config["resources"]["gpus_per_node"]]
                * config["resources"]["num_nodes"],
            use_gpus=True,
            num_gpus_per_node=config["resources"]["gpus_per_node"],
        )

        # Load reward model on GPUs
        self.reward_model_policy = Policy(
            cluster=self.virtual_cluster,
            config=self.config,
            tokenizer=self.tokenizer,
            weights_path=checkpoint_path,
        )

    def step(self, message_logs: List[LLMMessageLogType], env_info):
        """
        Score conversations with reward model.

        Args:
            message_logs: Full conversation history per sample
            env_info: Additional environment metadata

        Returns:
            EnvironmentReturn with rewards from model
        """
        # Tokenize conversations
        batch = self.preprocess_data(message_logs)

        # Run reward model inference on GPU
        scores = self.reward_model_policy.forward(batch)

        # Return rewards
        return EnvironmentReturn(
            rewards=scores,
            terminateds=torch.ones(len(message_logs), dtype=torch.bool),
            observations=[""] * len(message_logs),
            metadata=[{}] * len(message_logs),
            next_stop_strings=[None] * len(message_logs),
            answers=[""] * len(message_logs),
        )
```

**Configuration**:

```python
reward_model_config = {
    "enabled": True,
    "model_name": "Skywork/Skywork-Reward-V2-Qwen3-0.6B",
    "precision": "bfloat16",
    "batch_size": 32,
    "checkpoint_path": "/path/to/checkpoint",
    "resources": {
        "num_nodes": 1,
        "gpus_per_node": 2,  # 2 GPUs for reward model
    },
    "dtensor_cfg": {"enabled": True},
}

# Create reward environment as Ray actor
reward_env = RewardModelEnvironment.remote(reward_model_config)
```

**Usage in training**:

```python
# Setup: Reward model is just another environment
task_to_env = {
    "math": MathEnvironment.remote(...),
    "coding": CodeEnvironment.remote(...),
    "reward_scoring": RewardModelEnvironment.remote(...),  # Judge environment!
}

# During rollout: Call like any other environment
env_output = calculate_rewards(batch, task_to_env)
# Internally routes to RewardModelEnvironment.step()
```

**Key advantages**:
- ✅ **GPU acceleration**: Full GPU access for reward model
- ✅ **Batch inference**: Efficient batched scoring
- ✅ **Ray actor**: Distributed, fault-tolerant, async
- ✅ **Consistent interface**: Same as other environments (EnvironmentInterface)
- ✅ **Multi-node**: Can distribute across multiple machines

**Key insight**: **Reward model = Environment**. It "judges" trajectories like a tool env executes tools.

### VERL Pattern: Standalone Reward Model Manager

**Location**: `verl/verl/experimental/reward/reward_model.py:32-137`

**Pattern**: Separate reward model service with HTTP router

```python
class RewardModelManager:
    """Manages reward model servers with load balancing."""

    def __init__(self, config: RewardModelConfig, worker_group=None):
        self.config = config
        self._initialize_llm_servers()  # Spawn vLLM/SGLang servers
        self._initialize_router()       # Load balancer

    def _initialize_llm_servers(self):
        """Spawn multiple reward model replicas."""
        rollout_world_size = self.config.rollout.tensor_model_parallel_size
        num_replicas = self.config.n_gpus // rollout_world_size

        # Create replica servers
        self.rollout_replicas = [
            rollout_replica_class(
                replica_rank=rank,
                config=self.config.rollout,
                model_config=model_config,
                gpus_per_node=self.config.n_gpus_per_node,
                is_reward_model=True,  # Special flag
            )
            for rank in range(num_replicas)
        ]

        # Initialize servers (colocated or standalone)
        if self.worker_group:
            self._run_all([s.init_colocated(self.worker_group) for s in self.rollout_replicas])
        else:
            self._run_all([s.init_standalone() for s in self.rollout_replicas])

    def _initialize_router(self):
        """Create HTTP router to load balance across replicas."""
        worker_urls = [f"http://{addr}" for addr in self.server_addresses]
        self.router_address, _ = launch_router_process(worker_urls=worker_urls)

    async def chat_complete(self, chat_complete_request: dict):
        """Call reward model via HTTP (OpenAI-compatible)."""
        url = f"http://{self.router_address}/v1/chat/completions"
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=chat_complete_request) as resp:
                output = await resp.json()
                return ChatCompletion(**output)
```

**Usage**:

```python
# Setup reward model manager
reward_mgr = RewardModelManager(
    config=RewardModelConfig(
        model={"path": "Skywork/Skywork-Reward-V2-Qwen3-0.6B"},
        rollout={"tensor_model_parallel_size": 2},
        n_gpus_per_node=4,
        nnodes=1,
    )
)

# Call reward model
async def score_trajectory(messages):
    request = {
        "model": "Skywork/Skywork-Reward-V2-Qwen3-0.6B",
        "messages": messages,
        "temperature": 0.0,
    }
    response = await reward_mgr.chat_complete(request)
    return response.choices[0].message.content
```

**Key advantages**:
- ✅ **Load balancing**: Router distributes across replicas
- ✅ **OpenAI-compatible**: Standard HTTP API
- ✅ **Colocated or standalone**: Flexible deployment
- ✅ **Multiple replicas**: High throughput

**Difference from NeMo-RL**: Standalone service, not part of environment interface.

### Comparison: Where Does LLM Judge Live?

| Framework | Judge Location | Implementation | GPU Support | API | Best For |
|-----------|---------------|----------------|-------------|-----|----------|
| **Verifiers** | `Rubric` (separate from env) | `AsyncOpenAI` client | ❌ API-only | OpenAI | API-based judging |
| **NeMo-RL** | `RewardModelEnvironment` (IS the env) | Ray actor with Policy | ✅ Full GPU | Ray `.remote()` | Local GPU judge |
| **VERL** | `RewardModelManager` (standalone) | HTTP server + router | ✅ Full GPU | HTTP (OpenAI-compatible) | Standalone service |
| **OpenEnv** | Environment (implicit) | Not shown in examples | ⚠️ Depends on impl | Depends | Rule-based rewards |

### Proposed Design for Forge

**Option 1: Rubric Pattern (API-based judge)** - Similar to Verifiers

```python
from openai import AsyncOpenAI
from forge.data.rewards import BaseReward

class LLMJudgeReward(BaseReward):
    """Reward function using LLM judge via API."""

    def __init__(
        self,
        judge_model: str = "gpt-4.1-mini",
        judge_client: AsyncOpenAI | None = None,
        judge_prompt: str = DEFAULT_PROMPT,
    ):
        self.judge_model = judge_model
        self.judge_client = judge_client or AsyncOpenAI()
        self.judge_prompt = judge_prompt

    async def evaluate_response(
        self,
        prompt: str,
        response: str,
        target: str,
    ) -> float:
        """Call LLM API to judge response."""
        judge_input = self.judge_prompt.format(
            question=prompt,
            answer=target,
            response=response
        )

        completion = await self.judge_client.chat.completions.create(
            model=self.judge_model,
            messages=[{"role": "user", "content": judge_input}],
            temperature=0.0,
            max_tokens=10,
        )

        judge_output = completion.choices[0].message.content.lower()
        return 1.0 if "yes" in judge_output else 0.0

# Usage in apps/grpo/main.py:
reward_actor = LLMJudgeReward(
    judge_model="gpt-4.1-mini",
    judge_client=AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
)

# During rollout
episode.reward = await reward_actor.evaluate_response(
    prompt=prompt,
    response=response.text,
    target=target
)
```

**Advantages**:
- ✅ Minimal changes to existing `apps/grpo/main.py`
- ✅ Works with any OpenAI-compatible API
- ✅ Simple to implement

**Limitations**:
- ❌ Requires API access (cost, latency)
- ❌ Cannot use local Forge actors

---

**Option 2: Forge Actor Judge (Local GPU)** ⭐ **RECOMMENDED**

```python
from dataclasses import dataclass
from forge.controller import ForgeActor
from monarch.actor import endpoint
from vllm.transformers_utils.tokenizer import get_tokenizer

@dataclass
class LLMJudgeActor(ForgeActor):
    """LLM judge running on GPU via Forge actor."""

    model_name: str = "Skywork/Skywork-Reward-V2-Qwen3-0.6B"
    engine_args: dict = field(default_factory=dict)

    def __post_init__(self):
        # Initialize vLLM engine on GPU (similar to Generator)
        from vllm.v1.engine import EngineCoreRequest
        self.tokenizer = get_tokenizer(self.model_name)
        # ... initialize vLLM engine (see Generator actor)

    @endpoint(async_mode=True)
    async def judge_trajectory(
        self,
        messages: list[dict],
        ground_truth: str | None = None
    ) -> float:
        """
        Judge a full trajectory (multi-turn conversation).

        Args:
            messages: Conversation history (OpenAI format)
            ground_truth: Expected answer (optional)

        Returns:
            Reward score (float)
        """
        # Format judge prompt
        judge_prompt = self._format_judge_prompt(messages, ground_truth)

        # Generate with vLLM
        response = await self.generate(judge_prompt, max_tokens=10)

        # Parse response to reward
        reward = self._parse_reward(response.text)
        return reward

    def _format_judge_prompt(self, messages, ground_truth):
        # Extract final response
        final_response = messages[-1]["content"]

        if ground_truth:
            return f"""Given the conversation and ground truth, rate the quality of the final answer.

Conversation:
{self._format_messages(messages)}

Ground Truth: {ground_truth}

Rate from 0.0 (incorrect) to 1.0 (perfect). Respond with just a number."""
        else:
            return f"""Rate the quality of this conversation from 0.0 (poor) to 1.0 (excellent).

Conversation:
{self._format_messages(messages)}

Respond with just a number between 0.0 and 1.0."""

    def _parse_reward(self, text: str) -> float:
        """Extract numeric reward from judge output."""
        import re
        match = re.search(r'(\d+\.?\d*)', text)
        if match:
            reward = float(match.group(1))
            return max(0.0, min(1.0, reward))  # Clamp to [0, 1]
        return 0.0  # Default if parsing fails

# Setup in apps/grpo/main.py:
llm_judge = LLMJudgeActor(
    model_name="Skywork/Skywork-Reward-V2-Qwen3-0.6B",
    engine_args={
        "model": "Skywork/Skywork-Reward-V2-Qwen3-0.6B",
        "tensor_parallel_size": 1,
        "dtype": "bfloat16",
    }
)

# During multi-turn rollout (after episode completes):
episode.reward = await llm_judge.judge_trajectory.route(
    messages=messages,  # Full conversation
    ground_truth=task.target  # Optional
)
```

**Advantages**:
- ✅ **GPU acceleration**: vLLM on local GPUs
- ✅ **Consistent with Forge**: Uses Monarch actors like Generator
- ✅ **Batch inference**: Can judge multiple trajectories in parallel
- ✅ **No API costs**: Runs locally

---

**Option 3: Hybrid (API + Local)**

Allow users to choose via config:

```python
# apps/grpo/main.py

if config.reward.type == "llm_judge_api":
    reward_actor = LLMJudgeReward(
        judge_model=config.reward.model,
        judge_client=AsyncOpenAI(api_key=config.reward.api_key)
    )
elif config.reward.type == "llm_judge_local":
    reward_actor = LLMJudgeActor(
        model_name=config.reward.model,
        engine_args=config.reward.engine_args
    )
elif config.reward.type == "rule_based":
    reward_actor = MathReward()  # Existing
else:
    raise ValueError(f"Unknown reward type: {config.reward.type}")

# Unified interface:
episode.reward = await reward_actor.evaluate_response.route(...)
```

### When to Use Each Pattern

| Pattern | When to Use | Example |
|---------|------------|---------|
| **API-based (Verifiers)** | Quick experiments, proprietary models (GPT-4) | Research prototyping |
| **Local GPU actor (NeMo-RL)** | Production, custom models, cost-sensitive | Training at scale |
| **Standalone service (VERL)** | Shared judge across multiple training jobs | Multi-user cluster |
| **Rule-based** | Deterministic rewards (math, code correctness) | GSM8K, MBPP |

### Key Takeaways

1. **Verifiers separates reward (Rubric) from environment** - clean abstraction
2. **NeMo-RL treats reward model as environment** - unified interface
3. **VERL uses standalone HTTP service** - good for sharing across jobs
4. **Forge should support both API and local GPU judges** - flexibility
5. **LLM judge = just another Forge actor** - consistent with Generator pattern

---

## References - Topic 3

### Verifiers (API-based)
- `verifiers/rubrics/judge_rubric.py:31-145` - `JudgeRubric` implementation
- `verifiers/rubrics/rubric.py` - Base `Rubric` class
- `verifiers/envs/tool_env.py` - How rubric is used in environment

### NeMo-RL (GPU actor)
- `RL/nemo_rl/environments/reward_model_environment.py:71-256` - Reward model as environment
- `RL/nemo_rl/models/policy/lm_policy.py` - Policy wrapper for reward models
- `RL/nemo_rl/distributed/virtual_cluster.py` - GPU resource management

### VERL (Standalone service)
- `verl/verl/experimental/reward/reward_model.py:32-137` - `RewardModelManager`
- `verl/verl/experimental/reward/router/` - HTTP router implementation
- `verl/verl/workers/rollout/replica.py` - Rollout replica servers

### OpenEnv
- `OpenEnv/src/core/client_types.py` - `StepResult` with reward field
- `OpenEnv/examples/` - Various examples with rule-based rewards
- No LLM-as-a-judge examples found

### Forge (Existing Patterns)
- `src/forge/actors/generator.py` - Generator actor (template for judge actor)
- `apps/grpo/main.py:385-398` - Current reward computation
- `forge/data/rewards.py` - `MathReward`, `ThinkingReward` (rule-based)

---

## Open Questions

After completing this research, here are remaining design questions:

1. **Multi-environment composition**: If a task needs websearch AND coding, should we:
   - Create a composite environment that manages both? (Tinker `EnvGroupBuilder`)
   - Route to different environments sequentially? (NeMo-RL `task_to_env`)
   - Allow environments to call other environments? (Not seen in any framework)

2. **GPU environment scaling**: For 100 coding environments on 8 GPUs:
   - Should each environment be a separate Forge actor? (High overhead)
   - Should we pool environments and route requests? (More complex)
   - Can Monarch handle 100 concurrent actors efficiently?

3. **LLM judge batching**: When judging 64 trajectories:
   - Should judge actor batch internally? (More efficient)
   - Should caller batch before calling judge? (More flexible)
   - How to handle variable-length conversations?

4. **Reward timing**: When does judging happen?
   - After each turn? (Per-step rewards, like OpenEnv)
   - After full episode? (Sparse reward, like current GRPO)
   - Both? (Hybrid approach)

5. **Environment lifecycle with Forge actors**:
   - How to properly initialize/shutdown Docker environments wrapped as actors?
   - Should `ForgeOpenEnvWrapper` create Docker containers on `__post_init__` or lazily?
   - How to handle Docker container cleanup when actor dies?

---

*Research completed for all 3 topics.*
