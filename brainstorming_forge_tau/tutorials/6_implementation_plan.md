# Part 6: Implementation Plan for Forge

This part shows how to integrate multi-turn tool calling into Forge GRPO.

## 6.1 High-Level Strategy

**Approach:**
1. Start with Pattern A (simple) to get multi-turn working
2. Add response masking
3. Refactor to Pattern B (Tinker-style) for clean code
4. Optimize with async (Pattern D) if needed

**Focus:**
- Reusable core utilities in `forge/`
- Task-specific code in `examples/tau2bench/`
- OpenEnv integration for training
- Tau2Bench for evaluation

## 6.2 Overall System Context

### Full System Configuration

```yaml
# examples/tau2bench/grpo/config.yaml

# Generator (vLLM)
policy:
  type: "Generator"
  model_path: "Qwen/Qwen2.5-1.5B-Instruct"
  engine_args:
    tensor_parallel_size: 1
    gpu_memory_utilization: 0.9
    max_model_len: 2048
    enable_prefix_caching: true  # Helps with multi-turn

# Trainer
trainer:
  type: "TitanTrainer"
  learning_rate: 1e-5
  beta: 0.1  # KL penalty
  batch_size: 32

# Replay Buffer
replay_buffer:
  type: "ReplayBuffer"
  capacity: 10000
  min_size: 100

# Reference Model
ref_model:
  type: "ReferenceModel"
  model_path: "Qwen/Qwen2.5-1.5B-Instruct"

# Rollout Configuration
rollout:
  group_size: 8  # GRPO group
  num_rollout_threads: 4  # Parallel rollout workers
  max_turns_per_episode: 10
  use_response_masking: true

# OpenEnv for Training
openenv:
  base_url: "http://localhost:8001"
  timeout: 30

# Tau2Bench for Evaluation
tau2bench:
  domain: "mock"
  task_split: "train"  # or "test" for final eval
```

### General Rollout Loop Structure

```python
# examples/tau2bench/grpo/main.py

async def continuous_rollouts(
    policy: Generator,
    trainer: TitanTrainer,
    replay_buffer: ReplayBuffer,
    ref_model: ReferenceModel,
    reward_actor: RewardActor,
    dataloader: DataLoader,
    config: dict,
):
    """
    Main rollout loop - where play_task() is called.
    Adapted from apps/grpo/main.py for multi-turn.
    """

    while True:
        # 1. Sample tasks from Tau2Bench dataset
        tasks = await sample_tasks(dataloader, batch_size=config.rollout.group_size)

        # 2. Run multi-turn episodes (THIS IS NEW!)
        episodes = []
        for task in tasks:
            episode = await play_task(
                task=task,
                policy=policy,
                tokenizer=tokenizer,
                env=create_env(),
                max_turns=config.rollout.max_turns_per_episode
            )
            episodes.append(episode)

        # 3. Get reference logprobs (existing Forge code)
        ref_logprobs = await get_reference_logprobs(episodes, ref_model)

        # 4. Compute advantages (group-relative)
        advantages = compute_advantages([ep.reward for ep in episodes])

        # 5. Add episodes to replay buffer
        for episode, advantage in zip(episodes, advantages):
            episode.advantage = advantage
            await replay_buffer.add.call_one(episode)


async def continuous_training(
    trainer: TitanTrainer,
    policy: Generator,
    replay_buffer: ReplayBuffer,
    config: dict,
):
    """Training loop (mostly unchanged)."""

    while True:
        # Sample batch
        batch = await replay_buffer.sample(config.trainer.batch_size)

        # Train with response masking (NEW!)
        await trainer.train_step(
            inputs=batch["inputs"],
            targets=batch["targets"],
            advantages=batch["advantages"],
            response_mask=batch["response_mask"]  # NEW!
        )

        # Update weights
        version = await trainer.push_weights()
        await policy.update_weights(version)
```

### Code Organization Philosophy

**Decision Framework: Core vs Tau2Bench-Specific?**

Ask these questions for each function:
1. **Reusable?** Can other benchmarks/tasks use this?
2. **Tau2-specific?** Uses Tau2Bench APIs or formats?
3. **Valuable to others?** Would users find this useful?
4. **Domain logic or infrastructure?** Business logic vs technical infrastructure?

**If YES to questions 1, 3, 4**: → **Core** (`forge/`)
**If YES to question 2**: → **Task-specific** (`examples/tau2bench/`)

**Core Utilities** (reusable):
```
forge/
├── utils/
│   ├── parsing.py           # parse_tool_call(), parse_response()
│   ├── prompts.py           # format_system_prompt() template builder
│   ├── renderers.py         # Renderer base class, Qwen3Renderer
│   └── masking.py           # build_response_mask(), apply_mask()
├── rollouts/
│   └── multiturn.py         # play_task(), do_rollout()
├── environments/
│   └── tool_env.py          # ToolEnv base class, OpenEnvToolEnv adapter
└── data/
    └── trajectory_processing.py  # trajectory_to_episode()
```

**Tau2Bench-Specific**:
```
examples/tau2bench/grpo/
├── main.py                  # Training script (continuous_rollouts, etc.)
├── tau2_env.py              # Tau2Bench environment adapter
├── tau2_utils.py            # Tau2-specific utilities (task loading, scoring)
├── config.yaml              # Configuration
└── prompts.py               # Task-specific prompt templates
```

## 6.3 Core Components Implementation

### play_task() - The Multi-turn Loop

**Classification:** ✅ **Core** (`forge/rollouts/multiturn.py`)

**Reasoning:**
- Reusable across different environments
- Generic multi-turn logic
- Not Tau2Bench-specific

```python
# forge/rollouts/multiturn.py

async def play_task(
    task: str,
    policy: Generator,
    tokenizer,
    env: ToolEnv,
    max_turns: int = 10,
) -> Episode:
    """
    Generic multi-turn tool calling loop.
    Works with any ToolEnv-compatible environment.
    """
    # Initialize
    messages = [{"role": "user", "content": task}]
    all_tokens = []
    all_logprobs = []
    response_mask = []
    done = False
    turn = 0

    # Multi-turn loop
    while not done and turn < max_turns:
        # 1. Format prompt
        prompt = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False
        )

        # 2. Generate
        response = await policy.generate.route(
            prompt,
            sampling_params={"temperature": 0.7, "max_tokens": 256}
        )

        # 3. Parse tool call
        tool_call = parse_tool_call(response.text)  # From forge.utils.parsing

        # 4. Execute or finalize
        if tool_call:
            # Execute via environment
            result = await env.execute_tool(tool_call)

            # Update messages
            messages.append({
                "role": "assistant",
                "tool_calls": [tool_call]
            })
            messages.append({
                "role": "tool",
                "content": result
            })

            # Accumulate tokens
            all_tokens.extend(response.token_ids)
            all_logprobs.extend(response.logprobs)
            response_mask.extend([1] * len(response.token_ids))  # Train on LLM

            # Tool result tokens
            tool_tokens = tokenizer.encode(result)
            all_tokens.extend(tool_tokens)
            response_mask.extend([0] * len(tool_tokens))  # Don't train

            done = env.is_done()
        else:
            # Final answer
            messages.append({"role": "assistant", "content": response.text})
            all_tokens.extend(response.token_ids)
            all_logprobs.extend(response.logprobs)
            response_mask.extend([1] * len(response.token_ids))
            done = True

        turn += 1

    # Get reward
    reward = env.get_final_reward()

    return Episode(
        token_ids=all_tokens,
        logprobs=all_logprobs,
        response_mask=response_mask,
        reward=reward,
        num_turns=turn,
        messages=messages  # For debugging
    )
```

### parse_response() - Tool Call Detection

**Classification:** ✅ **Core** (`forge/utils/parsing.py`)

**Reasoning:** Generic response parsing, reusable

```python
# forge/utils/parsing.py

def parse_tool_call(text: str) -> dict | None:
    """
    Parse tool call from model output.
    Supports multiple formats.
    """
    # Format 1: <function_call>...</function_call>
    match = re.search(r'<function_call>(.*?)</function_call>', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Format 2: <tool_call>...</tool_call>
    match = re.search(r'<tool_call>(.*?)</tool_call>', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    return None


def has_tool_call(text: str) -> bool:
    """Check if text contains a tool call."""
    return ('<function_call>' in text or
            '<tool_call>' in text or
            '{"name":' in text)  # JSON format
```

### format_system_prompt() - Prompt with Tools

**Classification:** 🔀 **Hybrid**

**Reasoning:**
- Core template builder: `forge/utils/prompts.py`
- Task-specific templates: `examples/tau2bench/prompts.py`

```python
# forge/utils/prompts.py (Core)

def build_tool_calling_system_prompt(
    tools: list[dict],
    format_style: str = "tags",
) -> str:
    """
    Generic tool calling system prompt builder.
    """
    # Format tool schemas
    tool_list = []
    for tool in tools:
        tool_list.append(
            f"- {tool['name']}: {tool.get('description', '')}\n"
            f"  Parameters: {json.dumps(tool.get('parameters', {}), indent=2)}"
        )
    tools_text = "\n".join(tool_list)

    # Base template
    if format_style == "tags":
        return f"""You are a helpful assistant with access to tools.

Available tools:
{tools_text}

To call a tool, use this format:
<function_call>{{"name": "tool_name", "args": {{"param": "value"}}}}</function_call>

When you're done with the task, respond normally without calling any tools.
"""
    elif format_style == "hermes":
        return f"""You have access to the following tools:
{tools_text}

Use tools to complete tasks. Format tool calls as JSON."""

    else:
        raise ValueError(f"Unknown format_style: {format_style}")
```

```python
# examples/tau2bench/prompts.py (Task-specific)

def build_tau2_system_prompt(domain: str, tools: list[dict]) -> str:
    """Tau2Bench-specific system prompt."""
    base_prompt = build_tool_calling_system_prompt(tools, format_style="tags")

    # Add Tau2-specific instructions
    domain_instructions = {
        "mock": "You are managing tasks for users. Always confirm actions.",
        "airline": "You are a flight booking assistant. Be professional.",
        "retail": "You are a customer service agent. Be helpful and courteous.",
    }

    return f"""{base_prompt}

Domain: {domain}
{domain_instructions.get(domain, "")}

Remember to call done() when you've completed the task.
"""
```

### OpenEnv Integration for Tau2Bench

**Classification:** ⚠️ **Tau2Bench-specific** (`examples/tau2bench/tau2_env.py`)

**Reasoning:** Tau2-specific setup, task loading, tool registration

```python
# examples/tau2bench/tau2_env.py

class Tau2OpenEnv:
    """
    OpenEnv adapter for Tau2Bench tasks.
    Handles Tau2-specific setup and reward computation.
    """

    def __init__(self, base_url: str, domain: str, task_id: str):
        self.client = OpenEnv(base_url=base_url)
        self.domain = domain
        self.task_id = task_id
        self.task_data = self._load_task()
        self.tools = self._get_tools()

    def _load_task(self) -> dict:
        """Load Tau2Bench task data."""
        # Load from tau2-bench/data/tau2/domains/{domain}/tasks.json
        task_file = f"tau2-bench/data/tau2/domains/{self.domain}/tasks.json"
        with open(task_file) as f:
            tasks = json.load(f)
        return next(t for t in tasks if t["id"] == self.task_id)

    def _get_tools(self) -> list[dict]:
        """Get tool schemas for this domain."""
        # Domain-specific tools
        if self.domain == "mock":
            return [
                {
                    "name": "create_task",
                    "description": "Create a new task",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "user_id": {"type": "string"},
                            "title": {"type": "string"},
                            "description": {"type": "string"},
                            "deadline": {"type": "string"}
                        },
                        "required": ["user_id", "title"]
                    }
                },
                {
                    "name": "update_task",
                    "description": "Update task status",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "task_id": {"type": "string"},
                            "status": {"type": "string"}
                        },
                        "required": ["task_id", "status"]
                    }
                },
                {
                    "name": "done",
                    "description": "Signal task completion",
                    "parameters": {"type": "object", "properties": {}}
                }
            ]
        else:
            # Load from domain config
            raise NotImplementedError(f"Domain {self.domain} not implemented")

    def reset(self) -> EnvResult:
        """Reset environment for this task."""
        result = self.client.reset(
            task_id=self.task_id,
            domain=self.domain
        )
        return result

    def execute_tool(self, tool_call: dict) -> str:
        """Execute tool via OpenEnv."""
        result = self.client.step(tool_call)
        return result.observation.text

    def is_done(self) -> bool:
        """Check if episode is complete."""
        return self.client.state.get("done", False)

    def get_final_reward(self) -> float:
        """
        Compute Tau2Bench reward.
        Uses Tau2's evaluation criteria.
        """
        # Get episode history
        history = self.client.get_history()

        # Score using Tau2Bench evaluator
        from tau2.evaluator import evaluate_episode

        result = evaluate_episode(
            history=history,
            evaluation_criteria=self.task_data["evaluation_criteria"]
        )

        return result.final_reward  # 0.0 or 1.0
```

**Reward Computation:**
```python
# examples/tau2bench/tau2_utils.py

def compute_tau2_reward(
    task_data: dict,
    episode_history: list[dict],
) -> float:
    """
    Compute Tau2Bench reward from episode history.
    """
    from tau2.evaluator import Evaluator

    evaluator = Evaluator()

    # Evaluate based on criteria
    scores = evaluator.evaluate(
        history=episode_history,
        evaluation_criteria=task_data["evaluation_criteria"]
    )

    # Final reward = product of all scores
    final_reward = 1.0
    for score_type, score_value in scores.items():
        final_reward *= score_value

    return final_reward
```

## 6.4 Episode Structure for Multi-turn

```python
# forge/data/episode.py

@dataclass
class Episode:
    """Multi-turn episode with response masking."""
    episode_id: str
    pad_id: int

    # Token data (concatenated across all turns)
    token_ids: list[int]       # All tokens
    logprobs: list[float]      # Per-token logprobs
    response_mask: list[int]   # 1=train, 0=ignore (NEW!)

    # Metadata
    reward: float
    advantage: float | None = None
    num_turns: int = 1
    task_id: str = ""

    # Optional: for debugging
    messages: list[dict] | None = None

    def mask_tensor(self, max_len: int) -> torch.Tensor:
        """Get padded response mask tensor."""
        mask = self.response_mask + [0] * (max_len - len(self.response_mask))
        return torch.tensor(mask[:max_len], dtype=torch.float32)

    def masked_response_tensor(self, max_len: int) -> torch.Tensor:
        """Get response tokens with masking applied."""
        response = torch.tensor(self.token_ids, dtype=torch.long)
        mask = self.mask_tensor(max_len)
        # Apply mask (set masked tokens to pad_id)
        response = torch.where(
            mask.bool(),
            response,
            torch.tensor(self.pad_id, dtype=torch.long)
        )
        return response
```

## 6.5 Integration with Forge GRPO

**Update continuous_rollouts:**

```python
# examples/tau2bench/grpo/main.py

async def continuous_rollouts(
    policy: Generator,
    trainer: TitanTrainer,
    replay_buffer: ReplayBuffer,
    ref_model: ReferenceModel,
    dataloader: DataLoader,
    config: dict,
):
    """
    Updated rollout loop for multi-turn tool calling.
    """
    while True:
        # 1. Sample tasks
        tasks = await sample_tau2_tasks(dataloader, config.rollout.group_size)

        # 2. Run multi-turn episodes (parallel)
        episode_tasks = [
            play_task(
                task=task["ticket"],
                policy=policy,
                tokenizer=tokenizer,
                env=Tau2OpenEnv(
                    base_url=config.openenv.base_url,
                    domain=task["domain"],
                    task_id=task["id"]
                ),
                max_turns=config.rollout.max_turns_per_episode
            )
            for task in tasks
        ]

        episodes = await asyncio.gather(*episode_tasks)

        # 3. Get reference logprobs
        # Batch all episodes together
        all_token_ids = [ep.token_ids for ep in episodes]
        max_len = max(len(ids) for ids in all_token_ids)

        # Pad and stack
        input_ids = torch.stack([
            torch.tensor(ids + [pad_id] * (max_len - len(ids)))
            for ids in all_token_ids
        ])

        ref_logprobs = await ref_model.forward.route(
            input_ids=input_ids,
            return_logprobs=True
        )

        for i, episode in enumerate(episodes):
            episode.ref_logprobs = ref_logprobs[i, :len(episode.token_ids)]

        # 4. Compute advantages (group-relative)
        rewards = [ep.reward for ep in episodes]
        advantages = compute_advantages(rewards)

        for episode, advantage in zip(episodes, advantages):
            episode.advantage = advantage

        # 5. Add to replay buffer
        for episode in episodes:
            await replay_buffer.add.call_one(episode)


def compute_advantages(rewards: list[float]) -> list[float]:
    """Group-relative advantage computation (GRPO)."""
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards) + 1e-8
    advantages = [(r - mean_reward) / std_reward for r in rewards]
    return advantages
```

**Episode Creation Strategy:**

For Forge, **Strategy B (concatenated)** is recommended:
- All turns concatenated into one Episode
- Response mask distinguishes LLM output from tool results
- Gradient flows through entire trajectory
- Matches Forge's existing Episode structure better

## 6.6 GRPO Loss with Response Masking

**Reference existing Forge code:**
- `/home/felipemello/forge/src/forge/losses/reinforce_loss.py` already has `target_mask`
- `/home/felipemello/forge/apps/grpo/main.py` uses `compute_logprobs` and `F.cross_entropy`

**Add response_mask parameter:**

```python
# forge/losses/grpo_loss.py

def grpo_loss_with_masking(
    logits: torch.Tensor,           # [batch, seq_len, vocab_size]
    response: torch.Tensor,         # [batch, seq_len]
    response_mask: torch.Tensor,    # [batch, seq_len] - NEW!
    ref_logprobs: torch.Tensor,     # [batch, seq_len]
    advantages: torch.Tensor,       # [batch, seq_len]
    padding_mask: torch.Tensor,     # [batch, seq_len]
    beta: float = 0.1,
) -> torch.Tensor:
    """
    GRPO loss with response masking.
    Combines padding_mask (existing) with response_mask (new).
    """
    # Compute logprobs (memory-efficient using F.cross_entropy)
    logprobs = compute_logprobs(logits, response)

    # Combine masks: padding AND response masking
    combined_mask = padding_mask * response_mask

    # KL divergence
    kl = logprobs - ref_logprobs

    # Policy gradient loss
    pg_loss = -advantages * (logprobs - beta * kl)

    # Apply combined mask and reduce
    masked_loss = pg_loss * combined_mask
    loss = masked_loss.sum() / (combined_mask.sum() + 1e-8)

    return loss


def compute_logprobs(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Compute log probabilities using cross_entropy (memory efficient)."""
    # Shift for next-token prediction
    shift_logits = logits[..., :-1, :].contiguous()
    shift_targets = targets[..., 1:].contiguous()

    # Compute log probs
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_targets.view(-1),
        reduction='none'
    )

    return -loss.view(shift_logits.size(0), shift_logits.size(1))
```

**Key addition:** `response_mask` is the only new parameter. Loss computation is unchanged.

## 6.7 Enabling Async in Forge (Performance)

### Current Forge Async Mechanism

Forge uses Monarch actors for async communication (not vLLM's `async_engine` flag).

**How Forge handles async:**
- Generator is a distributed actor
- `await policy.generate.route()` sends async request to Generator actor
- vLLM engine runs on separate GPUs
- Response returned via actor system

**No configuration needed** - Forge handles this automatically!

### Making play_task Async

Already async in implementation above (`async def play_task()`).

### Running Multiple Tasks Concurrently

```python
# Pattern from 6.5 above
episode_tasks = [
    play_task(task, policy, tokenizer, env)
    for task in tasks
]
episodes = await asyncio.gather(*episode_tasks)
```

### Performance Best Practices

**1. Parallel Episode Processing:**

```python
# DON'T: Sequential reward computation
for episode in episodes:
    episode.reward = await compute_reward(episode)  # Slow!

# DO: Parallel reward computation
reward_tasks = [compute_reward(ep) for ep in episodes]
rewards = await asyncio.gather(*reward_tasks)
for episode, reward in zip(episodes, rewards):
    episode.reward = reward
```

**2. Batching Reference Model Calls:**

```python
# DON'T: One episode at a time
for episode in episodes:
    ref_logprobs = await ref_model.forward(episode.token_ids)

# DO: Batch all episodes
all_token_ids = [ep.token_ids for ep in episodes]
ref_logprobs_batch = await ref_model.forward(batch_tensor)
# Huge speedup!
```

**3. Pipeline Rollouts and Training:**

Forge already does this via replay buffer!
- Rollout threads: `continuous_rollouts()` (multiple parallel)
- Training thread: `continuous_training()`
- Decoupled via replay buffer
- No changes needed

---

**Next**: Part 7 shows how to evaluate your trained model on Tau2Bench.
