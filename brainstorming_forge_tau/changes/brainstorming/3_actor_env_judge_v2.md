# Multi-Environment Management for Forge + OpenEnv (CPU Only)

**Goal:** Enable >1 concurrent rollouts with tool execution using CPU-based OpenEnv environments.

**Key Principle:** Keep data and environment separate. Dataset provides tasks, environments provide tool execution.

---

## Problem Statement

From `3_5_ideal_state.md`, a single task needs N rollouts (group_size):

```python
# Need G rollouts for same task
for _ in range(group_size):  # e.g., G=8
    episode = await play_task(task_prompt, tool_schemas, env, max_turns)
```

**Issue:** If we have 1 environment and play tasks sequentially, we waste time. Environments can execute tools while LLM generates responses.

**Blackjack approach:** Creates env client per game, plays sequentially. Works but inefficient for tool calling.

---

## Proposed Solution: Environment Pool with Async Routing

Create a pool of N environment instances and route requests to available environments.

### Architecture

```
┌──────────────┐
│  DataLoader  │ ──→ tasks (prompt, task_type)
└──────────────┘
       │
       ↓
┌──────────────────────────────────────┐
│         Environment Pool             │
│  ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐ │
│  │Env 1│  │Env 2│  │Env 3│  │Env 4│ │
│  └─────┘  └─────┘  └─────┘  └─────┘ │
│    ↓         ↓         ↓         ↓   │
│ [free]   [busy]   [free]   [busy]   │
└──────────────────────────────────────┘
       │
       ↓
   Tool execution
```

**Core concept:** Maintain a queue of available environments. When a rollout needs tools, acquire an env from the pool, use it, then release it back.

---

## Implementation

### 1. Environment Pool Manager

```python
import asyncio
from typing import Dict, List
from openenv.core.http_env_client import HTTPEnvClient

class EnvPool:
    """Pool of OpenEnv instances for concurrent tool execution."""

    def __init__(
        self,
        env_type: str,  # e.g., "coding", "websearch"
        docker_image: str,
        pool_size: int = 4,
    ):
        self.env_type = env_type
        self.docker_image = docker_image
        self.pool_size = pool_size

        # Pool of environment clients
        self.envs: List[HTTPEnvClient] = []
        self.available = asyncio.Queue()

    async def initialize(self):
        """Create pool of environment instances."""
        # Start environment servers (separate Docker containers)
        for i in range(self.pool_size):
            port = 8000 + i
            env = await self._create_env(port)
            self.envs.append(env)
            await self.available.put(env)

    async def _create_env(self, port: int) -> HTTPEnvClient:
        """Create single environment instance."""
        # OpenEnv pattern: from_docker_image starts container + returns client
        env = HTTPEnvClient.from_docker_image(
            self.docker_image,
            ports={port: 8000},  # Map host:container ports
            name=f"{self.env_type}_env_{port}"
        )
        return env

    async def acquire(self) -> HTTPEnvClient:
        """Get available environment from pool (blocks if all busy)."""
        return await self.available.get()

    async def release(self, env: HTTPEnvClient):
        """Return environment to pool."""
        await self.available.put(env)

    async def shutdown(self):
        """Cleanup all environments."""
        for env in self.envs:
            env.close()
```

**Key points:**
- Each environment = separate Docker container on different port
- `acquire()` blocks if all envs busy (backpressure)
- Simple queue-based routing

---

### 2. Modified play_task() with Pool

```python
async def play_task(
    policy: Generator,
    task_prompt: str,
    env_pool: EnvPool,  # Changed from single env
    max_turns: int = 10
) -> Episode:
    """Play one task using environment from pool."""

    # Acquire environment from pool
    env = await env_pool.acquire()

    try:
        # Reset environment to get tools
        result = env.reset()
        tool_schemas = result.observation.tools

        messages = [{"role": "user", "content": task_prompt}]
        all_tokens = []
        all_logprobs = []
        response_mask = []

        done = False
        turn = 0

        while not done and turn < max_turns:
            # 1. Generate response
            prompt = tokenizer.apply_chat_template(
                messages,
                tools=tool_schemas,
                add_generation_prompt=True,
                tokenize=False
            )
            response = await policy.generate.route(prompt, n=1)

            # 2. Parse tool calls
            tool_calls = parse_tool_calls(response.text)

            if tool_calls:
                # Add assistant message
                messages.append({
                    "role": "assistant",
                    "content": response.text,
                    "tool_calls": tool_calls
                })

                # Collect LLM tokens
                all_tokens.extend(response.token_ids)
                all_logprobs.extend(response.logprobs)
                response_mask.extend([1] * len(response.token_ids))

                # 3. Execute tools with acquired env
                tool_results = []
                for tc in tool_calls:
                    result = env.step(ToolCallAction(
                        name=tc["name"],
                        args=tc["args"]
                    ))
                    tool_results.append(result)

                # Add tool results to conversation
                for tr in tool_results:
                    tool_content = tr.observation.content
                    tool_tokens = tokenizer.encode(tool_content, add_special_tokens=False)
                    tool_tokens = tool_tokens[:256]  # Truncate

                    messages.append({
                        "role": "tool",
                        "content": tokenizer.decode(tool_tokens)
                    })

                    # Collect tool tokens (don't train on these)
                    all_tokens.extend(tool_tokens)
                    all_logprobs.extend([0.0] * len(tool_tokens))
                    response_mask.extend([0] * len(tool_tokens))

                done = tool_results[-1].done if tool_results else False
            else:
                # Final answer
                messages.append({"role": "assistant", "content": response.text})
                all_tokens.extend(response.token_ids)
                all_logprobs.extend(response.logprobs)
                response_mask.extend([1] * len(response.token_ids))
                done = True

            turn += 1

        # Get final reward
        final_reward = env.get_reward() if hasattr(env, 'get_reward') else 0.0

        # Create episode
        completion = Completion(
            prompt_ids=None,
            token_ids=torch.tensor(all_tokens),
            logprobs=torch.tensor(all_logprobs),
            text=tokenizer.decode(all_tokens),
            generator_version=0
        )

        episode = Episode(
            episode_id=str(uuid.uuid4()),
            pad_id=tokenizer.pad_token_id,
            request_len=0,
            response_len=len(all_tokens),
            target=None,
            completion=completion,
            response_mask=torch.tensor(response_mask),
            ref_logprobs=None,
            reward=final_reward,
            advantage=None,
            metadata={"num_turns": turn, "truncated": turn >= max_turns}
        )

        return episode

    finally:
        # Always release environment back to pool
        await env_pool.release(env)
```

**Key changes:**
- Takes `env_pool` instead of single `env`
- Acquires env at start, releases at end (in finally block)
- Environment lifecycle managed by pool, not play_task

---

### 3. Rollout Loop with Pool

```python
async def continuous_rollouts(
    policy: Generator,
    dataloader: DataLoader,
    env_pools: Dict[str, EnvPool],  # Map task_type -> pool
    replay_buffer: ReplayBuffer,
    group_size: int = 8
):
    """Continuous rollout loop with environment pools."""

    while True:
        # Sample task from dataloader
        task = await dataloader.sample.call_one()

        # Get pool for this task type
        env_pool = env_pools[task.task_type]

        # Play G rollouts concurrently using pool
        rollout_tasks = [
            play_task(
                policy=policy,
                task_prompt=task.prompt,
                env_pool=env_pool,
                max_turns=10
            )
            for _ in range(group_size)
        ]

        # Wait for all rollouts to complete
        episodes = await asyncio.gather(*rollout_tasks)

        # Add to replay buffer
        for episode in episodes:
            await replay_buffer.add.call_one(episode)
```

**Key points:**
- Uses `asyncio.gather()` to run rollouts concurrently
- Pool handles contention - if all envs busy, rollouts wait
- Each rollout acquires/releases env independently

---

### 4. Setup and Configuration

```python
# Main setup
async def main():
    # 1. Create services
    policy = Generator(...)
    trainer = TitanTrainer(...)
    replay_buffer = ReplayBuffer(...)
    dataloader = DataLoader(Tau2BenchDataset(...))

    # 2. Create environment pools
    env_pools = {}

    # Coding environment pool (4 instances)
    coding_pool = EnvPool(
        env_type="coding",
        docker_image="tau2bench/coding:latest",
        pool_size=4
    )
    await coding_pool.initialize()
    env_pools["coding"] = coding_pool

    # WebSearch environment pool (4 instances)
    websearch_pool = EnvPool(
        env_type="websearch",
        docker_image="tau2bench/websearch:latest",
        pool_size=4
    )
    await websearch_pool.initialize()
    env_pools["websearch"] = websearch_pool

    # 3. Start rollout and training loops
    try:
        rollout_task = asyncio.create_task(
            continuous_rollouts(policy, dataloader, env_pools, replay_buffer, group_size=8)
        )
        training_task = asyncio.create_task(
            continuous_training(trainer, replay_buffer, policy)
        )

        await asyncio.gather(rollout_task, training_task)
    finally:
        # Cleanup
        for pool in env_pools.values():
            await pool.shutdown()
```

---

## Performance Analysis

### Pool Size vs Concurrency

| Pool Size | Group Size | Behavior |
|-----------|------------|----------|
| 1 | 8 | Sequential (like blackjack) - slow |
| 4 | 8 | 4 concurrent, 4 wait - better |
| 8 | 8 | All concurrent - optimal |
| 16 | 8 | Wastes resources (idle envs) |

**Recommendation:** Pool size ≈ group_size for optimal throughput.

### Bottleneck Analysis

Where does time go in a rollout?

```
┌─────────────────┐
│ LLM generation  │  ~200-500ms per turn
└─────────────────┘
         ↓
┌─────────────────┐
│ Tool execution  │  ~50-200ms per tool call
└─────────────────┘
```

**Key insight:** LLM generation and tool execution can overlap across different rollouts!

Example timeline with pool_size=4, group_size=8:

```
Time →
Env1: [R1-tool] ─────── [R5-tool] ───────
Env2: ────── [R2-tool] ─────── [R6-tool]
Env3: [R3-tool] ─────── [R7-tool] ───────
Env4: ────── [R4-tool] ─────── [R8-tool]

R1-R4 execute concurrently, R5-R8 wait then execute
```

vs Sequential (pool_size=1):
```
Env1: [R1] [R2] [R3] [R4] [R5] [R6] [R7] [R8]
```

**Speedup:** ~3-4x with pool_size=4.

---

## Open Questions

1. **Docker startup cost:** How long does `from_docker_image()` take? If slow, pre-warm pool at startup. If fast, create on-demand.

2. **Environment cleanup:** Should envs be reused across tasks or reset? OpenEnv allows `env.reset()` to clear state.

3. **Pool size tuning:** How to determine optimal pool size? Depends on tool execution time vs generation time.

4. **Mixed task types:** If batch has websearch + coding tasks, need both pools. Does this waste resources?

5. **Error handling:** If env crashes, should pool recreate it or fail? Need retry logic.

---

## Comparison to Actor-Based Approach

**Environment Pool (this doc):**
- ✅ Simple implementation
- ✅ Works with existing OpenEnv
- ✅ CPU-only, no GPU complexity
- ❌ Limited to single machine (Docker on localhost)
- ❌ Manual pool management

**Actor-Based (future):**
- ✅ Distributed across machines
- ✅ GPU support for environments
- ✅ Fault tolerance (Forge actors)
- ❌ More complex
- ❌ Requires Forge actor infrastructure

---

## Next Steps

1. **Implement EnvPool class** in `src/forge/envs/pool.py`
2. **Test with single task type** (e.g., coding only)
3. **Measure speedup** vs sequential (blackjack approach)
4. **Tune pool size** based on profiling
5. **Add error handling** for env crashes

Once CPU pooling works well, consider scaling to actors for distributed execution.


# Actor-Based Environment Management: Do We Need Sticky Sessions?

**Context:** We want multiple environments for concurrent rollouts. Should we use manual pooling (doc 9) or Forge actors?

---

## Understanding State and Sessions

From `2_Forge_Internals.md`, sticky sessions solve this problem:

```python
# WITHOUT SESSIONS: Each .route() goes to different replica
await counter_service.increment.route()  # → replica 2
await counter_service.increment.route()  # → replica 1
await counter_service.increment.route()  # → replica 3
# Result: Inconsistent state across replicas

# WITH SESSIONS: All calls go to same replica
async with counter_service.session():
    await counter_service.reset.route()      # → replica 2
    await counter_service.increment.route()  # → replica 2
    await counter_service.increment.route()  # → replica 2
# Result: Consistent state within session
```

**When needed:** Multi-turn conversations (KV cache), stateful computations.

---

## Environment State Analysis

### Blackjack: Per-Game State

From `grpo_blackjack/grpo_utils.py:384-492`:

```python
async def play_game(...):
    env = OpenSpielEnv(base_url=server_url)  # Fresh client

    try:
        result = env.reset()  # Initialize game state
        done = False
        step_num = 0

        while not done and step_num < 10:
            # Generate action
            responses = await policy.generate.route(prompt)
            action_id = parse_action(responses[0].text, obs.legal_actions)

            # Execute in same environment
            result = env.step(OpenSpielAction(action_id=action_id, game_name="blackjack"))
            done = result.done
            step_num += 1

        final_reward = result.reward  # Game outcome
        return all_step_results
    finally:
        env.close()  # Cleanup
```

**State characteristics:**
- **Stateful within game:** Cards, player hand, dealer hand, score
- **Stateless between games:** Each `play_game()` creates fresh env
- **State duration:** Single game (3-10 steps)

### Coding Env: Per-Task State

Similar pattern for code execution:

```python
async def play_task(...):
    env = CodingEnv(...)  # Fresh environment

    try:
        result = env.reset()  # Initialize execution context

        while not done and turn < max_turns:
            # Generate code/action
            response = await policy.generate.route(prompt)
            tool_calls = parse_tool_calls(response.text)

            # Execute in same environment
            for tc in tool_calls:
                result = env.step(ToolCallAction(name=tc["name"], args=tc["args"]))

        final_reward = env.get_reward()
        return episode
    finally:
        env.close()
```

**State characteristics:**
- **Stateful within task:** Variables, file system, execution history
- **Stateless between tasks:** Each task gets fresh env
- **State duration:** Single task (1-15 turns)

---

## Question: Do We Need Sticky Sessions?

**Short answer:** No, if we acquire env at start of task and release at end.

**Why?**
1. Each task uses ONE environment throughout (no load balancing mid-task)
2. We're not doing `.route()` to envs during the task
3. The pool/actor handles routing at task level, not step level

**Comparison:**

| Pattern | Load Balancing Level | Needs Sessions? |
|---------|----------------------|-----------------|
| **Policy service** | Per generation call | Yes (for multi-turn with KV cache) |
| **Environment pool** | Per task | No (task acquires one env) |
| **Environment service** | Per step (if we .route()) | Yes (to maintain task state) |

---

## Three Approaches to Environment Management

### Approach 1: Manual Pool (Doc 9) - Simplest

```python
class EnvPool:
    def __init__(self, docker_image: str, pool_size: int):
        self.available = asyncio.Queue()

    async def acquire(self) -> HTTPEnvClient:
        return await self.available.get()  # Blocks if all busy

    async def release(self, env: HTTPEnvClient):
        await self.available.put(env)

# Usage
async def play_task(env_pool: EnvPool):
    env = await env_pool.acquire()  # Get one env
    try:
        # Use env for entire task
        while not done:
            result = env.step(action)
    finally:
        await env_pool.release(env)  # Return to pool
```

**Pros:**
- ✅ Simple, explicit control
- ✅ No sticky sessions needed
- ✅ Works with existing OpenEnv

**Cons:**
- ❌ Manual pool management
- ❌ No fault tolerance
- ❌ Not distributed

---

### Approach 2: Environment as Actor (No Sessions) - Recommended

Each environment = separate actor. Acquire at task start, use for full task.

```python
from forge.controller import ForgeActor
from monarch.actor import endpoint

@dataclass
class CodingEnvActor(ForgeActor):
    """Single coding environment as Forge actor."""

    docker_image: str = "tau2bench/coding:latest"

    def __post_init__(self):
        from openenv.envs.coding_env import CodingEnv
        self.env = CodingEnv.from_docker_image(self.docker_image)

    @endpoint(async_mode=True)
    async def reset(self):
        """Reset environment for new task."""
        result = self.env.reset()
        return result

    @endpoint(async_mode=True)
    async def step(self, action):
        """Execute action in environment."""
        result = self.env.step(action)
        return result

    @endpoint
    async def get_reward(self) -> float:
        """Get final reward for task."""
        return self.env.get_reward()

    @endpoint
    def close(self):
        """Cleanup environment."""
        self.env.close()


# Create pool of environment actors
env_actors = await asyncio.gather(*[
    CodingEnvActor.options(procs=1).as_actor(
        docker_image="tau2bench/coding:latest"
    )
    for _ in range(pool_size)
])

# Create simple pool manager
class ActorPool:
    def __init__(self, actors: list):
        self.available = asyncio.Queue()
        for actor in actors:
            self.available.put_nowait(actor)

    async def acquire(self):
        return await self.available.get()

    async def release(self, actor):
        await self.available.put(actor)

env_pool = ActorPool(env_actors)

# Usage in play_task
async def play_task(env_pool: ActorPool):
    env_actor = await env_pool.acquire()  # Get one actor

    try:
        # Reset for new task
        await env_actor.reset.call_one()

        # Use actor for entire task
        while not done:
            result = await env_actor.step.call_one(action)

        final_reward = await env_actor.get_reward.call_one()
        return episode
    finally:
        await env_pool.release(env_actor)  # Return to pool
```

**Pros:**
- ✅ Clean Forge integration
- ✅ Actor fault tolerance (automatic restart)
- ✅ No sessions needed (acquire/release pattern)
- ✅ Explicit actor per task

**Cons:**
- ❌ Still manual pool management (ActorPool class)
- ❌ Not using service abstraction
- ❌ More boilerplate than both alternatives

**When to use:** Don't use this - Service + sessions is better (automatic pool management).

---

### Approach 3: Environment as Service WITH Sessions - Most Complex

Each task creates a session to stick to one environment replica.

```python
# Create environment service
env_service = await CodingEnvActor.options(
    procs=1,
    num_replicas=4  # 4 environment replicas
).as_service(docker_image="tau2bench/coding:latest")

# Usage in play_task - WITH SESSION
async def play_task(env_service):
    # Session ensures all calls go to same replica
    async with env_service.session():
        await env_service.reset.route()  # → replica 2

        while not done:
            # All steps hit same replica = maintains state
            result = await env_service.step.route(action)  # → replica 2

        final_reward = await env_service.get_reward.route()  # → replica 2
    # Session ends, replica available for other tasks
```

**Pros:**
- ✅ Uses service abstraction
- ✅ Automatic load balancing across replicas
- ✅ Fault tolerance

**Cons:**
- ⚠️ Must use `async with service.session()` (but this is simpler than manual pool!)
- ⚠️ Slightly more overhead than manual pool

**When to use:** Preferred over Actor Pool (Approach 2) because service handles replica management automatically.

---

## Recommendation: Manual Pool vs Service + Sessions

**Key insight:** Service + sticky sessions = automatic pool management! No need for manual ActorPool.

### When to use Manual Pool (Approach 1):
- ✅ Simplest implementation (no actors)
- ✅ Good for CPU-only, single machine
- ✅ Minimal overhead
- ❌ No fault tolerance
- ❌ No distributed execution

### When to use Service + Sessions (Approach 3):
- ✅ Fault tolerance (automatic actor restart)
- ✅ Automatic load balancing (service picks replica)
- ✅ Session handles routing (no manual pool!)
- ✅ Distributed execution ready
- ❌ More setup overhead
- ⚠️ Need to remember `async with service.session()`

**Approach 2 (Actor Pool) is unnecessary** - it's manual pool management with actors, which is more complex than both alternatives.

---

## Sticky Sessions: When Actually Needed?

**Needed:**
1. **Multi-turn LLM with KV cache:**
   ```python
   async with policy.session():
       r1 = await policy.generate.route(turn1)  # Cache hit
       r2 = await policy.generate.route(turn1 + r1)  # Cache hit
   ```

2. **Stateful computation across multiple service calls:**
   ```python
   async with counter_service.session():
       await counter_service.increment.route()
       await counter_service.increment.route()
   ```

**NOT needed:**
1. **Single environment for entire task:**
   ```python
   env = await env_pool.acquire()  # Get one
   # Use env throughout task
   await env_pool.release(env)  # Return
   ```

2. **Fresh state per call:**
   ```python
   # Each call independent
   reward = await reward_actor.evaluate_response.route(...)
   ```

---

## State Analysis: Blackjack vs Coding

| Aspect | Blackjack | Coding Env |
|--------|-----------|------------|
| **State holder** | OpenSpiel server | Docker container |
| **State content** | Cards, scores, history | Variables, files, stdout |
| **State duration** | 3-10 steps (one game) | 1-15 turns (one task) |
| **State between tasks** | None (fresh game) | None (fresh container) |
| **Needs sessions?** | No | No |
| **Why not?** | Acquire env once per game | Acquire env once per task |

**Key insight:** Both are stateful WITHIN a task but stateless BETWEEN tasks. Since we acquire environment at task start and hold it until task end, we don't need sessions.

---

## Implementation Recommendation

**For now (CPU only, simple):**
Use manual pool from Doc 9. It's clear, explicit, and sufficient.

**Future (GPU, distributed):**
Convert to actor pool when you need:
- GPU environments (Forge actors can claim GPUs)
- Fault tolerance
- Remote execution

**Don't use service + sessions for environments** unless you have a specific need for automatic load balancing at the step level (unlikely).

---

## Code Example: Manual Pool vs Service + Sessions

```python
# OPTION A: Manual pool (simplest, no actors)
class EnvPool:
    def __init__(self, docker_image: str, pool_size: int):
        self.available = asyncio.Queue()
        for i in range(pool_size):
            env = HTTPEnvClient.from_docker_image(docker_image, port=8000+i)
            self.available.put_nowait(env)

env_pool = EnvPool("tau2bench/coding:latest", pool_size=4)

async def play_task():
    env = await env_pool.acquire()  # Get env from queue
    await env.reset()
    await env.step(action)
    await env_pool.release(env)  # Return to queue

# OPTION B: Service with sessions (automatic pool management)
env_service = await CodingEnvActor.options(
    procs=1,
    num_replicas=4  # Service manages 4 replicas
).as_service(docker_image="tau2bench/coding:latest")

async def play_task():
    # Session automatically picks a replica and sticks to it
    async with env_service.session():
        await env_service.reset.route()  # → replica N
        await env_service.step.route(action)  # → same replica N
    # Session ends, replica automatically becomes available
```

**Comparison:**
- **Option A:** Manual queue management, explicit acquire/release
- **Option B:** Service manages replicas, session handles routing - no manual pool needed!

**For your use case (OpenSpiel with state):** Either works, but Option B is cleaner once you're using actors.

---

## Summary

| Question | Answer |
|----------|--------|
| **Do environments have state?** | Yes, within a task (game/episode) - OpenSpiel holds cards, score, etc. |
| **Do we need sticky sessions?** | Only if using service (Approach 3) - session ensures same replica |
| **Best approach for CPU-only?** | Manual pool (Approach 1) - simplest |
| **Best approach with actors?** | Service + sessions (Approach 3) - automatic pool management |
| **What about Actor Pool (Approach 2)?** | Skip it - unnecessary manual work |

**Key insight from your question:** Yes, sticky sessions ensure same env/replica, eliminating need for manual ActorPool!

```python
# Service + session replaces manual pool:
async with env_service.session():  # Service picks replica, session sticks to it
    await env_service.reset.route()   # → replica 2 (has state)
    await env_service.step.route(a1)  # → replica 2 (state preserved)
    await env_service.step.route(a2)  # → replica 2 (state preserved)
# Session ends, replica 2 becomes available for other tasks
```

**Next step:** Start with manual pool (simplest). Use service + sessions if you need actor benefits.
