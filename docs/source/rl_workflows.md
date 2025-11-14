# RL Workflows

This guide shows you how to write RL algorithms with TorchForge, from simple episode generation to complex asynchronous training loops.

## Writing RL Algorithms

With TorchForge's foundations (Monarch, Services, TorchStore), here's what RL code looks like:

### Episode Generation

```python
async def generate_episode(dataloader, policy, reward, replay_buffer):
    # Sample a prompt
    prompt, target = await dataloader.sample.route()

    # Generate response (vLLM handles this efficiently)
    response = await policy.generate.route(prompt)

    # Score the response
    reward_value = await reward.evaluate_response.route(
        prompt=prompt,
        response=response.text,
        target=target
    )

    # Store for training
    await replay_buffer.add.route(
        Episode(
            prompt_ids=response.prompt_ids,
            response_ids=response.token_ids,
            reward=reward_value
        )
    )
```

Notice what's **not** here:
- No retry logic
- No resource allocation
- No synchronization code
- No infrastructure complexity

Just your algorithm.

### Asynchronous RL

Compose this into fully async, off-policy training:

```python
async def async_rl_loop(num_rollout_loops: int):
    # Multiple concurrent rollout generators
    rollout_tasks = [
        asyncio.create_task(continuous_rollouts())
        for _ in range(num_rollout_loops)
    ]

    # Continuous training
    training_task = asyncio.create_task(continuous_training())

    await asyncio.gather(*rollout_tasks, training_task)

async def continuous_rollouts():
    """Generate rollouts continuously using latest policy."""
    while True:
        await generate_episode(dataloader, policy, reward, replay_buffer)

async def continuous_training():
    """Train continuously on available experience."""
    training_step = 0
    while True:
        batch = await replay_buffer.sample.route(
            curr_policy_version=training_step
        )

        if batch is None:
            await asyncio.sleep(0.1)  # Wait for more experience
        else:
            loss = await trainer.train_step.route(batch)
            training_step += 1

            # Push updated weights (TorchStore handles this)
            await trainer.push_weights.route(training_step)
            # Broadcast to all policy replicas
            await policy.update_weights.fanout(training_step)
```

### Synchronous RL

The same `generate_episode()` function works for on-policy algorithms like PPO - just compose it differently:

```python
async def synchronous_rl(batch_size: int):
    """Synchronous on-policy RL: collect batch, then train."""
    version = 0

    while True:
        # Collect a full batch with current policy version
        for _ in range(batch_size):
            await generate_episode(dataloader, policy, reward, replay_buffer)

        # Sample the batch we just collected
        batch = await replay_buffer.sample.route(
            curr_policy_version=version,
            batch_size=batch_size
        )

        # Train on the complete batch
        loss = await trainer.train_step.route(batch)

        # Update weights in lockstep
        await trainer.push_weights.route(version + 1)
        await policy.update_weights.fanout(version + 1)
        version += 1
```

**The Power of Composition**: Write your rollout logic once, compose it into any paradigm - on-policy, off-policy, or anywhere in between.

## Extensible Environments

RL often requires interacting with environments beyond text generation - executing code, using tools, running simulations. TorchForge makes these first-class citizens through the same service abstraction.

### Code Execution

For RL on coding tasks (RLVR - Reinforcement Learning with Verifiable Rewards):

```python
# Lightweight CPU-only service for parallel execution
coder = SandboxedPythonCoder.options(
    procs=1,
    with_gpus=False,
    num_replicas=16
).as_service()

# In your RL code
async def generate_episode():
    prompt = await dataloader.sample.route()
    code = await policy.generate.route(prompt)

    # Execute safely in sandbox
    stdout, stderr = await coder.execute.route(code)
    reward = 1.0 if stderr == "" else 0.0  # Reward based on execution

    await replay_buffer.add.route(Episode(...))
```

### Tool Integration

Services make tools ephemeral - spawn them with your job, scale them independently, tear down when finished. The same coordination primitives work for any environment type.

```python
# Create a web browsing environment
browser = WebBrowsingEnv.options(
    procs=1,
    with_gpus=False,
    num_replicas=8
).as_service()

# Use it in your RL loop
async def generate_episode():
    task = await dataloader.sample.route()

    # Agent decides on actions
    action = await policy.generate.route(task)

    # Execute action in browser
    result = await browser.execute_action.route(action)

    # Evaluate outcome
    reward = await reward_model.evaluate.route(task, result)

    await replay_buffer.add.route(Episode(...))
```

This pattern extends naturally to **agentic workflows** - agents that interact with tools, query APIs, and navigate complex environments while learning from outcomes.

### Custom Environments

Build your own environment service:

```python
from monarch.actor import Actor, endpoint

class CustomEnv(Actor):
    def __init__(self):
        # Initialize your environment
        self.state = self.reset()

    @endpoint
    async def reset(self):
        """Reset environment to initial state."""
        return initial_state

    @endpoint
    async def step(self, action):
        """Execute action and return (observation, reward, done)."""
        # Your environment logic here
        return observation, reward, done

# Deploy as a service
env = CustomEnv.options(
    procs=1,
    num_replicas=16
).as_service()

# Use in training
obs = await env.reset.route()
while not done:
    action = await policy.act.route(obs)
    obs, reward, done = await env.step.route(action)
```

## Common Patterns

### Warmup Phase

Start training after collecting initial experience:

```python
async def warmup_then_train(warmup_episodes: int):
    # Collect initial experience
    for _ in range(warmup_episodes):
        await generate_episode(dataloader, policy, reward, replay_buffer)

    # Now start training
    await continuous_training()
```

### Evaluation Episodes

Interleave evaluation with training:

```python
async def train_with_eval(eval_interval: int):
    training_step = 0

    while True:
        # Training phase
        for _ in range(eval_interval):
            await generate_episode(dataloader, policy, reward, replay_buffer)
            batch = await replay_buffer.sample.route()
            await trainer.train_step.route(batch)
            training_step += 1

        # Evaluation phase
        eval_rewards = []
        for _ in range(100):
            episode = await generate_episode(
                eval_dataloader, policy, reward, None  # Don't store in buffer
            )
            eval_rewards.append(episode.reward)

        print(f"Step {training_step}: Eval reward = {np.mean(eval_rewards)}")
```

### Curriculum Learning

Gradually increase task difficulty:

```python
async def curriculum_training():
    difficulty = 0

    while difficulty < max_difficulty:
        # Train on current difficulty
        for _ in range(episodes_per_difficulty):
            prompt = await dataloader.sample.route(difficulty=difficulty)
            await generate_episode_with_prompt(prompt, policy, reward, replay_buffer)

        # Evaluate performance
        success_rate = await evaluate(policy, difficulty)

        # Move to next difficulty if threshold met
        if success_rate > 0.8:
            difficulty += 1
            print(f"Advanced to difficulty {difficulty}")
```

### Multi-Task Training

Train on multiple tasks simultaneously:

```python
async def multi_task_training(tasks: List[str]):
    # Create separate dataloaders for each task
    dataloaders = {task: create_dataloader(task) for task in tasks}

    while True:
        # Sample task uniformly (or with custom distribution)
        task = random.choice(tasks)
        dataloader = dataloaders[task]

        # Generate episode for this task
        await generate_episode(dataloader, policy, reward, replay_buffer)

        # Train uses mixed experience from all tasks
        batch = await replay_buffer.sample.route()
        await trainer.train_step.route(batch)
```

## Debugging Tips

### Start Small

Begin with a minimal setup to validate your logic:

```python
# Single GPU, single replica, synchronous
policy = PolicyActor.options(procs=1, with_gpus=True).as_service()
reward = RewardActor.options(procs=1, with_gpus=True).as_service()

# Run a few episodes
for _ in range(10):
    await generate_episode(dataloader, policy, reward, replay_buffer)
```

Once this works, scale up to multi-GPU and async training.

### Add Logging

Insert logging at key points:

```python
async def generate_episode(dataloader, policy, reward, replay_buffer):
    start = time.time()

    prompt, target = await dataloader.sample.route()
    print(f"Sampled prompt in {time.time() - start:.2f}s")

    gen_start = time.time()
    response = await policy.generate.route(prompt)
    print(f"Generated response in {time.time() - gen_start:.2f}s")

    reward_start = time.time()
    reward_value = await reward.evaluate_response.route(prompt, response.text, target)
    print(f"Computed reward in {time.time() - reward_start:.2f}s")

    await replay_buffer.add.route(Episode(...))
    print(f"Total episode time: {time.time() - start:.2f}s")
```

### Monitor Metrics

Track key metrics:

```python
from collections import deque

recent_rewards = deque(maxlen=100)
recent_kls = deque(maxlen=100)

async def continuous_training():
    training_step = 0

    while True:
        batch = await replay_buffer.sample.route()
        if batch:
            loss, kl = await trainer.train_step.route(batch)
            recent_kls.append(kl)

            if training_step % 100 == 0:
                print(f"Step {training_step}")
                print(f"  Avg reward: {np.mean(recent_rewards):.3f}")
                print(f"  Avg KL: {np.mean(recent_kls):.3f}")
                print(f"  Loss: {loss:.3f}")

            training_step += 1
```

## See Also

- {doc}`concepts` - Core philosophy and abstractions
- {doc}`architecture` - How Services and TorchStore enable these patterns
- {doc}`technology_stack` - Understanding the underlying components
- {doc}`usage` - Configuration and practical examples
- {doc}`tutorials` - Step-by-step guides
- {doc}`api` - Complete API reference
