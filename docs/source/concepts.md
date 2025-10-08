# Concepts

This guide covers the fundamental concepts and architecture behind TorchForge, helping you understand how the system works and how to effectively use its components.

## The Core Philosophy

TorchForge is built on one principle: **researchers should write algorithms, not infrastructure**.

The traditional approach to distributed RL requires you to write complex coordination logic, retry mechanisms, resource management, and synchronization code. TorchForge abstracts all of this away, letting you express RL algorithms as naturally as pseudocode while powerful infrastructure handles the distributed complexity underneath.

## The Foundation: Monarch

At TorchForge's core is **Monarch**, a PyTorch-native distributed programming framework that brings single-controller orchestration to entire GPU clusters.

### Single-Controller vs SPMD

Traditional distributed training uses **SPMD (Single Program, Multiple Data)** - where multiple copies of the same script run across different machines, each with only a local view of the workflow. This works well for simple data-parallel training, but becomes notoriously difficult for complex RL workflows with:
- Asynchronous generation and training
- Multiple heterogeneous components (policy, reward model, reference model)
- Dynamic resource allocation
- Fault tolerance across components

**Monarch's single-controller model** changes this entirely. You write one Python script that orchestrates all distributed resources, making them feel almost local. The code looks and feels like a single-machine program, but can scale across thousands of GPUs.

### Actor Meshes

Monarch organizes resources into multidimensional arrays called **meshes**:

**Process Mesh**
: An array of processes spread across many hosts, typically one process per GPU

**Actor Mesh**
: An array of actors, each running inside a separate process

Like array programming in NumPy or PyTorch, meshes make it simple to dispatch operations efficiently across large systems. You can slice meshes, broadcast operations, and operate on entire meshes with simple APIs.

```python
from monarch.actor import Actor, this_host

# Create a process mesh with 8 GPUs
procs = this_host().spawn_procs({"gpus": 8})

# Define an actor
class PolicyActor(Actor):
    @endpoint
    def generate(self, prompt):
        return self.model.generate(prompt)

# Spawn actors across the mesh
actors = procs.spawn("policy", PolicyActor)

# Call methods on the entire mesh
results = actors.generate.call_all("Hello world")
```

### Fault Tolerance

Monarch provides **progressive fault handling** - you write your code as if nothing fails. When something does fail, Monarch fails fast by default, stopping the whole program like an uncaught exception.

But you can progressively add fine-grained fault handling exactly where you need it:

```python
try:
    result = await policy.generate.route(prompt)
except ActorFailure:
    # Handle failure - maybe retry with different replica
    result = await policy.generate.route(prompt)
```

For long-running RL training, this is crucial. Hardware failures are common at scale - in Meta's Llama 3 training, there were 419 interruptions across 54 days on a 16K GPU job (roughly one failure every 3 hours).

### RDMA and Data Plane

Monarch separates the **control plane** (messaging) from the **data plane** (bulk data transfers). This enables direct GPU-to-GPU memory transfers across your cluster using RDMA (Remote Direct Memory Access).

Control commands go through one optimized path, while large data transfers (like model weights) go through another path optimized for bandwidth.

## TorchForge Architecture

With Monarch as the foundation, TorchForge builds higher-level abstractions specifically for RL workflows.

### Services: RL-Friendly Actor Abstraction

**Services** wrap Monarch's ActorMesh with patterns common in RL. A service is a managed group of actor replicas with built-in load balancing, fault tolerance, and routing primitives.

```python
# Create a policy service with 16 replicas, each using 8 GPUs
policy = PolicyActor.options(
    procs=8,
    with_gpus=True,
    num_replicas=16
).as_service()

# Create a lightweight coding environment service
coder = SandboxedCoder.options(
    procs=1,
    with_gpus=False,
    num_replicas=16
).as_service()
```

**Service Adverbs** provide intuitive operations:

**route()**
: Load-balanced request to one replica
```python
response = await policy.generate.route(prompt)
```

**fanout()**
: Broadcast to ALL replicas in parallel
```python
await policy.update_weights.fanout(version)
```

**session()**
: Sticky sessions for stateful operations (maintains KV cache consistency)
```python
async with policy.session():
    response1 = await policy.generate.route(prompt1)
    response2 = await policy.generate.route(prompt2)  # Same replica
```

### Why Services Matter for RL

Services solve critical infrastructure challenges:

**Heterogeneous Scaling**
: Different components need different resources. Your policy might need 16 replicas × 8 GPUs for high-throughput vLLM inference. Your reward model might need 4 replicas × 4 GPUs. Your coding environment might need 16 lightweight CPU-only replicas. Services let each component scale independently.

**Load Balancing**
: In async RL, multiple `continuous_rollouts()` tasks run concurrently. Services automatically distribute these rollouts across available replicas - no manual worker pool management.

**Fault Tolerance**
: If a replica fails during a rollout, services detect it, mark it unhealthy, and route subsequent requests to healthy replicas. The failed replica gets restarted automatically. Your RL code never sees the failure.

**Ephemeral Infrastructure**
: Services are created with your job and torn down when finished. Want to try a new reward model? Change your Python code. No standing deployments to maintain, no infrastructure to provision ahead of time.

## TorchStore: Distributed Weight Storage

In async RL, every training step produces new policy weights that must propagate to all inference replicas. For a 70B parameter model across 16 replicas, this means moving hundreds of gigabytes of data. **TorchStore** makes this efficient.

### The Weight Synchronization Challenge

Traditionally, you have two options:
1. **Build complex p2p mappings** between training and inference sharding strategies (fast but extremely complex)
2. **Use network filesystem** like NFS (simple but slow, with high infrastructure cost)

TorchStore combines the **UX of central storage** with the **performance of in-memory p2p operations**.

### How TorchStore Works

TorchStore is a distributed, in-memory key-value store for PyTorch tensors, built on Monarch primitives:

```python
import torchstore as ts
from torch.distributed._tensor import distribute_tensor, Shard
from torch.distributed.device_mesh import init_device_mesh

# Training process: store sharded weights
async def store_weights():
    device_mesh = init_device_mesh("cuda", (4,))
    tensor = model.state_dict()['layer.weight']
    dtensor = distribute_tensor(tensor, device_mesh, [Shard(0)])

    # Each rank stores its shard
    await ts.put("policy_weights_v123", dtensor)

# Inference process: fetch with different sharding
async def load_weights():
    device_mesh = init_device_mesh("cuda", (2, 2))  # Different topology!
    tensor = torch.empty_like(model.state_dict()['layer.weight'])
    dtensor = distribute_tensor(tensor, device_mesh, [Shard(0)])

    # TorchStore handles resharding automatically
    await ts.get("policy_weights_v123", dtensor)
```

**Key Features:**

**Automatic Resharding**
: Handles complex weight transfer between different sharding strategies transparently

**DTensor Native**
: Works seamlessly with PyTorch's distributed tensors

**RDMA Transfers**
: Uses RDMA for high-bandwidth data movement without blocking GPUs

**Asynchronous Updates**
: Training and inference can read/write weights independently, enabling true async RL

**Flexible Storage**
: Store tensors co-located with trainers, on their own storage tier, sharded or replicated - change with minimal code modifications

### Why TorchStore Matters

Without TorchStore, weight synchronization becomes the bottleneck in async RL. Traditional approaches either:
- Require synchronous GPU-to-GPU transfers (blocking training)
- Use slow network filesystems (minutes per update)
- Demand complex manual resharding logic (error-prone, hard to maintain)

TorchStore solves all of these, keeping data distributed across the cluster until requested and moving it efficiently with RDMA.

## The RL Stack: Proven Components

TorchForge made a conscious decision not to reinvent the wheel. We integrate battle-tested components:

### vLLM: High-Throughput Inference

**vLLM** handles policy generation with:
- **PagedAttention**: Memory-efficient attention that reduces fragmentation
- **Continuous Batching**: Dynamic batching for maximum GPU utilization
- **Production Performance**: Proven at scale

In RL, policy generation is often the bottleneck. Autoregressive generation is costly, and blocking training on it kills throughput. vLLM enables fast, efficient inference that doesn't bottleneck your training loop.

**Integration**: TorchForge integrates directly with vLLM's engine. You can customize generation strategies, memory management, and inference logic as your research demands.

### TorchTitan: Production Training

**TorchTitan** brings Meta's production-grade training infrastructure:
- **FSDP**: Shard parameters, gradients, and optimizer states across GPUs
- **Pipeline Parallelism**: Split model layers across devices with efficient micro-batching
- **Tensor Parallelism**: Split individual tensors across devices for very large models
- **Proven at Scale**: Used to train Llama models on thousands of GPUs

Modern models are too large for single GPUs. TorchTitan provides the sophisticated sharding and parallelism strategies needed to train them efficiently.

**Integration**: TorchForge provides direct access to TorchTitan's training step logic and sharding strategies, enabling deep customization.

### Role of Integration

These integrations give you:
- **Direct component access**: Customize deeply when needed
- **Proven performance**: Battle-tested at massive scale
- **Flexible composition**: Mix and match with custom components

TorchForge's role is **coordination** - making these components work together seamlessly so you can express your RL algorithm naturally.

## Writing RL Algorithms

With these foundations, here's what RL code looks like in TorchForge:

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

This pattern extends naturally to **agentic workflows** - agents that interact with tools, query APIs, and navigate complex environments while learning from outcomes.

## Resource Management

Effective resource management is crucial for training large models.

### Memory Optimization

**Gradient Checkpointing**
: Trade computation for memory by recomputing activations during backward pass

**Mixed Precision**
: Use FP16/BF16 for reduced memory footprint while maintaining quality

**Activation Offloading**
: Move activations to CPU when not needed on GPU

**Parameter Sharding**
: Distribute model parameters across devices using FSDP

### Compute Optimization

**Asynchronous Execution**
: Overlap communication with computation to hide latency

**Batch Size Tuning**
: Balance memory usage and throughput for optimal training speed

**Dynamic Batching**
: Group requests efficiently for inference (vLLM does this)

**Kernel Fusion**
: Combine operations to reduce memory bandwidth (torch.compile helps)

## Distributed Training Strategies

TorchForge leverages multiple parallelism strategies through TorchTitan:

### Data Parallelism

Each GPU processes different batches using the same model replica. Gradients are synchronized via all-reduce operations.

### FSDP (Fully Sharded Data Parallel)

**Sharding** splits model parameters, gradients, and optimizer states across multiple devices, dramatically reducing memory per GPU. FSDP is the strategy that enables training models larger than single-GPU memory.

### Tensor Parallelism

Split individual tensors (like large weight matrices) across devices. Each device computes a portion of the operation.

### Pipeline Parallelism

Partition model layers across devices, pipeline micro-batches through the stages for efficient utilization.

## Key Abstractions

Understanding these core abstractions helps you use TorchForge effectively:

### Actor

A component that encapsulates a model along with its execution logic. Actors provide isolation (independent resources), flexibility (different parallelism strategies), and composability (combine to create complex pipelines).

### Service

A managed group of actor replicas with built-in routing, load balancing, and fault tolerance. Services handle operational complexity so your RL code stays clean.

### DTensor (Distributed Tensor)

A tensor sharded across multiple devices. TorchStore handles resharding DTensors between different topologies automatically.

### Episode

A complete RL interaction sequence - prompt, response, reward, and metadata. Episodes flow through your system from generation to training.

### Replay Buffer

Stores episodes for training. Can be implemented with various strategies (FIFO, prioritized, etc.) and integrates with TorchStore for efficient storage.

## Best Practices

### Model Selection

- Start with smaller models for prototyping
- Use pre-configured model setups when available
- Validate configurations before large-scale training

### Data Preparation

- Ensure balanced and diverse training data
- Implement proper train/validation splits
- Monitor data quality throughout training
- Verify token distributions match expectations

### Training Strategy

- Begin with SFT before attempting GRPO
- Use gradient accumulation for larger effective batch sizes
- Monitor KL divergence to prevent policy collapse
- Implement regular checkpointing for fault tolerance
- Apply warmup schedules for stable training

### Resource Optimization

- Profile memory usage to identify bottlenecks
- Tune batch sizes for your hardware configuration
- Consider mixed precision training to reduce memory
- Use appropriate parallelism strategies for your model size

### Debugging

- Start with single-GPU training to isolate issues
- Enable verbose logging for distributed runs
- Use profiling tools to identify bottlenecks
- Validate data pipelines before full training
- Monitor loss curves and generation quality

## Validation

TorchForge has been validated in real-world deployments:

- **Stanford Collaboration**: Integration with the Weaver weak verifier project, training models that hill-climb on challenging reasoning benchmarks (MATH, GPQA)
- **CoreWeave**: Large-scale training runs on 512 H100 GPU clusters with smooth, efficient performance
- **Scale**: Tested across hundreds of GPUs with continuous rollouts and asynchronous training

## See Also

- {doc}`getting_started` - Installation, setup, and first training run
- {doc}`usage` - Practical usage examples and configuration patterns
- {doc}`tutorials` - Step-by-step guides
- {doc}`api` - Complete API reference
- {doc}`faq` - Common questions and troubleshooting
