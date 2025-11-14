# Architecture

This guide provides a deep dive into TorchForge's architecture, explaining how Monarch, Services, and TorchStore work together to enable distributed RL.

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

# Create a process mesh with 8 processes (one per GPU)
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

## Services: RL-Friendly Actor Abstraction

**Services** wrap Monarch's ActorMesh with patterns common in RL. A service is a managed group of actor replicas with built-in load balancing, fault tolerance, and routing primitives.

```python
# Create a policy service with 16 replicas, each using 8 processes
policy = PolicyActor.options(
    procs=8,
    with_gpus=True,
    num_replicas=16
).as_service()
```

### Service Adverbs

Services provide intuitive operations called "adverbs":

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
: Different components need different resources. Your policy might need 16 replicas × 8 processes for high-throughput vLLM inference. Your reward model might need 4 replicas × 4 processes. Your coding environment might need 16 lightweight CPU-only replicas. Services let each component scale independently.

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

Weight synchronization becomes a bottleneck in async RL. Traditional approaches either:
- Require synchronous GPU-to-GPU transfers (blocking training)
- Use slow network filesystems (minutes per update)
- Demand complex manual resharding logic (error-prone, hard to maintain)

TorchStore solves all of these, keeping data distributed across the cluster until requested and moving it efficiently with RDMA.

## Distributed Training Strategies

TorchForge leverages multiple parallelism strategies through TorchTitan. [See their docs for more](https://github.com/pytorch/torchtitan).

## See Also

- {doc}`concepts` - Core philosophy and key abstractions
- {doc}`technology_stack` - Understanding the dependency stack
- {doc}`rl_workflows` - Writing RL algorithms with these components
- {doc}`getting_started` - Installation and setup
