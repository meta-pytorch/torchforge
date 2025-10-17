# Concepts

This guide introduces the fundamental principles and concepts behind TorchForge, helping you understand the philosophy that drives the system.

## The Core Philosophy

TorchForge is built on one principle: **researchers should write algorithms, not infrastructure**.

The traditional approach to distributed RL requires you to write complex coordination logic, retry mechanisms, resource management, and synchronization code. TorchForge abstracts all of this away, letting you express RL algorithms as naturally as pseudocode while powerful infrastructure handles the distributed complexity underneath.

## Key Abstractions

Understanding these core abstractions helps you use TorchForge effectively:

### Actor

A component that encapsulates a model along with its execution logic. Actors provide:
- **Isolation**: Independent resources and failure domains
- **Flexibility**: Different parallelism strategies per actor
- **Composability**: Combine actors to create complex pipelines

### Service

A managed group of actor replicas with built-in routing, load balancing, and fault tolerance. Services handle operational complexity so your RL code stays clean. Think of services as horizontally scaled actors with automatic load distribution.

### DTensor (Distributed Tensor)

A tensor sharded across multiple devices. TorchStore handles resharding DTensors between different topologies automatically, making distributed tensor operations transparent.

### Episode

A complete RL interaction sequence containing:
- **Prompt**: Input to the policy
- **Response**: Generated output
- **Reward**: Feedback signal
- **Metadata**: Additional context (timestamps, model versions, etc.)

Episodes flow through your system from generation to replay buffer to training.

### Replay Buffer

Stores episodes for training. Can be implemented with various strategies:
- **FIFO**: Simple queue for on-policy algorithms
- **Prioritized**: Importance sampling for off-policy learning
- **Reservoir**: Uniform sampling from history
- **Hybrid**: Mix multiple strategies

Integrates with TorchStore for efficient distributed storage.

## Design Principles

### Single-Controller Model

Traditional distributed training uses **SPMD (Single Program, Multiple Data)** - where multiple copies of the same script run across different machines, each with only a local view of the workflow. This works well for simple data-parallel training, but becomes notoriously difficult for complex RL workflows with:
- Asynchronous generation and training
- Multiple heterogeneous components (policy, reward model, reference model)
- Dynamic resource allocation
- Fault tolerance across components

TorchForge adopts **Monarch's single-controller model**: You write one Python script that orchestrates all distributed resources, making them feel almost local. The code looks and feels like a single-machine program, but can scale across thousands of GPUs.

### Composable Components

Write your core logic once, compose it into any paradigm:
- **Synchronous on-policy** (PPO, GRPO)
- **Asynchronous off-policy** (continuous rollouts + training)
- **Hybrid approaches** (batch collection with async training)

The same `generate_episode()` function works everywhere. Just change how you compose it.

### Ephemeral Infrastructure

Services are created with your job and torn down when finished:
- No standing deployments to maintain
- No infrastructure to provision ahead of time
- Want to try a new reward model? Change your Python code and rerun

This dramatically reduces operational overhead and enables rapid experimentation.

### Progressive Fault Tolerance

Write code as if nothing fails. When failures do occur:
- Monarch fails fast by default (like uncaught exceptions)
- Add fine-grained fault handling exactly where you need it
- Services automatically route around failed replicas
- Failed actors restart automatically

You choose your fault tolerance granularity based on your needs.

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

## Learn More

Dive deeper into specific topics:

```{toctree}
:maxdepth: 1

architecture
technology_stack
rl_workflows
```

**Related Documentation:**
- {doc}`getting_started` - Installation and first training run
- {doc}`api` - Complete API reference
