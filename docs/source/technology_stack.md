# Technology Stack

TorchForge is built on a carefully curated stack of battle-tested components, each solving specific challenges in distributed RL. Understanding this stack helps you troubleshoot issues, optimize performance, and customize your setup.

## Monarch: The Distributed Foundation

**What it is:** Monarch is a PyTorch-native distributed programming framework that brings single-controller orchestration to entire clusters. It's implemented with a Python frontend and Rust backend for performance and robustness.

**Why TorchForge needs it:**
- **Single-Controller Model**: Write code that looks like a single Python program but scales to thousands of GPUs
- **Actor Meshes**: Organize processes and actors into scalable, multi-dimensional arrays
- **Fault Tolerance**: Progressive fault handling with fast failure detection and recovery
- **RDMA Support**: Direct GPU-to-GPU memory transfers for efficient data movement

**What it solves:** Traditional SPMD (Single Program, Multiple Data) approaches require complex coordination logic in your code. Monarch abstracts this away, letting you write RL algorithms naturally while it handles distributed complexity underneath.

**Technical capabilities:**
- Scalable messaging with multicast trees
- Multipart messaging for zero-copy data transfers
- Integration with PyTorch's distributed primitives
- Separation of control plane (messaging) and data plane (bulk transfers)

**Where you see it:** Every service creation, actor spawn, and distributed operation in TorchForge runs on Monarch primitives. It's the invisible orchestration layer that makes distributed RL feel simple.

## vLLM: High-Performance Inference

**What it is:** A fast and memory-efficient inference engine optimized for large language models, version 0.10.0 or higher required.

**Why TorchForge needs it:**
- **PagedAttention**: Memory-efficient attention mechanism that reduces memory fragmentation
- **Continuous Batching**: Dynamic batching that maximizes GPU utilization
- **High Throughput**: Handles generation for multiple concurrent rollouts efficiently
- **Production-Ready**: Battle-tested at scale with proven performance

**What it solves:** In RL for LLMs, policy generation is often the bottleneck. Autoregressive generation is costly, and blocking training on it kills throughput. vLLM enables fast, efficient inference that doesn't bottleneck your training loop.

**Integration depth:** TorchForge integrates directly with vLLM's engine, giving you access to customize generation strategies, memory management, and inference logic as your research demands. You control the vLLM configuration while TorchForge handles distributed orchestration.

**Where you see it:** Every policy generation call in your RL code runs through vLLM, whether you're doing synchronous PPO-style rollouts or fully asynchronous off-policy training.

## TorchTitan: Production Training Infrastructure

**What it is:** Meta's production-grade LLM training platform with advanced parallelism support, used to train Llama models on thousands of GPUs.

**Why TorchForge needs it:**
- **FSDP (Fully Sharded Data Parallel)**: Shard parameters, gradients, and optimizer states across GPUs
- **Pipeline Parallelism**: Split model layers across devices with efficient micro-batching
- **Tensor Parallelism**: Split individual tensors across devices for very large models
- **Proven at Scale**: Battle-tested optimizations from production training runs

**What it solves:** Modern models are too large to fit on single GPUs. TorchTitan provides the sophisticated sharding and parallelism strategies needed to train them efficiently, with optimizations proven in production environments.

**Integration depth:** TorchForge provides direct access to TorchTitan's training step logic and sharding strategies, enabling experimentation without framework constraints. You can customize the training loop while leveraging TorchTitan's proven infrastructure.

**Where you see it:** Policy training, whether supervised fine-tuning or RL policy updates, runs through TorchTitan's training infrastructure with your choice of parallelism strategies.

## TorchStore: Distributed Weight Storage

**What it is:** A distributed, in-memory key-value store for PyTorch tensors, built on Monarch primitives, designed specifically for weight synchronization in distributed RL.

**Why TorchForge needs it:**
- **Automatic Resharding**: Handles complex weight transfer between different sharding strategies
- **DTensor Support**: Native support for distributed tensors with automatic topology conversion
- **RDMA Transfers**: High-bandwidth weight movement without synchronous GPU transfers
- **Asynchronous Updates**: Training and inference can read/write weights independently

**What it solves:** In async RL, new policy weights must propagate to all inference replicas. For a 70B parameter model across 16 replicas, this means moving hundreds of gigabytes. Traditional approaches either require synchronous GPU-to-GPU transfers (blocking training), use slow network filesystems (minutes per update), or demand complex manual resharding logic (error-prone). TorchStore solves all of these.

**Technical capabilities:**
- Simple key-value interface with complex optimizations underneath
- Stays distributed across the cluster until requested
- Flexible storage: co-located with trainers, on storage tier, sharded or replicated

**Where you see it:** Weight synchronization between training and inference, allowing training to continue while inference replicas asynchronously fetch updated weights without blocking either process.

## PyTorch Nightly: Cutting-Edge Features

**Why Nightly is required:** TorchForge leverages the latest PyTorch features that aren't yet in stable releases:
- **Native DTensor Support**: Distributed tensors that span multiple devices with automatic sharding
- **Compiled Mode Optimizations**: Performance improvements through torch.compile
- **Advanced Memory Management**: Latest FSDP and memory optimization features
- **Bug Fixes**: Continuous improvements to distributed training primitives

**Where you see it:** Every tensor operation, distributed primitive, and training optimization builds on PyTorch nightly's latest capabilities.

## The Integration Philosophy

TorchForge made a conscious decision not to reinvent the wheel. Instead, we integrate battle-tested components and add the coordination layer that makes them work together seamlessly.

**What you get:**
- **Direct component access**: Customize deeply when your research demands it
- **Proven performance**: Battle-tested at massive scale in production environments
- **Flexible composition**: Mix and match components or replace them with custom implementations
- **Simplified orchestration**: TorchForge coordinates these components so you write algorithms, not infrastructure

**TorchForge's role:** Coordination. We make these powerful but complex components work together seamlessly, exposing simple APIs for distributed RL while preserving deep customization capabilities when you need them.

## Installation

All these components are installed automatically through TorchForge's installation script:

```bash
git clone https://github.com/meta-pytorch/forge.git
cd forge
conda create -n forge python=3.10
conda activate forge
./scripts/install.sh
```

The script provides pre-built wheels for PyTorch nightly, Monarch, vLLM, and TorchTitan, ensuring compatibility and reducing installation time.

See {doc}`getting_started` for detailed installation instructions and troubleshooting.

## See Also

- {doc}`concepts` - Core philosophy and key abstractions
- {doc}`architecture` - How Monarch, Services, and TorchStore work together
- {doc}`rl_workflows` - Using these components to write RL algorithms
- {doc}`getting_started` - Installation and setup guide
- {doc}`usage` - Practical configuration examples
