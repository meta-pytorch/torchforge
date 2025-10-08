# Concepts

This guide covers the fundamental concepts and architecture behind TorchForge,
helping you understand how the system works and how to effectively use its components.

## Architecture Overview

TorchForge is built as a modular, distributed system designed specifically for post-training large language models. The architecture follows a clear separation of concerns, with specialized components handling different aspects of the training pipeline.

### Core Components

The system is organized into several key layers:

**Controller**
: The orchestration component responsible for coordinating distributed training across multiple GPUs and nodes. The [Controller](api_controller.md) is the orchestration hub that manages resource allocation and scheduling, inter-process communication coordination, fault tolerance and recovery, and distributed state management.

**Actor**
: A component responsible for executing training or inference tasks. [Actors](api_actors.md) are the workhorses of TorchForge, handling model operations:
- **Policy Actor**: Trains the primary model being optimized
- **Reward Actor**: Evaluates generated outputs and provides reward signals
- **Reference Actor**: Maintains a frozen baseline for KL divergence computation
- **Inference Actor**: Handles efficient generation of model outputs

**Data Layer**
: The [Data](api_data.md) layer manages all aspects of data handling including dataset loading and preprocessing, batch construction and sampling, data distribution across workers, and custom data models and transformations. A **batch** represents the number of training examples processed together in a single forward and backward pass through the model.

**Environment Layer**
: [Environments](api_envs.md) provide execution contexts for both training and inference operations. They handle training environments for supervised and reinforcement learning, inference environments for generation and evaluation, and resource management with device allocation.

## Post-Training and Fine-Tuning

**Post-Training** refers to training phases that occur after initial pre-training. While **pre-training** is the initial phase where a model learns general language understanding from large text corpora, post-training adapts the model to specific tasks and behaviors.

**Fine-Tuning** is the process of adapting a pre-trained model to a specific task or domain by continuing training on task-specific data. TorchForge supports two primary post-training paradigms:

### Supervised Fine-Tuning (SFT)

**SFT** is a fine-tuning approach that trains a model on labeled input-output pairs using supervised learning. The process adapts a pre-trained model to specific tasks using structured **prompt**-response pairs:

1. **Data Preparation**: Structure your training data as prompt-response pairs, where a prompt is the input text provided to elicit a specific generation
2. **Model Setup**: Load a pre-trained model **checkpoint** (a saved snapshot of model weights and optimizer state)
3. **Training**: Optimize the model using a **loss function** (a mathematical function measuring the difference between predicted and actual outputs) to generate target responses given prompts
4. **Evaluation**: Validate performance on held-out examples

SFT is typically the first step in post-training, establishing a baseline before more advanced techniques.

### Generalized Reward Policy Optimization (GRPO)

**GRPO** is an advanced **reinforcement learning** algorithm for aligning large language models with human preferences and optimizing for specific reward functions. In reinforcement learning, an agent learns to make decisions by receiving rewards or penalties for its actions.

The GRPO workflow involves:

1. **Generation Phase**: The **policy model** (the model being trained to generate outputs that maximize rewards) generates multiple candidate responses for each prompt
2. **Scoring Phase**: The **reward model** (a model trained to score output quality) evaluates each candidate and assigns scores
3. **Advantage Computation**: Calculate advantages using reward values and **reference model** KL divergence. The reference model is a frozen copy of the policy model used to prevent the policy from deviating too far from its initial behavior
4. **Policy Update**: Update the policy to increase probability of high-reward actions during each **episode** (one complete training iteration)
5. **Constraint Enforcement**: Apply KL penalties to prevent policy collapse

GRPO requires a minimum of 3 GPUs:
- GPU 0: Policy model (trainable)
- GPU 1: Reference model (frozen)
- GPU 2: Reward model (frozen)

**RLHF (Reinforcement Learning from Human Feedback)** is a related training approach that uses human preferences to train a reward model, which then guides policy optimization through techniques like GRPO.

## Distributed Training

**Distributed Training** trains a model across multiple GPUs or machines to handle larger models and datasets. TorchForge leverages PyTorch's distributed capabilities to scale training across multiple devices. A **GPU (Graphics Processing Unit)** is the hardware accelerator used for parallel computation in deep learning training and **inference** (the process of using a trained model to make predictions without updating parameters).

### Data Parallelism

Each GPU processes different batches of data using the same model replica. Gradients are synchronized across devices during backpropagation using collective communication operations like **All-Reduce** (synchronize gradients across all workers).

### Model Parallelism

**Model Parallelism** is a distributed training strategy where different parts of a model are placed on different devices, necessary for models too large to fit on a single GPU. TorchForge supports several approaches:

**FSDP (Fully Sharded Data Parallel)**
: A PyTorch distributed training strategy that implements **sharding** (splitting model parameters, gradients, or optimizer states across multiple devices) to reduce memory footprint per device.

**Tensor Parallelism**
: A model parallelism strategy that splits individual tensors like large weight matrices across multiple devices, where each device computes a portion of the operation.

**Pipeline Parallelism**
: A form of model parallelism where model layers are partitioned across devices and micro-batches are pipelined through the stages for efficient utilization.

**DTensor (Distributed Tensor)**
: A tensor that is sharded across multiple devices in a distributed training setup, enabling transparent model parallelism.

### Communication Strategies

Efficient communication is critical for distributed training:
- **All-Reduce**: Synchronize gradients across all workers
- **Broadcast**: Share model parameters from one process to others
- **Point-to-Point**: Direct communication between specific processes

## Data Models and Types

Structured [data models](api_data.md) ensure type safety and consistency throughout the training pipeline:

**Prompt Models**
: Structured input representations containing the text provided to language models

**Response Models**
: Generated output with metadata, including the completion text and generation parameters

**Episode Models**
: Complete RL interaction sequences capturing prompt, response, and reward information

**Batch Models**
: Optimized batch representations for efficient parallel processing

**Token**
: The basic unit of text processed by language models, typically representing words, subwords, or characters

## Resource Management

Effective resource management is crucial for training **LLMs (Large Language Models)** - deep learning models trained on vast amounts of text data, capable of understanding and generating human-like text.

### Memory Optimization

TorchForge employs several memory optimization techniques:

**Gradient Checkpointing**
: Trade computation for memory by recomputing activations during the backward pass instead of storing them

**Mixed Precision**
: Use FP16/BF16 for reduced memory footprint while maintaining model quality

**Activation Offloading**
: Move activations to CPU when not needed on GPU, freeing up device memory

**Parameter Sharding**
: Distribute model parameters across devices using techniques like FSDP

### Compute Optimization

Maximize GPU utilization through:

**Asynchronous Execution**
: Overlap communication with computation to hide latency

**Batch Size Tuning**
: Balance memory usage and throughput for optimal training speed

**Dynamic Batching**
: Group requests efficiently for inference to maximize hardware utilization

**Kernel Fusion**
: Combine operations to reduce memory bandwidth requirements

## Integration Points

TorchForge integrates with several PyTorch ecosystem projects to provide comprehensive functionality:

### PyTorch Nightly

Built on the latest PyTorch features, including:
- Native DTensor support for distributed tensors
- Compiled mode optimizations
- Advanced memory management

### Monarch

**Monarch** is a Meta-developed optimization framework for PyTorch that TorchForge depends on for certain operations, providing enhanced optimization capabilities.

### vLLM

**vLLM** is a fast and memory-efficient inference engine for large language models that TorchForge integrates for **generation** (producing text output from a language model given an input prompt):
- Paged attention for memory efficiency
- Continuous batching for throughput
- Speculative decoding support

### TorchTitan

**TorchTitan** is a PyTorch-based framework that TorchForge builds upon for certain distributed training capabilities, leveraging its foundations for scaling to large clusters.

## Configuration Management

TorchForge uses a hierarchical configuration system with multiple layers of settings:

1. **Default Configs**: Sensible defaults for common scenarios
2. **Model Configs**: Pre-configured setups for popular models
3. **User Configs**: Custom overrides for specific needs
4. **Runtime Configs**: Dynamic adjustments during execution

See [Usage](usage.md) for detailed configuration examples.

## Training Lifecycle

Understanding the complete training lifecycle helps you effectively use TorchForge:

### Checkpointing

A **checkpoint** is a saved snapshot of model weights and optimizer state, allowing training to be resumed or the model to be deployed for inference. Regular checkpointing provides fault tolerance and enables experimentation with different training configurations.

### Warmup

**Warmup** is a training phase where the learning rate gradually increases from a small value to the target learning rate, helping stabilize early training and prevent gradient explosions.

### Loss Functions

TorchForge provides specialized [loss functions](api_losses.md) for post-training:
- **Policy Gradient Losses**: For reinforcement learning optimization
- **Regularization Terms**: KL divergence constraints
- **Multi-Objective Losses**: Combine multiple training signals

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

## See Also

- {doc}`getting_started` - Installation and setup guide
- {doc}`usage` - Practical usage examples
- {doc}`tutorials` - Step-by-step guides
- {doc}`api` - Complete API reference
- {doc}`faq` - Common questions and troubleshooting
