# Get Started

Welcome to TorchForge! This guide will help you get up and running with TorchForge, a PyTorch-native platform specifically designed for post-training generative AI models.

TorchForge specializes in post-training techniques for large language models, including:

- **Supervised Fine-Tuning (SFT)**: Adapt pre-trained models to specific tasks using labeled data
- **Generalized Reward Policy Optimization (GRPO)**: Advanced reinforcement learning for model alignment
- **Multi-GPU Distributed Training**: Efficient scaling across multiple GPUs and nodes

## Prerequisites

Before installing TorchForge, ensure you have:

- **Python 3.10 or 3.11** (Python 3.12+ not currently supported)
- **CUDA-compatible GPU(s)**: Minimum 3 GPUs required for GRPO training
- **conda or mamba** for environment management
- **Git and GitHub CLI (gh)**: Required for downloading compatible packages

## Installation

### Basic Installation

The basic installation uses pre-packaged wheels with all necessary dependencies.

1. **Install GitHub CLI** if not already installed:

   ```bash
   gh auth login  # Login with your GitHub account
   ```

2. **Create and activate a conda environment**:

   ```bash
   conda create -n forge python=3.10
   conda activate forge
   ```

3. **Clone the repository and install**:
   ```bash
   git clone https://github.com/meta-pytorch/forge
   cd forge
   ./scripts/install.sh
   ```

   Optional: Use `--use-sudo` flag to install system packages instead of conda packages:
   ```bash
   ./scripts/install.sh --use-sudo
   ```

## Verify Installation

Test your installation by running a simple GRPO training example:

```bash
python -m apps.grpo.main --config apps/grpo/qwen3_1_7b.yaml
```

If successful, you should see output confirming that GRPO training is running.

## Quick Start Examples

### Example 1: Supervised Fine-Tuning (SFT)

Fine-tune a Llama3 8B model using supervised learning:

1. **Download the model**:
   ```bash
   uv run forge download meta-llama/Meta-Llama-3.1-8B-Instruct \
     --output-dir /tmp/Meta-Llama-3.1-8B-Instruct \
     --ignore-patterns "original/consolidated.00.pth"
   ```

2. **Run SFT training**:
   ```bash
   uv run forge run --nproc_per_node 2 apps/sft/main.py \
     --config apps/sft/llama3_8b.yaml
   ```

This will:
- Load the Llama3 8B model
- Fine-tune on the Alpaca dataset
- Use 2 GPUs for training
- Save checkpoints every 500 steps

### Example 2: GRPO Reinforcement Learning

Train a model using GRPO for better alignment:

```bash
python -m apps.grpo.main --config apps/grpo/qwen3_1_7b.yaml
```

This example:
- Uses the Qwen3 1.7B model
- Trains on GSM8K math problems
- Implements reward-based optimization
- Requires minimum 3 GPUs

## What's Next?

Now that you have TorchForge running, explore these resources to deepen your understanding:

- **[Usage](usage.md)** - Configuration files, common use cases, and practical examples
- **[Concepts](concepts.md)** - Architecture, algorithms, and design principles
- **[API Reference](api.md)** - Detailed documentation of all modules and classes
- **[Tutorials](tutorials.md)** - Step-by-step guides for specific scenarios

### Additional Resources

- Check the `apps/` directory for more training examples
- Join the community on [GitHub](https://github.com/pytorch-labs/forge) and [PyTorch Forums](https://discuss.pytorch.org/)
- Report issues or contribute improvements

## Getting Help

- **Documentation**: Browse the full [API Reference](./api.html)
- **Examples**: Check configuration files in [apps/*/](https://github.com/meta-pytorch/forge/tree/main/apps)
- **Issues**: Report bugs on [GitHub Issues](https://github.com/meta-pytorch/forge/issues)
- **Discussions**: Join the [PyTorch Forums](https://discuss.pytorch.org/)
