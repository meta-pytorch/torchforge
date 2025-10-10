# Getting Started

This guide will walk you through installing TorchForge, understanding its dependencies, verifying your setup, and running your first training job.

## System Requirements

Before installing TorchForge, ensure your system meets the following requirements.

| Component | Requirement | Notes |
|-----------|-------------|-------|
| **Operating System** | Linux (Fedora/Ubuntu/Debian) | MacOS and Windows not currently supported |
| **Python** | 3.10 or higher | Python 3.11 recommended |
| **GPU** | NVIDIA with CUDA support | AMD GPUs not currently supported |
| **CUDA** | 12.8 or higher | Required for GPU training |
| **Minimum GPUs** | 2 for SFT, 3 for GRPO | More GPUs enable larger models |
| **RAM** | 32GB+ recommended | Depends on model size |
| **Disk Space** | 50GB+ free | For models, datasets, and checkpoints |

## Prerequisites

- **Conda or Miniconda**: For environment management
  - Download from [conda.io](https://docs.conda.io/en/latest/miniconda.html)

- **GitHub CLI (gh)**: Required for downloading pre-packaged dependencies
  - Install instructions: [github.com/cli/cli#installation](https://github.com/cli/cli#installation)
  - After installing, authenticate with: `gh auth login`
  - You can use either HTTPS or SSH as the authentication protocol

- **Git**: For cloning the repository
  - Usually pre-installed on Linux systems
  - Verify with: `git --version`

## Understanding TorchForge's Dependencies

TorchForge is built on a carefully curated stack of components, each solving specific challenges in distributed RL. Understanding these dependencies helps you troubleshoot issues and customize your setup.

### Monarch

**What it is:** Monarch is a PyTorch-native distributed programming framework that brings single-controller orchestration to entire clusters.

**Why TorchForge needs it:**
- **Single-Controller Model**: Write code that looks like a single Python program but scales to thousands of GPUs
- **Actor Meshes**: Organize processes and actors into scalable, multi-dimensional arrays
- **Fault Tolerance**: Progressive fault handling with fast failure detection and recovery
- **RDMA Support**: Direct GPU-to-GPU memory transfers for efficient data movement

**What it solves:** Traditional SPMD (Single Program, Multiple Data) approaches require complex coordination logic in your code. Monarch abstracts this away, letting you write RL algorithms naturally while it handles distributed complexity.

**Technical details:** Monarch is implemented with a Python frontend and a Rust backend for performance and robustness. It provides:
- Scalable messaging with multicast trees
- Multipart messaging for zero-copy data transfers
- Integration with PyTorch's distributed primitives

### vLLM

**What it is:** A fast and memory-efficient inference engine optimized for large language models.

**Why TorchForge needs it:**
- **PagedAttention**: Memory-efficient attention mechanism that reduces memory fragmentation
- **Continuous Batching**: Dynamic batching that maximizes GPU utilization
- **High Throughput**: Handles generation for multiple concurrent rollouts efficiently
- **Production-Ready**: Battle-tested at scale with proven performance

**What it solves:** In RL for LLMs, policy generation is often the bottleneck. Autoregressive generation is costly, and blocking training on it kills throughput. vLLM enables fast, efficient inference that doesn't bottleneck your training loop.

**Technical details:** vLLM version 0.10.0+ is required. TorchForge integrates directly with vLLM's engine, giving you access to customize generation strategies, memory management, and inference logic.

### TorchTitan

**What it is:** Meta's production-grade LLM training platform with advanced parallelism support.

**Why TorchForge needs it:**
- **FSDP (Fully Sharded Data Parallel)**: Shard parameters, gradients, and optimizer states across GPUs
- **Pipeline Parallelism**: Split model layers across devices with efficient micro-batching
- **Tensor Parallelism**: Split individual tensors across devices for very large models
- **Proven at Scale**: Used to train Llama models on thousands of GPUs

**What it solves:** Modern models are too large to fit on single GPUs. TorchTitan provides the sophisticated sharding and parallelism strategies needed to train them efficiently, with optimizations battle-tested in production.

**Technical details:** TorchForge integrates with TorchTitan for training step logic and sharding strategies, enabling experimentation without framework constraints.

### TorchStore

**What it is:** A distributed, in-memory key-value store for PyTorch tensors, built on Monarch.

**Why TorchForge needs it:**
- **Automatic Resharding**: Handles complex weight transfer between different sharding strategies
- **DTensor Support**: Native support for distributed tensors
- **RDMA Transfers**: High-bandwidth weight movement without synchronous GPU transfers
- **Asynchronous Updates**: Training and inference can read/write weights independently

**What it solves:** In async RL, new policy weights must propagate to all inference replicas. For a 70B parameter model across 16 replicas, this means moving hundreds of gigabytes. TorchStore makes this efficient, handling resharding automatically and using RDMA for fast transfers.

**Technical details:** TorchStore provides a simple key-value interface while optimizing data movement behind the scenes, staying distributed across the cluster until requested.

### PyTorch Nightly

**Why Nightly:** TorchForge requires the latest PyTorch features:
- **Native DTensor Support**: Distributed tensors that span multiple devices
- **Compiled Mode Optimizations**: Performance improvements through torch.compile
- **Advanced Memory Management**: Latest FSDP and memory optimization features
- **Bug Fixes**: Continuous improvements to distributed training primitives

**Installation note:** The installation script provides pre-built wheels with PyTorch nightly already included.

## Installation

TorchForge uses pre-packaged wheels for all dependencies, making installation faster and more reliable.

1. **Clone the Repository**

   ```bash
   git clone https://github.com/meta-pytorch/forge.git
   cd forge
   ```

2. **Create Conda Environment**

   ```bash
   conda create -n forge python=3.10
   conda activate forge
   ```

3. **Run Installation Script**

   ```bash
   ./scripts/install.sh
   ```

   The installation script will:
   - Install system dependencies using DNF (or your package manager)
   - Download pre-built wheels for PyTorch nightly, Monarch, vLLM, and TorchTitan
   - Install TorchForge and all Python dependencies
   - Configure the environment for GPU training

   ```{tip}
   **Using sudo instead of conda**: If you prefer installing system packages directly rather than through conda, use:
   `./scripts/install.sh --use-sudo`
   ```

4. **Verify Installation**

   Test that TorchForge is properly installed:

   ```bash
   python -c "import forge; print(f'TorchForge version: {forge.__version__}')"
   python -c "import monarch; print('Monarch: OK')"
   python -c "import vllm; print(f'vLLM version: {vllm.__version__}')"
   ```

   ```{warning}
   When adding packages to `pyproject.toml`, use `uv sync --inexact` to avoid removing Monarch and vLLM.
   ```

## Verifying Your Setup

After installation, verify that all components are working correctly:

1. **Check GPU Availability**

   ```bash
   python -c "import torch; print(f'GPUs available: {torch.cuda.device_count()}')"
   ```

   Expected output: `GPUs available: 2` (or more)

2. **Check CUDA Version**

   ```bash
   python -c "import torch; print(f'CUDA version: {torch.version.cuda}')"
   ```

   Expected output: `CUDA version: 12.8` (or higher)

3. **Check All Dependencies**

   ```bash
   # Check core components
   python -c "import torch, forge, monarch, vllm; print('All imports successful')"

   # Check specific versions
   python -c "
   import torch
   import forge
   import vllm

   print(f'PyTorch: {torch.__version__}')
   print(f'TorchForge: {forge.__version__}')
   print(f'vLLM: {vllm.__version__}')
   print(f'CUDA: {torch.version.cuda}')
   print(f'GPUs: {torch.cuda.device_count()}')
   "
   ```

4. **Verify Monarch**

   ```bash
   python -c "
   from monarch.actor import Actor, this_host

   # Test basic Monarch functionality
   procs = this_host().spawn_procs({'gpus': 1})
   print('Monarch: Process spawning works')
   "
   ```

## Quick Start Examples

Now that TorchForge is installed, let's run some training examples.

### Example 1: Supervised Fine-Tuning (SFT)

Fine-tune Llama 3 8B on your data. **Requires: 2+ GPUs**

1. **Download the Model**

   ```bash
   uv run forge download meta-llama/Meta-Llama-3.1-8B-Instruct \
     --output-dir /tmp/Meta-Llama-3.1-8B-Instruct \
     --ignore-patterns "original/consolidated.00.pth"
   ```

   ```{note}
   Model downloads require Hugging Face authentication. Run `huggingface-cli login` first if you haven't already.
   ```

2. **Run Training**

   ```bash
   uv run forge run --nproc_per_node 2 \
     apps/sft/main.py \
     --config apps/sft/llama3_8b.yaml
   ```

   **What's Happening:**
   - `--nproc_per_node 2`: Use 2 GPUs for training
   - `apps/sft/main.py`: SFT training script
   - `--config apps/sft/llama3_8b.yaml`: Configuration file with hyperparameters
   - **TorchTitan** handles model sharding across the 2 GPUs
   - **Monarch** coordinates the distributed training

   **Expected Output:**
   ```
   Initializing process group...
   Loading model from /tmp/Meta-Llama-3.1-8B-Instruct...
   Starting training...
   Epoch 1/10 | Step 100 | Loss: 2.45 | LR: 0.0001
   ...
   ```

### Example 2: GRPO Training

Train a model using reinforcement learning with GRPO. **Requires: 3+ GPUs**

```bash
python -m apps.grpo.main --config apps/grpo/qwen3_1_7b.yaml
```

**What's Happening:**
- GPU 0: Policy model (being trained, powered by TorchTitan)
- GPU 1: Reference model (frozen baseline)
- GPU 2: Reward model (scoring outputs, powered by vLLM)
- **Monarch** orchestrates all three components
- **TorchStore** handles weight synchronization from training to inference

**Expected Output:**
```
Initializing GRPO training...
Loading policy model on GPU 0...
Loading reference model on GPU 1...
Loading reward model on GPU 2...
Episode 1 | Avg Reward: 0.75 | KL Divergence: 0.12
...
```

### Example 3: Inference with vLLM

Generate text using a trained model:

```bash
python -m apps.vllm.main --config apps/vllm/llama3_8b.yaml
```

This loads your model with vLLM and starts an interactive generation session.

## Understanding Configuration Files

TorchForge uses YAML configuration files to manage training parameters. Let's examine a typical config:

```yaml
# Example: apps/sft/llama3_8b.yaml
model:
  name: meta-llama/Meta-Llama-3.1-8B-Instruct
  path: /tmp/Meta-Llama-3.1-8B-Instruct

training:
  batch_size: 4
  learning_rate: 1e-5
  num_epochs: 10
  gradient_accumulation_steps: 4

distributed:
  strategy: fsdp  # Managed by TorchTitan
  precision: bf16

checkpointing:
  save_interval: 1000
  output_dir: /tmp/checkpoints
```

**Key Sections:**
- **model**: Model path and settings
- **training**: Hyperparameters like batch size and learning rate
- **distributed**: Multi-GPU strategy (FSDP, tensor parallel, etc.) handled by TorchTitan
- **checkpointing**: Where and when to save model checkpoints

See {doc}`usage` for detailed configuration options.

## Common Installation Issues

### Issue: `gh: command not found`

**Solution**: Install GitHub CLI:
```bash
# On Ubuntu/Debian
sudo apt install gh

# On Fedora
sudo dnf install gh

# Then authenticate
gh auth login
```

### Issue: `CUDA out of memory`

**Solution**: Reduce batch size in your config file:
```yaml
training:
  batch_size: 2  # Reduced from 4
  gradient_accumulation_steps: 8  # Increased to maintain effective batch size
```

### Issue: `ImportError: No module named 'torch'`

**Solution**: Ensure you activated the conda environment:
```bash
conda activate forge
```

### Issue: vLLM wheel download fails

**Solution**: The vLLM wheel is hosted on GitHub releases. Ensure you're authenticated with `gh auth login` and have internet access.

### Issue: `Unsupported GPU architecture`

**Solution**: Check your GPU compute capability:
```bash
python -c "import torch; print(torch.cuda.get_device_capability())"
```

TorchForge requires compute capability 7.0 or higher (Volta architecture or newer).

### Issue: Monarch actor spawn failures

**Symptom**: Errors like "Failed to spawn actors" or "Process allocation failed"

**Solution**: Verify your GPU count matches your configuration:
```bash
nvidia-smi  # Check available GPUs
```

Ensure your config requests fewer processes than available GPUs.

## Next Steps

Now that you have TorchForge installed and verified:

1. **Learn the Concepts**: Read {doc}`concepts` to understand TorchForge's architecture, including Monarch, Services, and TorchStore
2. **Explore Examples**: Check the `apps/` directory for more training examples
3. **Customize Training**: See {doc}`usage` for configuration patterns
4. **Read Tutorials**: Follow {doc}`tutorials` for step-by-step guides
5. **API Documentation**: Explore {doc}`api` for detailed API reference

## Getting Help

If you encounter issues:

1. **Check the FAQ**: {doc}`faq` covers common questions and solutions
2. **Search Issues**: Look through [GitHub Issues](https://github.com/meta-pytorch/forge/issues)
3. **File a Bug Report**: Create a new issue with:
   - Your system configuration (output of diagnostic command below)
   - Full error message
   - Steps to reproduce
   - Expected vs actual behavior

**Diagnostic command:**
```bash
python -c "
import torch
import forge

try:
    import monarch
    monarch_status = 'OK'
except Exception as e:
    monarch_status = str(e)

try:
    import vllm
    vllm_version = vllm.__version__
except Exception as e:
    vllm_version = str(e)

print(f'PyTorch: {torch.__version__}')
print(f'TorchForge: {forge.__version__}')
print(f'Monarch: {monarch_status}')
print(f'vLLM: {vllm_version}')
print(f'CUDA: {torch.version.cuda}')
print(f'GPUs: {torch.cuda.device_count()}')
"
```

Include this output in your bug reports!

## Additional Resources

- **GitHub Repository**: [github.com/meta-pytorch/forge](https://github.com/meta-pytorch/forge)
- **Example Notebooks**: Check `demo.ipynb` in the repository root
- **Contributing Guide**: [CONTRIBUTING.md](https://github.com/meta-pytorch/forge/blob/main/CONTRIBUTING.md)
- **Code of Conduct**: [CODE_OF_CONDUCT.md](https://github.com/meta-pytorch/forge/blob/main/CODE_OF_CONDUCT.md)
- **Monarch Documentation**: [meta-pytorch.org/monarch](https://meta-pytorch.org/monarch)
- **vLLM Documentation**: [docs.vllm.ai](https://docs.vllm.ai)
- **TorchTitan**: [github.com/pytorch/torchtitan](https://github.com/pytorch/torchtitan)

---

**Ready to start training?** Head to {doc}`usage` for practical configuration examples and workflows, or dive into {doc}`concepts` to understand how all the pieces work together.
