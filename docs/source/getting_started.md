# Getting Started

Welcome to TorchForge! This guide will walk you through installing TorchForge, verifying your setup, and running your first training job.

## Prerequisites

Before installing TorchForge, ensure your system meets the following requirements:

### System Requirements

| Component | Requirement | Notes |
|-----------|-------------|-------|
| **Operating System** | Linux (Fedora/Ubuntu/Debian) | MacOS and Windows not currently supported |
| **Python** | 3.10 or higher | Python 3.11 recommended |
| **GPU** | NVIDIA with CUDA support | AMD GPUs not currently supported |
| **CUDA** | 12.8 or higher | Required for GPU training |
| **Minimum GPUs** | 2 for SFT, 3 for GRPO | More GPUs enable larger models |
| **RAM** | 32GB+ recommended | Depends on model size |
| **Disk Space** | 50GB+ free | For models, datasets, and checkpoints |

### Required Tools

1. **Conda or Miniconda**: For environment management
   - Download from [conda.io](https://docs.conda.io/en/latest/miniconda.html)

2. **GitHub CLI (gh)**: Required for downloading pre-packaged dependencies
   - Install instructions: [github.com/cli/cli#installation](https://github.com/cli/cli#installation)
   - After installing, authenticate with: `gh auth login`
   - You can use either HTTPS or SSH as the authentication protocol

3. **Git**: For cloning the repository
   - Usually pre-installed on Linux systems
   - Verify with: `git --version`

## Installation

TorchForge offers two installation methods. Choose the one that fits your setup:

### Method 1: Basic Installation (Recommended)

This method uses pre-packaged wheels for all dependencies, making installation faster and more reliable.

**Step 1: Clone the Repository**

```bash
git clone https://github.com/meta-pytorch/forge.git
cd forge
```

**Step 2: Create Conda Environment**

```bash
conda create -n forge python=3.10
conda activate forge
```

**Step 3: Run Installation Script**

```bash
./scripts/install.sh
```

The installation script will:
- Install system dependencies using DNF (or your package manager)
- Download pre-built wheels for PyTorch nightly, Monarch, vLLM, and TorchTitan
- Install TorchForge and all Python dependencies

```{tip}
**Using sudo instead of conda**: If you prefer installing system packages directly rather than through conda, use:
`./scripts/install.sh --use-sudo`
```

**Step 4: Verify Installation**

Test that TorchForge is properly installed:

```bash
python -c "import forge; print(forge.__version__)"
```

### Method 2: Meta Internal Installation (Alternative)

For Meta employees or those with access to Meta's internal tools:

**Step 1: Install uv Package Manager**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Step 2: Clone and Setup**

```bash
git clone https://github.com/meta-pytorch/forge
cd forge
uv sync --all-extras
source .venv/bin/activate
```

**Step 3: Configure CUDA**

```bash
# Install CUDA if needed
feature install --persist cuda_12_9

# Set environment variables
export CUDA_VERSION=12.9
export NVCC=/usr/local/cuda-$CUDA_VERSION/bin/nvcc
export CUDA_NVCC_EXECUTABLE=/usr/local/cuda-$CUDA_VERSION/bin/nvcc
export CUDA_HOME=/usr/local/cuda-$CUDA_VERSION
export PATH="$CUDA_HOME/bin:$PATH"
export CUDA_INCLUDE_DIRS=$CUDA_HOME/include
export CUDA_CUDART_LIBRARY=$CUDA_HOME/lib64/libcudart.so
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

**Step 4: Build vLLM from Source**

```bash
git clone https://github.com/vllm-project/vllm.git --branch v0.10.0
cd vllm
python use_existing_torch.py
uv pip install -r requirements/build.txt
uv pip install --no-build-isolation -e .
```

```{warning}
When adding packages to `pyproject.toml`, use `uv sync --inexact` to avoid removing Monarch and vLLM.
```

## Verifying Your Setup

After installation, verify that all components are working correctly:

### Check GPU Availability

```bash
python -c "import torch; print(f'GPUs available: {torch.cuda.device_count()}')"
```

Expected output: `GPUs available: 2` (or more)

### Check CUDA Version

```bash
python -c "import torch; print(f'CUDA version: {torch.version.cuda}')"
```

Expected output: `CUDA version: 12.8` (or higher)

### Check Dependencies

```bash
# Check vLLM
python -c "import vllm; print(f'vLLM: {vllm.__version__}')"

# Check TorchForge modules
python -c "from forge import actors, controller, data; print('All modules imported successfully')"
```

## Quick Start Examples

Now that TorchForge is installed, let's run some training examples:

### Example 1: Supervised Fine-Tuning (SFT)

This example fine-tunes Llama 3 8B on your data. **Requires: 2+ GPUs**

**Step 1: Download the Model**

```bash
uv run forge download meta-llama/Meta-Llama-3.1-8B-Instruct \
  --output-dir /tmp/Meta-Llama-3.1-8B-Instruct \
  --ignore-patterns "original/consolidated.00.pth"
```

```{note}
Model downloads require Hugging Face authentication. Run `huggingface-cli login` first if you haven't already.
```

**Step 2: Run Training**

```bash
uv run forge run --nproc_per_node 2 \
  apps/sft/main.py \
  --config apps/sft/llama3_8b.yaml
```

**What's Happening:**
- `--nproc_per_node 2`: Use 2 GPUs for training
- `apps/sft/main.py`: SFT training script
- `--config apps/sft/llama3_8b.yaml`: Configuration file with hyperparameters

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
- GPU 0: Policy model (being trained)
- GPU 1: Reference model (frozen baseline)
- GPU 2: Reward model (scoring outputs)

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

This loads your model and starts an interactive generation session.

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
  strategy: fsdp
  precision: bf16

checkpointing:
  save_interval: 1000
  output_dir: /tmp/checkpoints
```

**Key Sections:**
- **model**: Model path and settings
- **training**: Hyperparameters like batch size and learning rate
- **distributed**: Multi-GPU strategy and precision
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

## Next Steps

Now that you have TorchForge installed and verified:

1. **Learn the Concepts**: Read {doc}`concepts` to understand TorchForge's architecture
2. **Explore Examples**: Check the `apps/` directory for more training examples
3. **Customize Training**: See {doc}`usage` for configuration patterns
4. **Read Tutorials**: Follow {doc}`tutorials` for step-by-step guides
5. **API Documentation**: Explore {doc}`api` for detailed API reference

## Getting Help

If you encounter issues:

1. **Check the FAQ**: {doc}`faq` covers common questions and solutions
2. **Search Issues**: Look through [GitHub Issues](https://github.com/meta-pytorch/forge/issues)
3. **File a Bug Report**: Create a new issue with:
   - Your system configuration
   - Full error message
   - Steps to reproduce
   - Expected vs actual behavior

```{tip}
When filing issues, include the output of:
```bash
python -c "import torch; import forge; print(f'PyTorch: {torch.__version__}\\nForge: {forge.__version__}\\nCUDA: {torch.version.cuda}\\nGPUs: {torch.cuda.device_count()}')"
```
```

## Additional Resources

- **GitHub Repository**: [github.com/meta-pytorch/forge](https://github.com/meta-pytorch/forge)
- **Example Notebooks**: Check `demo.ipynb` in the repository root
- **Contributing Guide**: [CONTRIBUTING.md](https://github.com/meta-pytorch/forge/blob/main/CONTRIBUTING.md)
- **Code of Conduct**: [CODE_OF_CONDUCT.md](https://github.com/meta-pytorch/forge/blob/main/CODE_OF_CONDUCT.md)

---

**Ready to start training?** Head to {doc}`usage` for practical configuration examples and workflows.
