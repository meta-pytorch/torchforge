# TorchForge Documentation

**TorchForge** is a PyTorch-native platform built for post-training generative AI models and agentic development. Designed with the principle that researchers should write algorithms, not infrastructure.

```{note}
**Early Development:** TorchForge is currently experimental. Expect bugs, incomplete features, and API changes. Please file issues on [GitHub](https://github.com/meta-pytorch/forge) for bug reports and feature requests.
```

## Why TorchForge?

TorchForge introduces a "service"-centric architecture that provides the right abstractions for distributed complexity:

- **Usability for Rapid Research**: Isolate your RL algorithms from infrastructure concerns
- **Hackability for Power Users**: Modify any part of the RL loop without touching infrastructure code
- **Scalability**: Seamlessly shift between async and synchronous training across thousands of GPUs

## Core Capabilities

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Supervised Fine-Tuning (SFT)
:link: concepts
:link-type: doc

Adapt pre-trained models to specific tasks using labeled data. Perfect for creating specialized models from foundation models.
:::

:::{grid-item-card} Reinforcement Learning (GRPO)
:link: concepts
:link-type: doc

Advanced policy optimization using Generalized Reward Policy Optimization for aligning models with human preferences and reward functions.
:::

:::{grid-item-card} Distributed Training
:link: concepts
:link-type: doc

Built-in support for multi-GPU and multi-node training with FSDP, tensor parallelism, and pipeline parallelism.
:::

:::{grid-item-card} Integration Ecosystem
:link: concepts
:link-type: doc

Seamlessly integrates with PyTorch nightly, Monarch, vLLM, and TorchTitan for a complete training and inference pipeline.
:::

::::

## Getting Started Paths

Choose your journey based on your experience level and goals:

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} 🚀 New to TorchForge?
:link: getting_started
:link-type: doc

Start here for installation instructions, system requirements, and your first training run.

**Time to first run: ~15 minutes**
:::

:::{grid-item-card} 📚 Understanding the Architecture
:link: concepts
:link-type: doc

Learn about TorchForge's architecture, key concepts, and how the components work together.

**Recommended for researchers**
:::

:::{grid-item-card} 💻 Practical Usage
:link: usage
:link-type: doc

Configuration patterns, common workflows, and real-world usage scenarios.

**For hands-on development**
:::

:::{grid-item-card} 📖 API Reference
:link: api
:link-type: doc

Complete API documentation for all modules, classes, and functions.

**For in-depth customization**
:::

::::

## Quick Example

Here's what a simple SFT training run looks like:

```bash
# Download a model
uv run forge download meta-llama/Meta-Llama-3.1-8B-Instruct \
  --output-dir /tmp/Meta-Llama-3.1-8B-Instruct

# Run supervised fine-tuning
uv run forge run --nproc_per_node 2 \
  apps/sft/main.py --config apps/sft/llama3_8b.yaml
```

See {doc}`getting_started` for complete installation and setup instructions.

## System Requirements

| Component | Requirement |
|-----------|-------------|
| **Python** | 3.10+ |
| **Operating System** | Linux (tested on Fedora/Ubuntu) |
| **GPU** | NVIDIA with CUDA support |
| **Minimum GPUs** | 2 for SFT, 3 for GRPO |
| **Dependencies** | PyTorch nightly, Monarch, vLLM, TorchTitan |

## Supported Models

TorchForge includes pre-configured setups for popular models:

- **Llama 3 8B**: Production-ready configuration for supervised fine-tuning
- **Qwen 3.1 7B**: Optimized settings for GRPO training
- **Qwen 3 8B**: Multi-node training configurations
- **Custom Models**: Extensible architecture for any transformer-based model

## Community & Support

- **Documentation**: You're reading it! Use the navigation above to explore
- **GitHub Issues**: [Report bugs and request features](https://github.com/meta-pytorch/forge/issues)
- **Contributing**: See [CONTRIBUTING.md](https://github.com/meta-pytorch/forge/blob/main/CONTRIBUTING.md)
- **Code of Conduct**: [CODE_OF_CONDUCT.md](https://github.com/meta-pytorch/forge/blob/main/CODE_OF_CONDUCT.md)

```{tip}
Signal your intention to contribute in the issue tracker before starting significant work to coordinate efforts with the maintainers.
```

## Documentation Contents

```{toctree}
:maxdepth: 1
:caption: Documentation

getting_started
concepts
usage
tutorials
api
faq
```

## Indices

* {ref}`genindex` - Index of all documented objects
* {ref}`modindex` - Python module index
* {ref}`search` - Search the documentation

---

**License**: BSD 3-Clause | **GitHub**: [meta-pytorch/forge](https://github.com/meta-pytorch/forge)
