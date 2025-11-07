# <img width="35" height="35" alt="image" src="https://github.com/user-attachments/assets/2700a971-e5d6-4036-b03f-2f89c9791609" /> torchforge

#### A PyTorch-native agentic RL library that lets you focus on algorithms—not infra.
[![GPU Tests](https://github.com/meta-pytorch/forge/actions/workflows/gpu_test.yaml/badge.svg?branch=main)](https://github.com/meta-pytorch/forge/actions/workflows/gpu_test.yaml?query=branch%3Amain)
[![Documentation](https://img.shields.io/badge/Docs-meta--pytorch.org-blue?style=flat&logo=readthedocs&logoColor=white)](https://meta-pytorch.org/torchforge/)
[![Discord](https://img.shields.io/badge/Discord-OpenEnv-7289da?style=flat&logo=discord&logoColor=white)](https://discord.gg/YsTYBh6PD9)

## Overview
The primary purpose of the torchforge ecosystem is to delineate infra concerns from model concerns thereby making RL experimentation easier. torchforge delivers this by providing clear RL abstractions and one scalable implementation of these abstractions. When you need fine-grained control over placement, fault handling/redirecting training loads during a run, or communication patterns, the primitives are there. When you don’t, you can focus purely on your RL algorithm.

Key features:
- Usability for rapid research (isolating the RL loop from infrastructure)
- Hackability for power users (all parts of the RL loop can be easily modified without interacting with infrastructure)
- Scalability (ability to shift between async and synchronous training and across thousands of GPUs)

> ⚠️ **Early Development Warning** torchforge is currently in an experimental
> stage. You should expect bugs, incomplete features, and APIs that may change
> in future versions. The project welcomes bugfixes, but to make sure things are
> well coordinated you should discuss any significant change before starting the
> work. It's recommended that you signal your intention to contribute in the
> issue tracker, either by filing a new issue or by claiming an existing one.

## 📖 Documentation

View torchforge's hosted documentation: https://meta-pytorch.org/torchforge.

## Tutorials

You can also find our notebook tutorials (coming soon)

## Installation

torchforge requires PyTorch 2.9.0 with [Monarch](https://github.com/meta-pytorch/monarch), [vLLM](https://github.com/vllm-project/vllm), and [torchtitan](https://github.com/pytorch/torchtitan).

Install torchforge with:

```bash
conda create -n forge python=3.12
conda activate forge
./scripts/install.sh
```

The install script installs system dependencies along with torchforge. Note that this install script uses [DNF](https://docs.fedoraproject.org/en-US/quick-docs/dnf/), but could be easily extended to other Linux OS.

Optional: By default, the packages installation uses conda. If you want to install system packages on the target machine instead of conda, you can pass the `--use-sudo` flag to the installation script: `./scripts/install.sh --use-sudo`.

> **Note:** We are actively working on enabling pure `uv` installation. Currently, Conda is the recommended approach. `uv` support is not fully working at the moment but is being tracked in [issue #494](https://github.com/meta-pytorch/torchforge/issues/494).

After install, you can run the following command and should see output confirming GRPO training is running (you need a minimum 3 GPU devices):

```
python -m apps.grpo.main --config apps/grpo/qwen3_1_7b.yaml
```

## Quick Start

To run SFT on a Llama3 8B model, run

```bash
python -m apps.sft.main --config apps/sft/llama3_8b.yaml
```

### Citation

## License

Source code is made available under a [BSD 3 license](./LICENSE), however you may have other legal obligations that govern your use of other content linked in this repository, such as the license or terms of service for third-party data and models.
