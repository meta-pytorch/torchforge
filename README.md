# <img width="35" height="35" alt="image" src="https://github.com/user-attachments/assets/2700a971-e5d6-4036-b03f-2f89c9791609" /> Forge


#### A PyTorch native agentic library for RL post-training and agentic development

## Overview
Forge was built with one core principle in mind: researchers should write algorithms, not infrastructure. Forge introduces a “service”-centric architecture that provides the right abstractions for distributed complexity. When you need fine-grained control over placement, fault handling or communication patterns, the primitives are there. When you don’t, you can focus purely on your RL algorithm.

Key features:
- Usability for rapid research (isolating the RL loop from infrastructure)
- Hackability for power users (all parts of the RL loop can be easily modified without interacting with infrastructure)
- Scalability (ability so shift between async and syncronous training and across thousands of GPUs)

> ⚠️ **Early Development Warning** Forge is currently in an experimental
> stage. You should expect bugs, incomplete features, and APIs that may change
> in future versions. The project welcomes bugfixes, but to make sure things are
> well coordinated you should discuss any significant change before starting the
> work. It's recommended that you signal your intention to contribute in the
> issue tracker, either by filing a new issue or by claiming an existing one.

## 📖 Documentation

View Forge's hosted documentation [at this link](https://meta-pytorch.org/forge/).

## Installation

### Basic

Forge requires the latest PyTorch nightly with Monarch, vLLM, and torchtitan. For convenience,
we have pre-packaged these dependencies as wheels in assets/wheels. (Note that the basic install script
uses [DNF](https://docs.fedoraproject.org/en-US/quick-docs/dnf/), but could be easily extended to other Linux OS.)

Forge requires the Github CLI (gh) to download a compatible vLLM package. See [here](https://github.com/cli/cli#installation) for gh install instructions before continuting. Please login to gh with your Github account before continuing with `gh auth login`. You may use either https or ssh as the protocol for authentication.

```bash
conda create -n forge python=3.10
conda activate forge
./scripts/install.sh
```

Optional: By default, the packages installation uses conda. If user wants to install system packages on the target machine instead of conda, they can pass the `--use-sudo` to the installation script: `./script/install.sh --use-sudo`.

After install, you can run the following command and should see output confirming GRPO training is running (you need a minimum 3 GPU devices).

```
python -m apps.grpo.main  --config apps/grpo/qwen3_1_7b.yaml
```

If you need to re-build the wheels for whatever reason, you can do so with:
```bash
./scripts/build_wheels.sh
```

For your information, since the vLLM wheel is too large for GitHub, we uploaded it as a release in the `install.sh` script:
```
$ gh release create v0.0.0 assets/wheels/vllm-*.whl --title "Forge Wheels v0.0.0"
```

### Meta Internal Build (Alternative Route)

1. Build uv package

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/pytorch-labs/forge
cd forge
uv sync --all-extras
source .venv/bin/activate
```

2. Setup CUDA on local machine

```bash
# feature install if you don't have /user/local/cuda-12.8
feature install --persist cuda_12_9

# add env variables
export CUDA_VERSION=12.9
export NVCC=/usr/local/cuda-$CUDA_VERSION/bin/nvcc
export CUDA_NVCC_EXECUTABLE=/usr/local/cuda-$CUDA_VERSION/bin/nvcc
export CUDA_HOME=/usr/local/cuda-$CUDA_VERSION
export PATH="$CUDA_HOME/bin:$PATH"
export CUDA_INCLUDE_DIRS=$CUDA_HOME/include
export CUDA_CUDART_LIBRARY=$CUDA_HOME/lib64/libcudart.so
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

3. Build vllm from source

```bash
git clone https://github.com/vllm-project/vllm.git --branch v0.10.0
cd vllm
python use_existing_torch.py
uv pip install -r requirements/build.txt
uv pip install --no-build-isolation -e .
```

> [!WARNING]
> If you add packages to the pyproject.toml, use `uv sync --inexact` so it doesn't remove Monarch and vLLM

## Quick Start

To run SFT for Llama3 8B, run

```bash
uv run forge download meta-llama/Meta-Llama-3.1-8B-Instruct --output-dir /tmp/Meta-Llama-3.1-8B-Instruct --ignore-patterns "original/consolidated.00.pth"
uv run forge run --nproc_per_node 2 apps/sft/main.py --config apps/sft/llama3_8b.yaml
```

### Citation

## License

Source code is made available under a [BSD 3 license](./LICENSE), however you may have other legal obligations that govern your use of other content linked in this repository, such as the license or terms of service for third-party data and models.
