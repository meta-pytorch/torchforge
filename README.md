# forge

#### A PyTorch native platform for post-training generative AI models

## Overview

## Installation

### Basic (Broken)

```bash
pip install uv
git clone https://github.com/pytorch-labs/forge
cd forge
uv sync

# Or for dev install:
uv sync --all-extras
```


### Internal Machine

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
feature install --persist cuda_12_8

# add env variables
export CUDA_VERSION=12.8
export NVCC=/usr/local/cuda-${CUDA_VERSION}/bin/nvcc
export CUDA_NVCC_EXECUTABLE=/usr/local/cuda-${CUDA_VERSION}/bin/nvcc
export CUDA_HOME=/usr/local/cuda-${CUDA_VERSION}
export PATH="${CUDA_HOME}/bin:$PATH"
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
uv pip install --no-build-isolation -e .bash
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
