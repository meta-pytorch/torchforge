<div align="center">

# forge

#### A PyTorch native platform for post-training generative AI models

## Overview

## Installation

```bash
pip install uv
git clone https://github.com/pytorch-labs/forge
cd forge
uv sync
```

## Quick Start

To run SFT for Llama3 8B, run

```bash
uv run forge download meta-llama/Meta-Llama-3.1-8B-Instruct --output-dir /tmp/Meta-Llama-3.1-8B-Instruct --ignore-patterns "original/consolidated.00.pth"
uv run forge run --nproc_per_node 2 apps/sft/main.py --config apps/sft/llama3_8b.yaml
```

### Citation

## License

Source code is made available under a [BSD 3 license](./LICENSE), however you may have other legal obligations that govern your use of other content linked in this repository, such as the license or terms of service for third-party data and models.
