<div align="center">

# forge

#### A PyTorch native platform for post-training generative AI models

## Overview

## Installation

torchforge depends on torchtitan, which should first be installed from source.

```bash
pip3 install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu126 --force-reinstall
[For AMD GPU] pip3 install --pre torch --index-url https://download.pytorch.org/whl/nightly/rocm6.3 --force-reinstall
git clone https://github.com/pytorch/torchtitan
pip install -e ./torchtitan
git clone https://github.com/pytorch-labs/forge
cd forge
pip install -r requirements.txt
```

## Quick Start

To run SFT for Llama3 8B, run

```bash
forge download meta-llama/Meta-Llama-3.1-8B-Instruct --output-dir /tmp/Meta-Llama-3.1-8B-Instruct --ignore-patterns "original/consolidated.00.pth"
forge run --nproc_per_node 2 apps/sft/main.py --config apps/sft/llama3_8b.yaml
```

### Citation

## License

Source code is made available under a [BSD 3 license](./LICENSE), however you may have other legal obligations that govern your use of other content linked in this repository, such as the license or terms of service for third-party data and models.
