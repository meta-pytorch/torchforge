# Welcome to TorchForge documentation!

**TorchForge** is a PyTorch-native platform specifically designed
for post-training generative AI models.

Key Features
------------

* **Post-Training Focus**: Specializes in techniques
  like Supervised Fine-Tuning (SFT) and Group Relative Policy Optimization (GRPO)
* **PyTorch Integration**: Built natively on PyTorch with
  dependencies on [PyTorch nightly](https://pytorch.org/get-started/locally/),
  [Monarch](https://meta-pytorch.org/monarch), [vLLM](https://docs.vllm.ai/en/latest/),
  and [TorchTitan](https://github.com/pytorch/torchtitan).
* **Multi-GPU Support**: Designed for distributed training
  with minimum 3 GPU requirement for GRPO training
* **Model Support**: Includes pre-configured setups for popular models
  like Llama3 8B and Qwen3.1 7B

```{toctree}
:maxdepth: 1
:caption: Contents:

getting_started
concepts
tutorials
api
```

## Indices and tables

* {ref}`genindex`
* {ref}`modindex`
* {ref}`search`
