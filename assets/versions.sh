# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Version Configuration for Forge Wheel Building
# This file contains all pinned versions and commits for dependencies

# Stable versions of upstream libraries for OSS repo
PYTORCH_VERSION="2.9.0"
# ROCm/XPU builds vLLM from source (no prebuilt ROCm/XPU wheels available)
VLLM_ROCM_VERSION="v0.10.0"
VLLM_XPU_VERSION="v0.17.0"
# PyTorch XPU version (vLLM v0.16+ dropped IPEX in favour of native XPU support)
PYTORCH_XPU_VERSION="2.10.0"
# vllm-xpu-kernels wheels only ship for Python 3.12
XPU_PYTHON_VERSION="3.12"
TORCHSTORE_BRANCH="no-monarch-2026.01.05"
# ROCm/XPU builds these from source (no ROCm/XPU wheels); CUDA uses pyproject pins.
TORCHTITAN_VERSION="v0.2.0"
MONARCH_VERSION="v0.2.0"
