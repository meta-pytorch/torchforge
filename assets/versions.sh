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
VLLM_XPU_VERSION="v0.13.0"
# IPEX wheels shipped with vLLM has hard python version requirement
IPEX_PYTHON_VERSION="3.12"
TORCHSTORE_BRANCH="no-monarch-2026.01.05"
# ROCm/XPU builds these from source (no ROCm/XPU wheels); CUDA uses pyproject pins.
TORCHTITAN_VERSION="v0.2.0"
TORCHTITAN_XPU_COMMIT="e61f2cce4fd9c54d314ff0a2dabe035b80a5d49c"
MONARCH_VERSION="v0.2.0"
