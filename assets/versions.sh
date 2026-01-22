# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Version Configuration for Forge Wheel Building
# This file contains all pinned versions and commits for dependencies

# Stable versions of upstream libraries for OSS repo
PYTORCH_VERSION="2.9.0"
VLLM_VERSION="v0.10.0"
TORCHSTORE_BRANCH="no-monarch-2026.01.05"
# ROCm install builds these from source (no ROCm wheels); CUDA uses pyproject pins.
TORCHTITAN_VERSION="v0.2.0"
MONARCH_VERSION="v0.2.0"
