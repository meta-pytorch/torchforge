#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# CUDA environment activation script for Pixi
# This script is automatically sourced when the pixi environment is activated

# CUDA environment variables
export CUDA_VERSION=12.8
export NVCC=/usr/local/cuda-${CUDA_VERSION}/bin/nvcc
export CUDA_NVCC_EXECUTABLE=/usr/local/cuda-${CUDA_VERSION}/bin/nvcc
export CUDA_HOME=/usr/local/cuda-${CUDA_VERSION}
export PATH="${CUDA_HOME}/bin:$PATH"
export CUDA_INCLUDE_DIRS=$CUDA_HOME/include
export CUDA_CUDART_LIBRARY=$CUDA_HOME/lib64/libcudart.so

# DO NOT set LD_LIBRARY_PATH globally - it breaks system tools
# Instead, we use python wrapper functions below to set it only for Python processes

# Define python wrappers that set LD_LIBRARY_PATH only for the launched process
# Priority: system CUDA driver (/usr/lib64) > CUDA toolkit > conda/pixi libs
# Use system CUDA driver (newer) instead of compat (outdated stub that may be incompatible)
python()  {
    LD_LIBRARY_PATH="/usr/lib64:/usr/local/cuda-12.8/lib64:${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    command python "$@"
}
python3() {
    LD_LIBRARY_PATH="/usr/lib64:/usr/local/cuda-12.8/lib64:${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    command python3 "$@"
}

# Export functions to subshells when possible (best-effort, shell-dependent)
if [ -n "${BASH_VERSION:-}" ]; then
    export -f python python3 2>/dev/null || true
elif [ -n "${ZSH_VERSION:-}" ]; then
    typeset -fx python python3 >/dev/null 2>&1 || true
fi
