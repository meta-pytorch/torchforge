#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Bootstrap script for the MAST client role
# This script sets up the environment and launches the client training script

set -eEx

LIBCUDA="/usr/local/fbcode/platform010/lib/libcuda.so"
if [ -f "$LIBCUDA" ]; then
    export LIBCUDA_DIR="${LIBCUDA%/*}"
    export TRITON_LIBCUDA_PATH="$LIBCUDA_DIR"
    export LD_PRELOAD="$LIBCUDA:/usr/local/fbcode/platform010/lib/libnvidia-ml.so${PRELOAD_PATH:+:$PRELOAD_PATH}"
fi

# Also preload put path to torch libs as for monarch dev workflow we dont
# install it into the env so we need to make sure the binaries can find
# libtorch and friends on mast and the rpaths set during dev install will
# be wrong on mast.
export LD_LIBRARY_PATH="${CONDA_DIR}/lib:${CONDA_DIR}/lib/python3.10/site-packages/torch/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$TORCHX_RUN_PYTHONPATH"

# shellcheck disable=SC1091
if [ -n "$CONDA_PREFIX" ]; then
    echo "A conda environment is already activated: $CONDA_DEFAULT_ENV"
else
    # Disable command printing to avoid log spew.
    set +x
    source "${CONDA_DIR}/bin/activate"
    # Re-enable command printing after conda activation.
    set -x
fi

if [ -z "$WORKSPACE_DIR" ] || [ ! -d "$WORKSPACE_DIR" ]; then
    WORKSPACE_DIR="$CONDA_PREFIX"
fi

cd "$WORKSPACE_DIR/forge"

export WANDB_MODE=offline
export HF_HUB_OFFLINE=1
export MONARCH_HOST_MESH_V1_REMOVE_ME_BEFORE_RELEASE=1
export TORCHSTORE_RDMA_ENABLED=1
export HF_HOME=/mnt/wsfuse/teamforge/huggingface

# Execute the client training script with all passed arguments
exec python -X faulthandler .meta/mast/main.py "$@"
