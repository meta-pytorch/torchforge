#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#SBATCH --qos=h200_capabilities_shared
#SBATCH --account=agentic-models
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=128
#SBATCH --mem=500G
#SBATCH --time=72:00:00

echo "Starting GRPO training job"

eval "$(conda shell.bash hook)"

conda activate forge

export TORCH_COMPILE_DISABLE=1
unset SLURM_MEM_PER_CPU SLURM_MEM_PER_GPU SLURM_MEM_PER_NODE
export TORCHSTORE_RDMA_ENABLED=0

cd /storage/home/$USER/torchforge

srun python -m apps.grpo.main --config apps/grpo/slurm/${CONFIG_NAME}.yaml
