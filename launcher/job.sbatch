# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#!/bin/bash
#SBATCH --job-name=forge
#SBATCH --output=slogs/forge.out
#SBATCH --error=slogs/forge.err
#SBATCH --partition=h100-low        # or h100-high / h100-prod / all
#SBATCH --nodes=1                   # 1 node
#SBATCH --ntasks=1                  # 1 task (process)
#SBATCH --gres=gpu:8                # request 8 GPUs
#SBATCH --time=01:00:00             # walltime hh:mm:ss

unset SLURM_MEM_PER_CPU SLURM_MEM_PER_GPU SLURM_MEM_PER_NODE
echo "Running on $SLURM_JOB_NODELIST"
python -m apps.grpo.main --config=apps/grpo/multihost.yaml
