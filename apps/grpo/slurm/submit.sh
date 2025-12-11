#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

CONFIG_NAME="${1}"

sbatch --job-name="${CONFIG_NAME}" \
       --export=ALL,CONFIG_NAME="${CONFIG_NAME}" \
       apps/grpo/slurm/submit_grpo.sh


# Usage:
# ./apps/grpo/slurm/submit.sh qwen3_8b
# ./apps/grpo/slurm/submit.sh qwen3_32b
# ./apps/grpo/slurm/submit.sh qwen3_30b_a3b
