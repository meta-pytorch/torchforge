#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

CONFIG_NAME="${1}"

sbatch --job-name="${CONFIG_NAME}_controller" \
       --export=ALL,CONFIG_NAME="${CONFIG_NAME}" \
       experimental/slurm/submit_grpo.sh


# Usage:
# ./experimental/slurm/submit.sh qwen3_8b
# ./experimental/slurm/submit.sh qwen3_32b
# ./experimental/slurm/submit.sh qwen3_30b_a3b
