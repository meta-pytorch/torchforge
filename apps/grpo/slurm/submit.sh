#!/bin/bash
CONFIG_NAME="${1:-qwen3_32b}"

sbatch --job-name="${CONFIG_NAME}" \
       --export=ALL,CONFIG_NAME="${CONFIG_NAME}" \
       apps/grpo/slurm/submit_grpo.sh


# Usage:
# ./apps/grpo/slurm/submit.sh qwen3_8b
# ./apps/grpo/slurm/submit.sh qwen3_32b
# ./apps/grpo/slurm/submit.sh qwen3_30b_a3b
