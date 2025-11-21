#!/bin/bash
#SBATCH --job-name=grpo-qwen3-32b
#SBATCH --qos=h200_agentic-models_high
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

export TORCHSTORE_RDMA_ENABLED=0

cd /storage/home/daniellepintz/torchforge

srun python -m apps.grpo.main --config apps/grpo/qwen3_32b.yaml
