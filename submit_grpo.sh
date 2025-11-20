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

# Initialize conda
eval "$(conda shell.bash hook)"

# Activate the conda environment (replace 'forge' with your actual environment name if different)
conda activate forge

# # Option 1: Set wandb API key (replace with your actual API key)
# export "WANDB_API_KEY=4cf092866223040751bacd9b149cfd87304d19a2"

# export WANDB_MODE=offline
# export WANDB_DIR="/mnt/wsfuse/teamforge/wandb/$(cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w 5 | head -n 1)"
# mkdir -p "$WANDB_DIR"

# Change to the torchforge directory
cd /storage/home/daniellepintz/torchforge

# Run the GRPO training
srun python -m apps.grpo.main --config apps/grpo/qwen3_8b.yaml
