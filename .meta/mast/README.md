# Forge MAST Environment Setup

A simple setup script to automatically configure your environment for running Forge with MAST jobs.
This only applies to Meta internal users.

## Quick Start

⚠️ Important Note: the setup script will clone the forge repository under "/data/users/$USER".

### 1. Run the Setup Script

The `env_setup.sh` script will automatically:
- ✅ Activate and configure the required conda environment
- ✅ Clone/update the Forge repository
- ✅ Install Forge package dependencies
- ✅ Mount the required oilfs workspace to `/mnt/wsfuse`
- ✅ Configure your environment for MAST job submission

```bash
# Make the script executable
chmod +x .meta/mast/env_setup.sh

# Run the setup
source .meta/mast/env_setup.sh

```

### 2. Submit MAST job

Use the launch script to submit a MAST job:

```bash
# Make the launch script executable (first time only)
chmod +x .meta/mast/launch.sh

# Launch a job with your desired config
./.meta/mast/launch.sh .meta/mast/qwen3_1_7b_mast.yaml
```

The launch script will automatically:
- Navigate to the forge root directory
- Reinstall the forge package with your latest changes
- Set the correct PYTHONPATH
- Launch the MAST job with the specified config

You can run it from anywhere, and it will figure out the correct paths.


## Managing HuggingFace Models in MAST

### The Problem: No Internet Access

MAST compute nodes cannot access the internet, which means they cannot download models directly from HuggingFace. To work around this, we store all HuggingFace models and cache data on OilFS at `/mnt/wsfuse/teamforge/hf`, which is accessible from MAST.

### Solution: Two-Step Process

You need to perform both steps below to ensure models work correctly in MAST:

#### 1. Download Model Weights to OilFS

First, download the model weights directly to the OilFS path. This should be done from a machine with internet access (like your devserver):

```bash
# Set HF_HOME to the OilFS path
export HF_HOME=/mnt/wsfuse/teamforge/hf

# Download the model (replace with your desired model)
huggingface-cli download Qwen/Qwen3-8B --local-dir /mnt/wsfuse/teamforge/hf_artifacts/qwen3_8b
```

#### 2. Hydrate the HuggingFace Cache

After downloading the weights, you need to hydrate the HuggingFace cache so that the transformers library can find the model metadata:

```bash
# Set HF_HOME to the OilFS path
export HF_HOME=/mnt/wsfuse/teamforge/hf

# Hydrate the cache for the model
python .meta/mast/hydrate_cache.py --model-id Qwen/Qwen3-8B
```

This ensures that when MAST runs with `HF_HUB_OFFLINE=1`, the transformers library can locate all necessary files from the cache.

### Directory Structure

Both cache and model files are stored under:
- **Cache**: `/mnt/wsfuse/teamforge/hf` (set via `HF_HOME`)
- **Model weights**: `/mnt/wsfuse/teamforge/hf/<model_name>`

Make sure your MAST config files point to the correct paths in `hf_artifacts`.
