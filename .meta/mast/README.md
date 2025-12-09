# Forge MAST Environment Setup

A simple setup script to automatically configure your environment for running Forge with MAST jobs.
This only applies to Meta internal users.

## Quick Start

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


## How MAST Launcher Works

The MAST launcher uses a two-stage architecture to run training jobs:

### Stage 1: Detached Mode (Local Machine)

When you run `./.meta/mast/launch.sh`, the `main.py` script starts in **detached mode**:

1. The launcher creates a MAST job with all the worker roles (GPU hosts)
2. It also creates a special **client role** - a CPU-only role that will run inside MAST
3. The client role's entrypoint is set to `client_bootstrap.sh`
4. All CLI arguments you pass are forwarded to the client role

At this point, the job is submitted to MAST and your local script exits. Everything now runs in the cluster.

### Stage 2: Remote Mode (Inside MAST)

The `client_bootstrap.sh` script runs inside the MAST client role and:

1. Calls `main.py` again, but now with `--mode=remote`
2. In **remote mode**, the script:
   - Mounts the OilFS workspace
   - Initializes the provisioner to connect to worker roles
   - Runs the actual training workload (e.g., GRPO)

This architecture allows the entire training workflow to run inside MAST without requiring a persistent connection from your local machine.

### Key Files

- **`main.py`**: Entry point that handles both detached and remote modes
- **`client_bootstrap.sh`**: Entrypoint for the client role in MAST
- **`launcher.py`**: Creates the MAST job specification and handles role configuration


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
hf download Qwen/Qwen3-8B --local-dir /mnt/wsfuse/teamforge/hf/qwen3_8b
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


#### Weights & Biases
If you are part of the torchforge team on WandB, then WandB will work out of the box; the link can be found in the MAST logs. If you are not part of the torchforge team on WandB, then you will need to set the "WANDB_API_KEY" environment variable to your WandB API key.
