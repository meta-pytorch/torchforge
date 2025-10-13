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
./.meta/mast/env_setup.sh

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
