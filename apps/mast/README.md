# Forge MAST Environment Setup

A simple setup script to automatically configure your environment for running Forge with MAST jobs.

## Quick Start

### 1. Run the Setup Script

The `env_setup.sh` script will automatically:
- ✅ Activate the required conda environment (`forge-8448524`)
- ✅ Clone/update the Forge repository
- ✅ Install Forge package dependencies
- ✅ Mount the required oilfs workspace to `/mnt/wsfuse`
- ✅ Configure your environment for MAST job submission

```bash
# Make the script executable
chmod +x env_setup.sh

# Run the setup
./apps/mast/env_setup.sh

```

### 2. Submit MAST job

```
pip install --force-reinstall --no-deps . && python -m apps.mast.main --config apps/mast/qwen3_1_7b_mast.yaml
```

⚠️ Important Note: `pip install --force-reinstall --no-deps .` is required every time you make a change to the local codebase. This ensures your latest changes are installed before job submission.
