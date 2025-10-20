#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# setup_forge_env.sh - Setup conda environment and install forge with mounting

# Configuration
CONDA_ENV_NAME="forge:41468b33a03eaf2bf5b44517f418028a"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to mount a single workspace to /mnt/wsfuse
mount_workspace() {
    local workspace_url="$1"
    local mount_dir="/mnt/wsfuse"

    if [ -z "$workspace_url" ]; then
        log_error "No workspace URL provided for mounting"
        return 1
    fi

    log_info "Setting up mount directory: $mount_dir"

    # Create the directory if it doesn't exist
    if [ ! -d "$mount_dir" ]; then
        log_info "Creating mount directory: $mount_dir"
        sudo mkdir -p "$mount_dir" || {
            log_error "Failed to create mount directory (may need sudo privileges)"
            log_error "You could alternatively try to unmount with `sudo umount /mnt/wsfuse`"
            return 1
        }
    fi

    # Check if the directory is already mounted
    if mountpoint -q "$mount_dir" 2>/dev/null; then
        log_warn "Directory $mount_dir is already mounted, skipping mount"
        return 0
    fi

    # Check if oilfs command exists
    if ! command -v oilfs >/dev/null 2>&1; then
        log_error "oilfs command not found. Please ensure it's installed and in PATH"
        return 1
    fi

    log_info "Mounting workspace $workspace_url to $mount_dir"

    # Store original LD_LIBRARY_PATH to restore after mounting (similar to Python code)
    original_ld_library_path="${LD_LIBRARY_PATH:-}"

    # Temporarily unset LD_LIBRARY_PATH for mounting
    unset LD_LIBRARY_PATH

    # Mount the workspace
    if sudo oilfs "$workspace_url" "$mount_dir"; then
        log_info "Successfully mounted $workspace_url to $mount_dir"
    else
        log_error "Failed to mount $workspace_url to $mount_dir"
        # Restore original LD_LIBRARY_PATH
        if [ -n "$original_ld_library_path" ]; then
            export LD_LIBRARY_PATH="$original_ld_library_path"
        fi
        return 1
    fi

    # Restore original LD_LIBRARY_PATH
    if [ -n "$original_ld_library_path" ]; then
        export LD_LIBRARY_PATH="$original_ld_library_path"
    fi

    # Verify mount was successful
    if [ -d "$mount_dir/huggingface_models" ]; then
        log_info "Mount verification successful - found expected directory structure"
    else
        log_warn "Mount verification: Expected directory structure not found, but mount appears successful"
    fi

    return 0
}

# Check if required environment variables are set
if [ -z "$USER" ]; then
    log_error "USER environment variable is not set"
    exit 1
fi

# Define paths
FBSOURCE_PATH="/data/users/$USER/fbsource"
CONDA_SCRIPT_PATH="$FBSOURCE_PATH/genai/xlformers/dev/xl_conda.sh"

# Workspace URL for mounting
WORKSPACE_URL="ws://ws.ai.pci0ai/genai_fair_llm"

log_info "Starting forge environment setup for user: $USER"

# Step 1: Mount workspace (do this early in case other steps need the mounted files)
log_info "Step 1: Mounting workspace..."
mount_workspace "$WORKSPACE_URL"
if [ $? -ne 0 ]; then
    log_warn "Failed to mount workspace, continuing with setup..."
    log_warn "Some functionality may not be available without the mounted workspace"
fi

# Step 2: Check if conda script exists and source it
log_info "Step 2: Activating conda environment..."
if [ ! -f "$CONDA_SCRIPT_PATH" ]; then
    log_error "Conda script not found at: $CONDA_SCRIPT_PATH"
    log_error "Please ensure fbsource is properly set up"
    exit 1
fi

log_info "Sourcing conda script: $CONDA_SCRIPT_PATH"
source "$CONDA_SCRIPT_PATH" activate "$CONDA_ENV_NAME"

if [ $? -ne 0 ]; then
    log_error "Failed to activate conda environment $CONDA_ENV_NAME"
    exit 1
fi

log_info "Conda environment activated successfully"


# Step 3: Install torchtitan
log_info "Step 3: Installing torchtitan..."

# Source versions.sh to get the pinned commit
VERSIONS_FILE="assets/versions.sh"
if [ -f "$VERSIONS_FILE" ]; then
    log_info "Sourcing version information from: $VERSIONS_FILE"
    source "$VERSIONS_FILE"

    if [ -n "$TORCHTITAN_COMMIT" ]; then
        log_info "Installing torchtitan from commit: $TORCHTITAN_COMMIT"
        pip uninstall -y torchtitan
        pip install "git+https://github.com/pytorch/torchtitan.git@$TORCHTITAN_COMMIT"

        if [ $? -eq 0 ]; then
            log_info "Torchtitan installed successfully"
        else
            log_error "Failed to install torchtitan"
            exit 1
        fi
    else
        log_error "TORCHTITAN_COMMIT not found in versions.sh"
        exit 1
    fi
else
    log_error "versions.sh not found at: $VERSIONS_FILE"
    log_error "Cannot proceed without version information"
    exit 1
fi

# Step 3.5: Apply monarch torch import hack
log_info "Step 3.5: Applying monarch torch import hack..."

MONARCH_INIT="$CONDA_PREFIX/lib/python3.10/site-packages/monarch/__init__.py"
if [ -f "$MONARCH_INIT" ]; then
    # Check if we already applied the hack
    if grep -q "^import torch  # Injected by forge setup" "$MONARCH_INIT"; then
        log_info "Monarch torch import hack already applied, skipping"
    else
        log_info "Injecting 'import torch' into monarch/__init__.py"

        # Create a backup
        cp "$MONARCH_INIT" "$MONARCH_INIT.bak"

        # Use sed to inject 'import torch' before the "# Import before monarch" comment
        # We add it right after "from typing import TYPE_CHECKING" and before the comment
        sed -i '/^from typing import TYPE_CHECKING$/a\
\
# Torch must be imported before monarch (injected by forge setup)\
import torch  # Injected by forge setup' "$MONARCH_INIT"

        if [ $? -eq 0 ]; then
            log_info "Successfully injected torch import into monarch/__init__.py"
        else
            log_error "Failed to inject torch import, restoring backup"
            mv "$MONARCH_INIT.bak" "$MONARCH_INIT"
            exit 1
        fi
    fi
else
    log_warn "monarch/__init__.py not found at: $MONARCH_INIT"
    log_warn "Skipping monarch torch import hack (monarch may not be installed yet)"
fi

# Step 4: Check for existing build directory and warn user
log_info "Step 4: Checking for existing build directory..."
if [ -d "build" ]; then
    log_warn "Detected existing build/ directory at: $(pwd)/build"
    log_warn "This directory may contain artifacts from a previous pip installation"
    log_warn "that could interfere with the current installation."
    log_warn "If you encounter issues, manually remove it with: rm -rf build"
    echo ""
    read -p "$(echo -e ${YELLOW}Do you want to continue anyway? [y/N]:${NC} )" -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log_info "Installation cancelled by user"
        log_info "You can manually remove the build/ directory with: rm -rf build"
        exit 0
    fi
    log_warn "Continuing with existing build/ directory. Things might go wrong!"
fi

# Step 5: Install forge package
log_info "Step 5: Installing forge package..."
pip install --no-deps --force-reinstall .
if [ $? -ne 0 ]; then
    log_error "Failed to install forge package"
    exit 1
fi
log_info "Forge package installed successfully"

log_info "Environment activation completed"

# Final verification
log_info "Setup completed successfully!"

# Check mount status
if mountpoint -q "/mnt/wsfuse" 2>/dev/null; then
    log_info "Workspace mount: ✓ Active at /mnt/wsfuse"
else
    log_warn "Workspace mount: ✗ Not mounted"
fi

# Check current environment
if command -v conda >/dev/null 2>&1 && conda info --envs >/dev/null 2>&1; then
    CURRENT_ENV=$(conda info --show-active-prefix 2>/dev/null | sed 's/.*\///' || echo "unknown")
    log_info "Current conda environment: $CURRENT_ENV"
else
    log_info "Current environment: Using xl_conda.sh managed environment"
fi

log_info "Current directory: $(pwd)"
log_info "Python location: $(which python)"

# Show installed packages
log_info "Key installed packages:"
pip list | grep -E "(forge|monarch)" || log_warn "No forge/monarch packages found in pip list"

log_info "Environment setup complete! You can now run your scripts."
log_info "Mounted workspace available at: /mnt/wsfuse"

log_info "Unsetting CUDA_HOME and overwriting the LD_LIBRARY_PATH"
unset CUDA_HOME
export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib

# Step 6: Ask user to test
echo ""
log_info "Installation completed successfully!"
echo ""
log_info "Test that this is working locally with:"
log_info "python -m apps.grpo.main --config=apps/grpo/qwen3_1_7b.yaml"
