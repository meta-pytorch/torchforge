#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# setup_forge_env.sh - Setup conda environment and install forge with mounting
set -e  # Exit on any error

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
FORGE_BASE_DIR="/data/users/$USER"
FORGE_REPO_DIR="$FORGE_BASE_DIR/forge"

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
source "$CONDA_SCRIPT_PATH" activate forge:e146614

if [ $? -ne 0 ]; then
    log_error "Failed to activate conda environment forge-e146614"
    exit 1
fi

log_info "Conda environment activated successfully"

# Step 3: Create and navigate to forge base directory
log_info "Step 3: Setting up forge directory..."
if [ ! -d "$FORGE_BASE_DIR" ]; then
    log_info "Creating forge base directory: $FORGE_BASE_DIR"
    mkdir -p "$FORGE_BASE_DIR"
fi

cd "$FORGE_BASE_DIR"
log_info "Changed to directory: $(pwd)"

# Step 4: Clone or update forge repository
log_info "Step 4: Setting up forge git repository..."
if [ -d "$FORGE_REPO_DIR" ]; then
    log_warn "Forge repository already exists at: $FORGE_REPO_DIR"
    cd "$FORGE_REPO_DIR"

    if [ -d ".git" ]; then
        log_info "Updating existing repository..."
        git fetch origin
        if [ $? -eq 0 ]; then
            log_info "Repository updated successfully"
        else
            log_warn "Failed to fetch updates, continuing with existing code"
        fi
    else
        log_error "Directory exists but is not a git repository"
        log_info "Removing directory and cloning fresh..."
        cd "$FORGE_BASE_DIR"
        rm -rf "$FORGE_REPO_DIR"
        git clone git@github.com:meta-pytorch/forge.git
        if [ $? -ne 0 ]; then
            log_error "Failed to clone forge repository"
            exit 1
        fi
        cd "$FORGE_REPO_DIR"
    fi
else
    log_info "Cloning forge repository..."
    git clone git@github.com:meta-pytorch/forge.git
    if [ $? -ne 0 ]; then
        log_error "Failed to clone forge repository"
        log_error "Please ensure:"
        log_error "1. You have SSH access to github.com"
        log_error "2. Your SSH key is added to GitHub"
        log_error "3. You have access to meta-pytorch/forge repository"
        exit 1
    fi
    cd "$FORGE_REPO_DIR"
fi

log_info "Current directory: $(pwd)"

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

# Step 6: Ask user to deactivate and activate conda env conda environment
echo ""
log_info "Installation completed successfully!"
echo ""
log_info "Re-activate the conda environment to make the changes take effect:"
log_info "conda deactivate && conda activate forge-e146614"
