#!/bin/bash

# setup_forge_env.sh - Setup conda environment and install forge
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
MONARCH_DIR="$HOME/monarch_no_torch_latest"

log_info "Starting forge environment setup for user: $USER"

# Step 1: Check if conda script exists and source it
log_info "Step 1: Activating conda environment..."
if [ ! -f "$CONDA_SCRIPT_PATH" ]; then
    log_error "Conda script not found at: $CONDA_SCRIPT_PATH"
    log_error "Please ensure fbsource is properly set up"
    exit 1
fi

log_info "Sourcing conda script: $CONDA_SCRIPT_PATH"
source "$CONDA_SCRIPT_PATH" activate forge:8448524

if [ $? -ne 0 ]; then
    log_error "Failed to activate conda environment forge:8448524"
    exit 1
fi

log_info "Conda environment activated successfully"

# Step 2: Create and navigate to forge base directory
log_info "Step 2: Setting up forge directory..."
if [ ! -d "$FORGE_BASE_DIR" ]; then
    log_info "Creating forge base directory: $FORGE_BASE_DIR"
    mkdir -p "$FORGE_BASE_DIR"
fi

cd "$FORGE_BASE_DIR"
log_info "Changed to directory: $(pwd)"

# Step 3: Clone or update forge repository
log_info "Step 3: Setting up forge git repository..."
if [ -d "$FORGE_REPO_DIR" ]; then
    log_warn "Forge repository already exists at: $FORGE_REPO_DIR"
    cd "$FORGE_REPO_DIR"

    # Check if it's a valid git repository
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

# Step 4: Install forge package
log_info "Step 4: Installing forge package..."
pip install --no-deps --force-reinstall .
if [ $? -ne 0 ]; then
    log_error "Failed to install forge package"
    exit 1
fi
log_info "Forge package installed successfully"

# Step 5: Navigate to monarch directory
log_info "Step 5: Setting up monarch directory..."
if [ ! -d "$MONARCH_DIR" ]; then
    log_info "Creating monarch directory: $MONARCH_DIR"
    mkdir -p "$MONARCH_DIR"
fi

cd "$MONARCH_DIR"
log_info "Changed to directory: $(pwd)"

# Step 6: Fetch monarch package
log_info "Step 6: Fetching monarch package..."
# TOD): remove hardcoded apcakge version
fbpkg fetch monarch_no_torch:23
if [ $? -ne 0 ]; then
    log_error "Failed to fetch monarch_no_torch:23"
    log_error "Please ensure fbpkg is properly configured"
    exit 1
fi
log_info "Monarch package fetched successfully"

# Step 7: Install monarch wheel
log_info "Step 7: Installing monarch wheel..."
WHEEL_FILE="monarch-0.0.0-py3.10-none-any.whl"
if [ ! -f "$WHEEL_FILE" ]; then
    log_error "Wheel file not found: $WHEEL_FILE"
    log_error "Available files in directory:"
    ls -la *.whl 2>/dev/null || log_error "No wheel files found"
    exit 1
fi

pip install --force-reinstall "$WHEEL_FILE"
if [ $? -ne 0 ]; then
    log_error "Failed to install monarch wheel"
    exit 1
fi
log_info "Monarch wheel installed successfully"

# Step 8: Ask user to deactivate and activate conda env conda environment
echo ""
log_info "Installation completed successfully!"
echo ""
log_info "Re-activate the conda environment to make the changes take effect:"
log_info "conda deactivate && conda activate forge-8448524"
