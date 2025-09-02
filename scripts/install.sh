#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -euo pipefail

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Configuration
PYTORCH_VERSION="2.9.0.dev20250828"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WHEEL_DIR="$SCRIPT_DIR/../assets/wheels"
RELEASE_TAG="v0.0.0"
GITHUB_REPO="meta-pytorch/forge"

# Check conda environment
check_conda_env() {
    if [ -z "${CONDA_DEFAULT_ENV:-}" ]; then
        log_error "Not running in a conda environment"
        log_info "Please create and activate your conda environment first:"
        log_info "  conda create -n forge python=3.10 -y"
        log_info "  conda activate forge"
        exit 1
    fi
    log_info "Installing in conda environment: $CONDA_DEFAULT_ENV"
}

# Check sudo access
check_sudo() {
    if ! sudo -n true 2>/dev/null; then
        log_error "This script requires passwordless sudo access for system packages"
        log_info "Run 'sudo -v' first, or configure passwordless sudo"
        exit 1
    fi
}

# Install required system packages
install_system_packages() {
    log_info "Installing required system packages..."
    sudo dnf install -y libibverbs rdma-core libmlx5 libibverbs-devel rdma-core-devel
    log_info "System packages installed successfully"
}

# Check wheels exist
check_wheels() {
    if [ ! -d "$WHEEL_DIR" ]; then
        log_error "Wheels directory not found: $WHEEL_DIR"
        exit 1
    fi

    local wheel_count=$(ls -1 "$WHEEL_DIR"/*.whl 2>/dev/null | wc -l)
    log_info "Found $wheel_count local wheels"
}

# Download vLLM wheel from GitHub releases
download_vllm_wheel() {
    log_info "Downloading vLLM wheel from GitHub releases..."

    # Check if gh is installed
    if ! command -v gh &> /dev/null; then
        log_error "GitHub CLI (gh) is required to download vLLM wheel"
        log_info "Install it with: sudo dnf install gh"
        log_info "Then run: gh auth login"
        exit 1
    fi

    # Get the vLLM wheel filename from the release
    local vllm_wheel_name
    vllm_wheel_name=$(gh release view "$RELEASE_TAG" --repo "$GITHUB_REPO" --json assets --jq '.assets[] | select(.name | contains("vllm")) | .name' | head -1)

    if [ -z "$vllm_wheel_name" ]; then
        log_error "Could not find vLLM wheel in release $RELEASE_TAG"
        log_info "Make sure you've uploaded the vLLM wheel to the GitHub release"
        exit 1
    fi

    local local_path="$WHEEL_DIR/$vllm_wheel_name"

    if [ -f "$local_path" ]; then
        log_info "vLLM wheel already downloaded: $vllm_wheel_name"
        return 0
    fi

    log_info "Downloading: $vllm_wheel_name"

    # Save current directory and change to wheel directory
    local original_dir=$(pwd)
    cd "$WHEEL_DIR"
    gh release download "$RELEASE_TAG" --repo "$GITHUB_REPO" --pattern "*vllm*"
    local download_result=$?

    # Always return to original directory
    cd "$original_dir"

    if [ $download_result -eq 0 ]; then
        log_info "Successfully downloaded vLLM wheel"
    else
        log_error "Failed to download vLLM wheel"
        exit 1
    fi
}


main() {
    echo "Forge User Installation"
    echo "======================"
    echo ""
    echo "Note: Run this from the root of the forge repository"
    echo "This script requires GitHub CLI (gh) to download large wheels"
    echo ""

    check_conda_env
    check_sudo
    check_wheels

    install_system_packages
    download_vllm_wheel

    log_info "Installing PyTorch nightly..."
    pip install torch==$PYTORCH_VERSION --index-url https://download.pytorch.org/whl/nightly/cu129

    log_info "Installing all wheels (local + downloaded)..."
    pip install "$WHEEL_DIR"/*.whl

    log_info "Installing Forge from source..."
    pip install -e .

    # Set up environment
    log_info "Setting up environment..."

     # Get conda environment directory
    local conda_env_dir="${CONDA_PREFIX}"

    if [ -z "$conda_env_dir" ]; then
        log_error "Could not determine conda environment directory"
        exit 1
    fi

    # Create activation directory if it doesn't exist
    mkdir -p "${conda_env_dir}/etc/conda/activate.d"
    mkdir -p "${conda_env_dir}/etc/conda/deactivate.d"

    local cuda_activation_script="${conda_env_dir}/etc/conda/activate.d/cuda_env.sh"
    cat > "$cuda_activation_script" << 'EOF'
# CUDA environment for Forge
export CUDA_VERSION=12.9
export NVCC=/usr/local/cuda-${CUDA_VERSION}/bin/nvcc
export CUDA_NVCC_EXECUTABLE=/usr/local/cuda-${CUDA_VERSION}/bin/nvcc
export CUDA_HOME=/usr/local/cuda-${CUDA_VERSION}
export PATH="${CUDA_HOME}/bin:$PATH"
export CUDA_INCLUDE_DIRS=$CUDA_HOME/include
export CUDA_CUDART_LIBRARY=$CUDA_HOME/lib64/libcudart.so
export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib:/usr/local/cuda-12.9/compat:${LD_LIBRARY_PATH:-}
export LIBRARY_PATH=${CUDA_HOME}/lib64:${LIBRARY_PATH:-}
EOF

    # Create deactivation script to clean up
    cat > "${conda_env_dir}/etc/conda/deactivate.d/cuda_env.sh" << 'EOF'
# Clean up CUDA environment variables when deactivating
unset CUDA_VERSION
unset NVCC
unset CUDA_NVCC_EXECUTABLE
unset CUDA_HOME
unset CUDA_INCLUDE_DIRS
unset CUDA_CUDART_LIBRARY
# Note: We don't unset PATH and LD_LIBRARY_PATH as they may have other content
EOF

    # Source the activation script so it works in the current session.
    log_info "Loading CUDA environment for current session..."
    source "$cuda_activation_script"

    # Test installation
    log_info "Testing installation..."
    python -c "import torch; print(f'PyTorch {torch.__version__} (CUDA: {torch.cuda.is_available()})')"
    python -c "import vllm; print('vLLM imported successfully')"

    # Test other imports if possible
    if python -c "import torchtitan" 2>/dev/null; then
        echo "torchtitan imported successfully"
    fi
    if python -c "import forge" 2>/dev/null; then
        echo "forge imported successfully"
    fi

    echo ""
    log_info "Installation completed successfully!"
    log_info ""
    log_info "To use the environment:"
    log_info "  conda activate $CONDA_DEFAULT_ENV"
    log_info ""
    log_info "Or add to your ~/.bashrc:"
    log_info "  echo 'source ~/.forge_cuda_env' >> ~/.bashrc"
}

main "$@"
