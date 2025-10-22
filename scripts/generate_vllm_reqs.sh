# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#!/bin/bash
set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Source version configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERSIONS_FILE="$SCRIPT_DIR/../assets/versions.sh"

if [ ! -f "$VERSIONS_FILE" ]; then
    echo -e "${RED}[ERROR]${NC} Versions file not found: $VERSIONS_FILE"
    exit 1
fi

source "$VERSIONS_FILE"

# Configuration
BUILD_DIR="$HOME/forge-build"
WHEEL_DIR="$(pwd)/assets/wheels"

# Logging functions
log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }


# Validation functions
check_conda_env() {
    if [ -z "${CONDA_DEFAULT_ENV:-}" ]; then
        log_error "Not running in a conda environment"
        log_info "Please create and activate your conda environment first:"
        log_info "  conda create -n forge python=3.10 -y"
        log_info "  conda activate forge"
        exit 1
    fi
    log_info "Running in conda environment: $CONDA_DEFAULT_ENV"
}

check_command() {
    if ! command -v "$1" &> /dev/null; then
        log_error "Required command '$1' not found"
        exit 1
    fi
}

check_sudo() {
    if ! sudo -n true 2>/dev/null; then
        log_error "This script requires passwordless sudo access"
        log_info "Run 'sudo -v' first, or configure passwordless sudo"
        exit 1
    fi
}

check_disk_space() {
    local required_gb=10
    local available_gb=$(df ~/ --output=avail -BG | tail -1 | sed 's/G//')
    if [ "$available_gb" -lt "$required_gb" ]; then
        log_error "Insufficient disk space. Need ${required_gb}GB, have ${available_gb}GB"
        exit 1
    fi
}

# Main validation
validate_environment() {
    log_info "Validating environment..."

    check_conda_env
    check_command git
    check_command curl
    check_command python
    check_command pip
    check_command conda
    check_sudo
    check_disk_space

    # Check if CUDA toolkit will be available
    if ! ldconfig -p | grep -q cuda; then
        log_warn "CUDA libraries not found in ldconfig. Will attempt to install CUDA toolkit."
    fi

    log_info "Environment validation passed"
}

# Setup build directory and wheels directory
setup_build_dir() {
    log_info "Setting up build directory: $BUILD_DIR"
    mkdir -p "$BUILD_DIR"
    log_info "Setting up wheels directory: $WHEEL_DIR"
    mkdir -p "$WHEEL_DIR"
    log_info "Build and wheels directories created"
}

# Setup CUDA environment variables
setup_cuda_env() {
    log_info "Setting up CUDA environment..."

    export CUDA_VERSION=12.8
    export NVCC=/usr/local/cuda-${CUDA_VERSION}/bin/nvcc
    export CUDA_NVCC_EXECUTABLE=/usr/local/cuda-${CUDA_VERSION}/bin/nvcc
    export CUDA_HOME=/usr/local/cuda-${CUDA_VERSION}
    export PATH="${CUDA_HOME}/bin:$PATH"
    export CUDA_INCLUDE_DIRS=$CUDA_HOME/include
    export CUDA_CUDART_LIBRARY=$CUDA_HOME/lib64/libcudart.so
    export LD_LIBRARY_PATH=/usr/local/cuda-12.8/compat:${LD_LIBRARY_PATH:-}
    export LIBRARY_PATH=$CUDA_HOME/lib64:${LIBRARY_PATH:-}

    # Save to file for persistence
    cat > ~/.forge_cuda_env << 'EOF'
export CUDA_VERSION=12.8
export NVCC=/usr/local/cuda-${CUDA_VERSION}/bin/nvcc
export CUDA_NVCC_EXECUTABLE=/usr/local/cuda-${CUDA_VERSION}/bin/nvcc
export CUDA_HOME=/usr/local/cuda-${CUDA_VERSION}
export PATH="${CUDA_HOME}/bin:$PATH"
export CUDA_INCLUDE_DIRS=$CUDA_HOME/include
export CUDA_CUDART_LIBRARY=$CUDA_HOME/lib64/libcudart.so
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/compat:${LD_LIBRARY_PATH:-}
export LIBRARY_PATH=${CUDA_HOME}/lib64:${LIBRARY_PATH:-}
EOF

    log_info "CUDA environment configured"
}

# Step 1: Install PyTorch stable
step1_pytorch() {
    pip3 install --pre torch==$PYTORCH_VERSION --index-url https://download.pytorch.org/whl/cu128
}

# Step 2: Install CUDA system packages
step2_cuda_packages() {
    sudo dnf install -y cuda-toolkit-12-8 cuda-compat-12-8
    setup_cuda_env
}

# Step 3: Build vLLM wheel
step3_vllm() {
    log_info "Building vLLM from branch: $VLLM_VERSION (from $VERSIONS_FILE)"
    cd "$BUILD_DIR"
    if [ -d "vllm" ]; then
        log_warn "vLLM directory exists, removing..."
        rm -rf vllm
    fi

    git clone https://github.com/vllm-project/vllm.git --branch $VLLM_VERSION
    cd "$BUILD_DIR/vllm"

    python use_existing_torch.py
    pip install -r requirements/build.txt
    pip install --no-build-isolation -e .
}

# Main execution
main() {
    echo "Forge Wheel Builder"
    echo "==================="
    echo ""

    validate_environment
    setup_build_dir

    # Install PyTorch, CUDA packages, and vLLM
    step1_pytorch
    step2_cuda_packages
    step3_vllm

    # Output requirements to .github/packaging/vllm_reqs_12_8.txt
    REQS_FILE="$SCRIPT_DIR/../.github/packaging/vllm_reqs_12_8.txt"
    pip freeze | grep -v "vllm*" > $REQS_FILE
    sed -i '1i# This file was generated by running ./scripts/generate_vllm_reqs.sh' $REQS_FILE
}


# Run main function
main "$@"
