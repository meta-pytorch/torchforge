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
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
PYTORCH_VERSION="2.9.0.dev20250905"
VLLM_BRANCH="v0.10.0"
MONARCH_COMMIT="6ca383aca99480aa1bf5853478d4d09fcb224035"
TORCHTITAN_COMMIT="0cfbd0b3c2d827af629a107a77a9e47229c31663"
TORCHSTORE_COMMIT="eed96eb55ce87d4a9880597dd7dfd0d291e9ac81"
BUILD_DIR="$HOME/forge-build"
WHEEL_DIR="$(pwd)/assets/wheels"

# Logging functions
log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_step() { echo -e "${BLUE}[$1/$2]${NC} $3"; }

# Track current step for recovery
CURRENT_STEP=0
TOTAL_STEPS=8

# Function to handle step failures
handle_failure() {
    local step_name="$1"
    local step_num="$2"
    local exit_code="$3"
    local retry_cmd="$4"

    log_error "Step $step_num failed: $step_name"
    log_error "Exit code: $exit_code"
    log_error "Working directory: $(pwd)"
    echo ""
    log_info "To retry this step manually:"
    echo "  $retry_cmd"
    echo ""
    log_info "Or to resume from step $step_num:"
    echo "  $0 --resume-from=$step_num"
    echo ""
    exit $exit_code
}

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

    export CUDA_VERSION=12.9
    export NVCC=/usr/local/cuda-${CUDA_VERSION}/bin/nvcc
    export CUDA_NVCC_EXECUTABLE=/usr/local/cuda-${CUDA_VERSION}/bin/nvcc
    export CUDA_HOME=/usr/local/cuda-${CUDA_VERSION}
    export PATH="${CUDA_HOME}/bin:$PATH"
    export CUDA_INCLUDE_DIRS=$CUDA_HOME/include
    export CUDA_CUDART_LIBRARY=$CUDA_HOME/lib64/libcudart.so
    export LD_LIBRARY_PATH=/usr/local/cuda-12.9/compat:${LD_LIBRARY_PATH:-}
    export LIBRARY_PATH=$CUDA_HOME/lib64:${LIBRARY_PATH:-}

    # Save to file for persistence
    cat > ~/.forge_cuda_env << 'EOF'
export CUDA_VERSION=12.9
export NVCC=/usr/local/cuda-${CUDA_VERSION}/bin/nvcc
export CUDA_NVCC_EXECUTABLE=/usr/local/cuda-${CUDA_VERSION}/bin/nvcc
export CUDA_HOME=/usr/local/cuda-${CUDA_VERSION}
export PATH="${CUDA_HOME}/bin:$PATH"
export CUDA_INCLUDE_DIRS=$CUDA_HOME/include
export CUDA_CUDART_LIBRARY=$CUDA_HOME/lib64/libcudart.so
export LD_LIBRARY_PATH=/usr/local/cuda-12.9/compat:${LD_LIBRARY_PATH:-}
export LIBRARY_PATH=${CUDA_HOME}/lib64:${LIBRARY_PATH:-}
EOF

    log_info "CUDA environment configured"
}

# Parse command line arguments
RESUME_FROM=1
while [[ $# -gt 0 ]]; do
    case $1 in
        --resume-from=*)
            RESUME_FROM="${1#*=}"
            shift
            ;;
        *)
            log_error "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# Step execution wrapper
run_step() {
    local step_num="$1"
    local step_name="$2"
    local step_function="$3"

    if [ "$step_num" -lt "$RESUME_FROM" ]; then
        log_info "Skipping step $step_num: $step_name (resuming from step $RESUME_FROM)"
        return 0
    fi

    CURRENT_STEP=$step_num
    log_step "$step_num" "$TOTAL_STEPS" "$step_name"

    if ! $step_function; then
        handle_failure "$step_name" "$step_num" "$?" "Run step $step_num manually"
    fi
}

# Step 1: Install PyTorch nightly
step1_pytorch() {
    pip3 install --pre torch==$PYTORCH_VERSION --index-url https://download.pytorch.org/whl/nightly/cu129
}

# Step 2: Install CUDA system packages
step2_cuda_packages() {
    sudo dnf install -y cuda-toolkit-12-9 cuda-compat-12-9
    setup_cuda_env
}

# Step 3: Build vLLM wheel
step3_vllm() {
    cd "$BUILD_DIR"
    if [ -d "vllm" ]; then
        log_warn "vLLM directory exists, removing..."
        rm -rf vllm
    fi

    git clone https://github.com/vllm-project/vllm.git --branch $VLLM_BRANCH
    cd "$BUILD_DIR/vllm"

    python use_existing_torch.py
    pip install -r requirements/build.txt
    pip wheel --no-build-isolation --no-deps . -w "$WHEEL_DIR"
}

# Step 4: Setup Rust toolchain
step4_rust_setup() {
    # Install Rust if not present
    if ! command -v rustup &> /dev/null; then
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
        source ~/.cargo/env
    fi

    rustup toolchain install nightly
    rustup default nightly

    # Install additional system packages
    conda install -y libunwind
    sudo dnf install -y clang-devel libnccl-devel
    sudo dnf install -y libibverbs rdma-core libmlx5 libibverbs-devel rdma-core-devel
}

# Step 5: Build Monarch wheel
step5_monarch() {
    cd "$BUILD_DIR"
    if [ -d "monarch" ]; then
        log_warn "Monarch directory exists, removing..."
        rm -rf monarch
    fi

    git clone https://github.com/meta-pytorch/monarch.git
    cd "$BUILD_DIR/monarch"
    git checkout $MONARCH_COMMIT

    pip install -r build-requirements.txt
    pip wheel --no-build-isolation --no-deps . -w "$WHEEL_DIR"
}

# Step 6: Build torchtitan wheel
step6_torchtitan() {
    cd "$BUILD_DIR"
    if [ -d "torchtitan" ]; then
        log_warn "torchtitan directory exists, removing..."
        rm -rf torchtitan
    fi

    git clone https://github.com/pytorch/torchtitan.git
    cd "$BUILD_DIR/torchtitan"
    git checkout $TORCHTITAN_COMMIT

    pip wheel --no-deps . -w "$WHEEL_DIR"
}

# Step 7: Build torchstore wheel
step7_torchstore() {
    cd "$BUILD_DIR"
    if [ -d "torchstore" ]; then
        log_warn "torchstore directory exists, removing..."
        rm -rf torchstore
    fi

    git clone https://github.com/meta-pytorch/torchstore.git
    cd "$BUILD_DIR/torchstore"
    git checkout $TORCHSTORE_COMMIT

    pip wheel --no-deps . -w "$WHEEL_DIR"
}

# Verification
verify_installation() {
    log_info "Verifying wheel builds..."

    python -c "import torch; print(f'PyTorch {torch.__version__} (CUDA: {torch.cuda.is_available()})')"

    # Check that wheels were created
    wheel_count=$(ls -1 "$WHEEL_DIR"/*.whl 2>/dev/null | wc -l)
    if [ "$wheel_count" -gt 0 ]; then
        log_info "Built $wheel_count wheels:"
        ls -1 "$WHEEL_DIR"/*.whl | sed 's/.*\//  /'
    else
        log_error "No wheels found in $WHEEL_DIR"
        return 1
    fi

    log_info "Wheel building verification complete!"
}

# Main execution
main() {
    echo "Forge Wheel Builder"
    echo "==================="
    echo ""

    if [ "$RESUME_FROM" -gt 1 ]; then
        log_info "Resuming from step $RESUME_FROM..."
        # Source CUDA env if resuming
        if [ -f ~/.forge_cuda_env ]; then
            source ~/.forge_cuda_env
        fi
        # Source Rust env if resuming
        if [ -f ~/.cargo/env ]; then
            source ~/.cargo/env
        fi
    else
        validate_environment
        setup_build_dir
    fi

    run_step 1 "Installing PyTorch nightly" step1_pytorch
    run_step 2 "Installing CUDA packages and setting environment" step2_cuda_packages
    run_step 3 "Building vLLM wheel" step3_vllm
    run_step 4 "Setting up Rust toolchain and additional packages" step4_rust_setup
    run_step 5 "Building Monarch wheel" step5_monarch
    run_step 6 "Building torchtitan wheel" step6_torchtitan
    run_step 7 "Building torchstore wheel" step7_torchstore

    verify_installation

    echo ""
    log_info "Wheel building completed successfully!"
    log_info ""
    log_info "Built wheels are in: $WHEEL_DIR"
    log_info ""
    log_info "Users can now install with:"
    log_info "  conda create -n forge python=3.10 -y"
    log_info "  conda activate forge"
    log_info "  pip install torch==$PYTORCH_VERSION --index-url https://download.pytorch.org/whl/nightly/cu129"
    log_info "  pip install $WHEEL_DIR/*.whl"
    log_info "  source ~/.forge_cuda_env"
    log_info ""
    log_info "Build artifacts are in: $BUILD_DIR"
    log_info "You can remove them with: rm -rf $BUILD_DIR"
}

# Set trap for cleanup on failure
cleanup() {
    if [ $? -ne 0 ] && [ $CURRENT_STEP -gt 0 ]; then
        echo ""
        log_error "Setup failed at step $CURRENT_STEP"
        log_info "You can resume with: $0 --resume-from=$CURRENT_STEP"
    fi
}
trap cleanup EXIT

# Run main function
main "$@"
