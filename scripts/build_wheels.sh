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

# Source version configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERSIONS_FILE="$SCRIPT_DIR/../assets/versions.sh"

if [ ! -f "$VERSIONS_FILE" ]; then
    echo -e "${RED}[ERROR]${NC} Versions file not found: $VERSIONS_FILE"
    exit 1
fi

source "$VERSIONS_FILE"

# Validate required variables are set
validate_versions() {
    local missing_vars=()

    [ -z "${PYTORCH_VERSION:-}" ] && missing_vars+=("PYTORCH_VERSION")
    [ -z "${VLLM_BRANCH:-}" ] && missing_vars+=("VLLM_BRANCH")
    [ -z "${MONARCH_COMMIT:-}" ] && missing_vars+=("MONARCH_COMMIT")
    [ -z "${TORCHTITAN_COMMIT:-}" ] && missing_vars+=("TORCHTITAN_COMMIT")
    [ -z "${TORCHSTORE_COMMIT:-}" ] && missing_vars+=("TORCHSTORE_COMMIT")

    if [ ${#missing_vars[@]} -gt 0 ]; then
        echo -e "${RED}[ERROR]${NC} Missing required variables in $VERSIONS_FILE:"
        for var in "${missing_vars[@]}"; do
            echo "  - $var"
        done
        exit 1
    fi
}

validate_versions

# Configuration
BUILD_DIR="$HOME/forge-build"
WHEEL_DIR="$(pwd)/assets/wheels"

# Logging functions
log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_step() { echo -e "${BLUE}[STEP]${NC} $1"; }

# Function to handle step failures
handle_failure() {
    local step_name="$1"
    local exit_code="$2"

    log_error "Step failed: $step_name"
    log_error "Exit code: $exit_code"
    log_error "Working directory: $(pwd)"
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

    export CUDA_VERSION=12.
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

# Parse command line arguments
BUILD_TARGETS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        vllm|monarch|torchtitan|torchstore)
            BUILD_TARGETS+=("$1")
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [TARGETS...]"
            echo ""
            echo "Build wheels for Forge dependencies."
            echo ""
            echo "Targets (default: all):"
            echo "  vllm              Build vLLM wheel"
            echo "  monarch           Build Monarch wheel"
            echo "  torchtitan        Build torchtitan wheel"
            echo "  torchstore        Build torchstore wheel"
            echo ""
            echo "Examples:"
            echo "  $0                      # Build all wheels"
            echo "  $0 vllm                 # Build only vLLM"
            echo "  $0 monarch torchtitan   # Build Monarch and torchtitan"
            exit 0
            ;;
        *)
            log_error "Unknown argument: $1"
            log_info "Use --help to see available options"
            exit 1
            ;;
    esac
done

# If no targets specified, build all
if [ ${#BUILD_TARGETS[@]} -eq 0 ]; then
    BUILD_TARGETS=("vllm" "monarch" "torchtitan" "torchstore")
    log_info "No targets specified, building all wheels"
else
    log_info "Building wheels: ${BUILD_TARGETS[*]}"
fi

# Helper function to check if a target should be built
should_build() {
    local target="$1"
    for t in "${BUILD_TARGETS[@]}"; do
        if [ "$t" == "$target" ]; then
            return 0
        fi
    done
    return 1
}

# Step execution wrapper
run_step() {
    local step_name="$1"
    local step_function="$2"

    log_step "$step_name"

    if ! $step_function; then
        handle_failure "$step_name" "$?"
    fi
}

# Step 1: Install PyTorch nightly
step1_pytorch() {
    pip3 install --pre torch==$PYTORCH_VERSION --index-url https://download.pytorch.org/whl/nightly/cu128
}

# Step 2: Install CUDA system packages
step2_cuda_packages() {
    sudo dnf install -y cuda-toolkit-12-9 cuda-compat-12-9
    setup_cuda_env
}

# Step 3: Build vLLM wheel
step3_vllm() {
    log_info "Building vLLM from branch: $VLLM_BRANCH (from $VERSIONS_FILE)"
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
    log_info "Building Monarch from commit: $MONARCH_COMMIT (from $VERSIONS_FILE)"
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
    log_info "Building torchtitan from commit: $TORCHTITAN_COMMIT (from $VERSIONS_FILE)"
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
    log_info "Building torchstore from commit: $TORCHSTORE_COMMIT (from $VERSIONS_FILE)"
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

    validate_environment
    setup_build_dir

    # PyTorch is needed for all builds
    run_step "Installing PyTorch nightly" step1_pytorch

    # CUDA packages are needed for vLLM and Monarch
    if should_build "vllm" || should_build "monarch"; then
        run_step "Installing CUDA packages and setting environment" step2_cuda_packages
    fi

    # Build requested wheels
    if should_build "vllm"; then
        run_step "Building vLLM wheel" step3_vllm
    fi

    # Rust setup is needed for Monarch
    if should_build "monarch"; then
        run_step "Setting up Rust toolchain and additional packages" step4_rust_setup
        run_step "Building Monarch wheel" step5_monarch
    fi

    if should_build "torchtitan"; then
        run_step "Building torchtitan wheel" step6_torchtitan
    fi

    if should_build "torchstore"; then
        run_step "Building torchstore wheel" step7_torchstore
    fi

    verify_installation

    echo ""
    log_info "Wheel building completed successfully!"
    log_info ""
    log_info "Built wheels are in: $WHEEL_DIR"
    log_info ""
    log_info "Users can now install with:"
    log_info "  conda create -n forge python=3.10 -y"
    log_info "  conda activate forge"
    log_info "  pip install torch==$PYTORCH_VERSION --index-url https://download.pytorch.org/whl/nightly/cu128"
    log_info "  pip install $WHEEL_DIR/*.whl"
    if should_build "vllm" || should_build "monarch"; then
        log_info "  source ~/.forge_cuda_env"
    fi
    log_info ""
    log_info "Build artifacts are in: $BUILD_DIR"
    log_info "You can remove them with: rm -rf $BUILD_DIR"
}


# Run main function
main "$@"
