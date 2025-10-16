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
YELLOW='\033[0;33m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1";}

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERSIONS_FILE="$SCRIPT_DIR/../assets/versions.sh"

if [ ! -f "$VERSIONS_FILE" ]; then
    log_error "Versions file not found: $VERSIONS_FILE"
    exit 1
fi

source "$VERSIONS_FILE"

# Validate required variables are set
if [ -z "${PYTORCH_VERSION:-}" ]; then
    log_error "PYTORCH_VERSION not set in $VERSIONS_FILE"
    exit 1
fi

WHEEL_DIR="$SCRIPT_DIR/../assets/wheels"
RELEASE_TAG="v0.0.0-93025"  # Why is this hardcoded?
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

# Check sudo access and if it is not available; continue with Conda
check_sudo() {
    if ! sudo -n true 2>/dev/null; then
        log_warning "Passwordless sudo access is not available."
        log_info "The script will continue and attempt to install packages via conda instead."
    else
        log_info "Passwordless sudo access detected."
    fi
}

# Detect OS distribution from /etc/os-release
detect_os_family() {
    if [ ! -f /etc/os-release ]; then
        log_error "/etc/os-release not found. Cannot determine OS distribution."
        return 1
    fi

    # Source the os-release file to get variables
    . /etc/os-release

    # Check ID_LIKE field for supported distributions
    case "${ID_LIKE:-}" in
        *"rhel"*|*"fedora"*)
            echo "rhel_fedora"
            ;;
        *"debian"*)
            echo "debian"
            ;;
        *)
            # Fallback to ID if ID_LIKE is not set or doesn't match
            case "${ID:-}" in
                "rhel"|"fedora"|"centos"|"rocky"|"almalinux")
                    echo "rhel_fedora"
                    ;;
                "debian"|"ubuntu")
                    echo "debian"
                    ;;
                *)
                    echo "unknown"
                    ;;
            esac
            ;;
    esac
}

# Install required system packages
install_system_packages() {
    local use_sudo=${1:-false}

    log_info "Installing required system packages..."

    if [ "$use_sudo" = "true" ]; then
        # User explicitly requested sudo installation
        if sudo -n true 2>/dev/null; then
            # Detect OS family using /etc/os-release
            local os_family
            os_family=$(detect_os_family)

            case "$os_family" in
                "rhel_fedora")
                    log_info "Detected RHEL/Fedora-based OS - using system package manager"
                    sudo dnf install -y libibverbs rdma-core libmlx5 libibverbs-devel rdma-core-devel
                    ;;
                "debian")
                    log_info "Detected Debian-based OS - using system package manager"
                    sudo apt-get update
                    sudo apt-get install -y libibverbs1 rdma-core libmlx5-1 libibverbs-dev rdma-core-dev
                    ;;
                "unknown")
                    log_error "Unsupported OS for automatic system package installation"
                    log_info "Supported distributions: RHEL/Fedora-based (rhel fedora) and Debian-based (debian)"
                    exit 1
                    ;;
            esac
            log_info "System packages installed successfully via system package manager"
        else
            log_error "Sudo installation requested but no sudo access available"
            log_info "Either run with sudo privileges or remove the --use-sudo flag to use conda"
            exit 1
        fi
    else
        # Default to conda installation
        log_info "Installing system packages via conda (default method)"
        conda install -c conda-forge rdma-core libibverbs-cos7-x86_64 -y
        log_info "Conda package installation completed. Packages installed in conda environment."
    fi
}

# Check to see if gh is installed, if not, it will be installed via conda-forge channel
check_gh_install() {
  if ! command -v gh &> /dev/null; then
    log_warning "GitHub CLI (gh) not found. Installing via Conda..."
    conda install gh --channel conda-forge -y
    log_info "GitHub CLI (gh) installed successfully."
    log_info "Please run 'gh auth login' to authenticate with GitHub."
  else
    log_info "GitHub CLI (gh) already installed."
  fi
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

# Generic package installation function supporting multiple sources
# Args: package_name, version, github_repo
install_package() {
    local package_name="$1"
    local version="$2"
    local github_repo="$3"

    log_info "Installing $package_name..."

    # Determine installation method based on version format
    if [[ "$version" =~ ^v[0-9]+\.[0-9]+\.[0-9]+ ]]; then
        # GitHub release tag format (e.g., v0.10.0)
        log_info "  Method: GitHub release tag ($version)"
        install_from_github_release "$package_name" "$version" "$github_repo"
    elif [[ "$version" =~ ^[0-9a-f]{40}$ ]]; then
        # GitHub commit SHA format (40-character hex string)
        log_info "  Method: GitHub commit SHA ($version)"
        install_from_github_commit "$package_name" "$version" "$github_repo"
    else
        # PyPI or PyTorch index format (e.g., 0.1.0, 2.9.0, 0.1.0.dev20251015)
        log_info "  Method: PyPI/PyTorch index ($version)"
        install_from_index "$package_name" "$version"
    fi
}

# Install from PyPI or PyTorch index
install_from_index() {
    local package_name="$1"
    local version="$2"

    if [[ "$version" == *"dev"* ]]; then
        pip install --pre "${package_name}==${version}" --extra-index-url https://download.pytorch.org/whl/nightly/cu128
    else
        pip install "${package_name}==${version}"
    fi

    if [ $? -ne 0 ]; then
        log_error "Failed to install $package_name from index"
        exit 1
    fi
}

# Install from GitHub release tag
install_from_github_release() {
    local package_name="$1"
    local release_tag="$2"
    local github_repo="$3"

    # Check if gh is installed
    if ! command -v gh &> /dev/null; then
        log_error "GitHub CLI (gh) is required to download from GitHub releases"
        log_info "Run the installation script - it will install gh via conda"
        exit 1
    fi

    # Get the wheel URL from the release
    local wheel_url
    wheel_url=$(gh release view "$release_tag" --repo "$github_repo" --json assets --jq ".assets[] | select(.name | contains(\"$package_name\")) | .url" | head -1)

    if [ -z "$wheel_url" ]; then
        log_error "Could not find $package_name wheel in release $release_tag"
        log_info "Make sure the $package_name wheel has been uploaded to the GitHub release"
        exit 1
    fi

    log_info "  Installing from: $wheel_url"
    pip install "$wheel_url"

    if [ $? -ne 0 ]; then
        log_error "Failed to install $package_name from GitHub release"
        exit 1
    fi
}

# Install from GitHub commit SHA
install_from_github_commit() {
    local package_name="$1"
    local commit_sha="$2"
    local github_repo="$3"

    log_info "  Installing from: git+https://github.com/$github_repo.git@$commit_sha"
    pip install "git+https://github.com/$github_repo.git@$commit_sha"

    if [ $? -ne 0 ]; then
        log_error "Failed to install $package_name from GitHub commit"
        exit 1
    fi
}


# Parse command line arguments
parse_args() {
    USE_SUDO=false

    while [[ $# -gt 0 ]]; do
        case $1 in
            --use-sudo)
                USE_SUDO=true
                shift
                ;;
            -h|--help)
                echo "Usage: $0 [OPTIONS]"
                echo ""
                echo "Options:"
                echo "  --use-sudo    Use system package manager instead of conda for system packages"
                echo "  -h, --help    Show this help message"
                echo ""
                echo "By default, system packages are installed via conda for better isolation."
                echo "Use --use-sudo to install system packages via the system package manager."
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                log_info "Use --help for usage information"
                exit 1
                ;;
        esac
    done
}

main() {
    # Parse command line arguments first
    parse_args "$@"

    echo "Forge User Installation"
    echo "======================"
    echo ""
    echo "Note: Run this from the root of the forge repository"
    echo "This script requires GitHub CLI (gh) to download large wheels"
    if [ "$USE_SUDO" = "true" ]; then
        echo "System packages will be installed via system package manager (requires sudo)"
        check_sudo
    else
        echo "System packages will be installed via conda (default, safer)"
    fi
    echo ""

    check_conda_env
    # check_wheels

    # Install openssl as we overwrite the default version when we update LD_LIBRARY_PATH
    conda install -y openssl

    install_system_packages "$USE_SUDO"
    check_gh_install

    # Install all packages using the generic install_package function
    # Syntax: install_package "package_name" "version" "github_repo"
    install_package "torch" "$PYTORCH_VERSION" "pytorch/pytorch"
    install_package "vllm" "$VLLM_VERSION" "vllm-project/vllm"
    install_package "torchmonarch" "$MONARCH_VERSION" "meta-pytorch/monarch"
    install_package "torchstore" "$TORCHSTORE_VERSION" "meta-pytorch/torchstore"
    install_package "torchtitan" "$TORCHTITAN_VERSION" "pytorch/torchtitan"
    install_package "torch" "$PYTORCH_VERSION" "pytorch/pytorch"

    log_info "Installing TorchForge from source..."
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
export CUDA_VERSION=12.8
export NVCC=/usr/local/cuda-${CUDA_VERSION}/bin/nvcc
export CUDA_NVCC_EXECUTABLE=/usr/local/cuda-${CUDA_VERSION}/bin/nvcc
export CUDA_HOME=/usr/local/cuda-${CUDA_VERSION}
export PATH="${CUDA_HOME}/bin:$PATH"
export CUDA_INCLUDE_DIRS=$CUDA_HOME/include
export CUDA_CUDART_LIBRARY=$CUDA_HOME/lib64/libcudart.so

# Temporary measure until this environment variable is removed
export MONARCH_HOST_MESH_V1_REMOVE_ME_BEFORE_RELEASE=1

# Add only CUDA compat libs to LD_LIBRARY_PATH (safe for system tools)
if [ -n "${LD_LIBRARY_PATH:-}" ]; then
  export LD_LIBRARY_PATH="/usr/local/cuda-${CUDA_VERSION}/compat:${LD_LIBRARY_PATH}"
else
  export LD_LIBRARY_PATH="/usr/local/cuda-${CUDA_VERSION}/compat"
fi
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
# We intentionally do not mutate PATH or LD_LIBRARY_PATH here.
EOF

    ##########################################
    # 2) Python-only LD_LIBRARY_PATH shim(s) #
    ##########################################
    # These shell *functions* ensure that any `python`/`python3` invocation
    # gets ${CONDA_PREFIX}/lib in its environment, without polluting the shell.
    # This avoids OpenSSL/libcrypto collisions with system tools like /usr/bin/conda.
    # TODO: Build Monarch with ABI3 to avoid this hack.
    local py_shim_activate="${conda_env_dir}/etc/conda/activate.d/python_ld_shim.sh"
    cat > "$py_shim_activate" << 'EOF'
# Define python wrappers that only set LD_LIBRARY_PATH for the launched process
python()  { LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" command python  "$@"; }
python3() { LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" command python3 "$@"; }

# Export functions to subshells when possible (best-effort, shell-dependent)
if [ -n "${BASH_VERSION:-}" ]; then
  export -f python python3 2>/dev/null || true
elif [ -n "${ZSH_VERSION:-}" ]; then
  typeset -fx python python3 >/dev/null 2>&1 || true
fi
EOF

    # Deactivation script to remove the function wrappers
    cat > "${conda_env_dir}/etc/conda/deactivate.d/python_ld_shim.sh" << 'EOF'
unset -f python  2>/dev/null || true
unset -f python3 2>/dev/null || true
EOF

    log_info "Loading CUDA env and python LD shim for current session..."
    # shellcheck source=/dev/null
    source "$cuda_activation_script"
    # shellcheck source=/dev/null
    source "$py_shim_activate"

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
    echo ""
    log_info "Re-activate the conda environment to make the changes take effect:"
    log_info "  conda deactivate && conda activate $CONDA_DEFAULT_ENV"
}

main "$@"
