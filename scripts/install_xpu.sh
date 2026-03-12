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
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VERSIONS_FILE="$REPO_ROOT/assets/versions.sh"
PYPROJECT_FILE="$REPO_ROOT/pyproject.toml"

if [ ! -f "$VERSIONS_FILE" ]; then
    log_error "Versions file not found: $VERSIONS_FILE"
    exit 1
fi

source "$VERSIONS_FILE"

# Validate required variables are set
if [ -z "${VLLM_XPU_VERSION:-}" ]; then
    log_error "VLLM_XPU_VERSION not set in $VERSIONS_FILE"
    exit 1
fi
if [ -z "${TORCHSTORE_BRANCH:-}" ]; then
    log_error "TORCHSTORE_BRANCH not set in $VERSIONS_FILE"
    exit 1
fi
if [ -z "${TORCHTITAN_VERSION:-}" ]; then
    log_error "TORCHTITAN_VERSION not set in $VERSIONS_FILE"
    exit 1
fi
if [ -z "${MONARCH_VERSION:-}" ]; then
    log_error "MONARCH_VERSION not set in $VERSIONS_FILE"
    exit 1
fi
if [ -z "${PYTORCH_XPU_VERSION:-}" ]; then
    log_error "PYTORCH_XPU_VERSION not set in $VERSIONS_FILE"
    exit 1
fi

# Defaults (override via environment variables)
FORGE_DEPS_DIR="${FORGE_DEPS_DIR:-$HOME/.cache/torchforge}"

# Check conda environment
check_conda_env() {
    if [ -z "${CONDA_DEFAULT_ENV:-}" ]; then
        log_error "Not running in a conda environment"
        log_info "Please create and activate your conda environment first:"
        log_info "  conda create -n forge python=3.12 -y"
        log_info "  conda activate forge"
        exit 1
    fi
    log_info "Installing in conda environment: $CONDA_DEFAULT_ENV"
}

check_python_version() {
    local required="$XPU_PYTHON_VERSION"
    local actual
    actual=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")

    if [ "$actual" != "$required" ]; then
        log_error "Python ${actual} detected, but vllm-xpu-kernels requires Python ${required}"
        log_info "Recreate your conda env with the correct version:"
        log_info "  conda create -n forge python=${required} -y"
        exit 1
    fi
    log_info "Python version ${actual} matches XPU requirement"
}

# Check required command
check_command() {
    if ! command -v "$1" >/dev/null 2>&1; then
        log_error "Required command '$1' not found"
        exit 1
    fi
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
                    sudo dnf install -y libibverbs rdma-core libmlx5 libibverbs-devel rdma-core-devel \
                        libunwind libunwind-devel clang protobuf-compiler
                    ;;
                "debian")
                    log_info "Detected Debian-based OS - using system package manager"
                    sudo apt-get update
                    sudo apt-get install -y libibverbs1 rdma-core libmlx5-1 libibverbs-dev rdma-core-dev \
                        libunwind-dev clang protobuf-compiler
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
        conda install -c conda-forge rdma-core libibverbs-cos7-x86_64 libunwind clang libprotobuf -y
        log_info "Conda package installation completed. Packages installed in conda environment."
    fi
}

setup_xpu_env() {
    local conda_env_dir="${CONDA_PREFIX}"

    if [ -z "$conda_env_dir" ]; then
        log_error "Could not determine conda environment directory"
        exit 1
    fi

    mkdir -p "${conda_env_dir}/etc/conda/activate.d"

    cat > "${conda_env_dir}/etc/conda/activate.d/xpu_env.sh" << 'EOF'
# Source oneAPI if not already active
if [ -z "${CMPLR_ROOT:-}" ] && [ -z "${MKLROOT:-}" ]; then
    if [ -n "${ONEAPI_ROOT:-}" ] && [ -f "${ONEAPI_ROOT}/setvars.sh" ]; then
        source "${ONEAPI_ROOT}/setvars.sh" --force 2>/dev/null || true
    elif [ -f /opt/intel/oneapi/setvars.sh ]; then
        source /opt/intel/oneapi/setvars.sh --force 2>/dev/null || true
    fi
fi
EOF

    # Source for current session
    # shellcheck source=/dev/null
    set +euo pipefail
    source "${conda_env_dir}/etc/conda/activate.d/xpu_env.sh"
    set -euo pipefail

    # Validate oneAPI is now available
    if [ -z "${CMPLR_ROOT:-}" ] && [ -z "${MKLROOT:-}" ]; then
        # Check module system as fallback
        if command -v module >/dev/null 2>&1 && module list 2>&1 | grep -qi "oneapi\|intel"; then
            log_info "oneAPI loaded via module system"
        else
            log_error "Intel oneAPI not found after sourcing activation script"
            log_info "Expected locations:"
            log_info "  \$ONEAPI_ROOT/setvars.sh"
            log_info "  /opt/intel/oneapi/setvars.sh"
            log_info "Or load via: module load intel/oneapi"
            exit 1
        fi
    else
        log_info "oneAPI environment active (CMPLR_ROOT or MKLROOT set)"
    fi

    log_info "XPU conda activation hook installed"
}

ensure_repo() {
    local repo_url=$1
    local dest=$2
    local ref=$3

    if [ ! -d "$dest/.git" ]; then
        log_info "Cloning $repo_url into $dest"
        git clone "$repo_url" "$dest"
    else
        log_info "Reusing existing repo at $dest"
    fi

    git -C "$dest" fetch origin --tags
    if [ -n "$ref" ]; then
        git -C "$dest" checkout "$ref"
    fi
}

ensure_rust() {
    if ! command -v rustup >/dev/null 2>&1; then
        log_info "rustup not found; installing rustup"
        check_command curl
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    fi

    if [ -f "$HOME/.cargo/env" ]; then
        # shellcheck disable=SC1090
        source "$HOME/.cargo/env"
    fi

    log_info "Ensuring Rust nightly toolchain"
    rustup toolchain install nightly
    rustup default nightly
}

create_constraints_file() {
    local torch_version
    torch_version=$(python -c "import torch; print(torch.__version__)")

    local constraints_file="${FORGE_DEPS_DIR}/constraints.txt"
    cat > "$constraints_file" <<EOF
torch==${torch_version}
EOF
    export PIP_CONSTRAINT="$constraints_file"
    log_info "Pip constraints locked: torch==${torch_version}"
}

install_vllm_xpu() {
    local vllm_dir="${FORGE_DEPS_DIR}/vllm"

    log_info "Installing vLLM ${VLLM_XPU_VERSION} from source (XPU)"
    ensure_repo "https://github.com/vllm-project/vllm.git" "$vllm_dir" "$VLLM_XPU_VERSION"

    # Let vLLM's xpu requirements drive the PyTorch + triton-xpu install.
    python -m pip install -r "${vllm_dir}/requirements/xpu.txt"

    # triton-xpu (required by torch 2.10+xpu) and vanilla triton (required by
    # xgrammar) both install into the same `triton/` namespace directory.
    # In PyTorch <=2.9 the XPU package was called pytorch-triton-xpu and used a
    # separate namespace, so the two coexisted.  After the rename to triton-xpu
    # pip installs both, and vanilla triton's libtriton.so overwrites the XPU
    # one — stripping the 'intel' backend symbol.
    #
    # Fix: force-reinstall triton-xpu so its libtriton.so (with 'intel') wins.
    # We keep vanilla triton installed so xgrammar's pip dependency stays
    # satisfied (triton-xpu does not declare Provides: triton).
    local triton_xpu_version
    triton_xpu_version=$(python -c "import importlib.metadata; print(importlib.metadata.version('triton-xpu'))")
    log_info "Fixing triton namespace conflict: reinstalling triton-xpu ${triton_xpu_version}"
    python -m pip install "triton-xpu==${triton_xpu_version}" --force-reinstall --no-deps \
        --extra-index-url https://download.pytorch.org/whl/xpu

    # Lock torch so later installs can't clobber it
    create_constraints_file

    VLLM_TARGET_DEVICE=xpu \
        python -m pip install -e "$vllm_dir" --no-build-isolation
}

verify_pytorch_xpu() {
    local actual_version
    actual_version=$(python -c "import torch; print(torch.__version__.split('+')[0])")

    if [ "$actual_version" != "${PYTORCH_XPU_VERSION}" ]; then
        log_error "Expected PyTorch ${PYTORCH_XPU_VERSION} but got ${actual_version}"
        log_info "vLLM's requirements may have installed an incompatible version"
        exit 1
    fi
    log_info "PyTorch ${actual_version}+xpu verified"
}

install_torchstore() {
    log_info "Installing torchstore from branch ${TORCHSTORE_BRANCH}"
    python -m pip install "git+https://github.com/meta-pytorch/torchstore.git@${TORCHSTORE_BRANCH}"
}

install_torchtitan() {
    log_info "Installing torchtitan from tag ${TORCHTITAN_VERSION}"
    python -m pip install "git+https://github.com/pytorch/torchtitan.git@${TORCHTITAN_VERSION}"
}

install_monarch() {
    local monarch_dir="${FORGE_DEPS_DIR}/monarch"

    log_info "Installing Monarch ${MONARCH_VERSION} from source"
    ensure_repo "https://github.com/meta-pytorch/monarch.git" "$monarch_dir" "$MONARCH_VERSION"

    python -m pip install -r "${monarch_dir}/build-requirements.txt"
    if ! ulimit -n 2048; then
        log_warning "Unable to raise open file limit to 2048, continuing anyway"
    fi

    # XPU builds disable tensor_engine (RDMA/distributed tensor features).
    USE_TENSOR_ENGINE=0 LIBRARY_PATH="${CONDA_PREFIX}/lib${LIBRARY_PATH:+:$LIBRARY_PATH}" \
        python -m pip install --no-build-isolation -e "$monarch_dir"
}

read_project_deps() {
    local dep_kind=$1
    local output=""

    if ! output=$(DEP_KIND="$dep_kind" PYPROJECT_FILE="$PYPROJECT_FILE" python - <<'PY'
import os
import re
from pathlib import Path

try:
    import tomllib
except ImportError:
    import tomli as tomllib  # type: ignore

dep_kind = os.environ["DEP_KIND"]
pyproject_file = Path(os.environ["PYPROJECT_FILE"])
data = tomllib.loads(pyproject_file.read_text())

deps = []
if dep_kind == "base":
    deps = data.get("project", {}).get("dependencies", [])
    skip = {
        "torch",
        "vllm",
        "torchstore",
        "torchtitan",
        "torchmonarch",
    }
    def name_of(req):
        return re.split(r"[<=>!~ \\[]", req, 1)[0].strip()
    deps = [d for d in deps if name_of(d) not in skip]
elif dep_kind == "dev":
    deps = data.get("project", {}).get("optional-dependencies", {}).get("dev", [])
else:
    raise SystemExit(f"Unknown dep kind: {dep_kind}")

if deps:
    print("\n".join(deps))
PY
); then
        log_warning "Failed to parse pyproject.toml; installing tomli and retrying"
        python -m pip install tomli
        output=$(DEP_KIND="$dep_kind" PYPROJECT_FILE="$PYPROJECT_FILE" python - <<'PY'
import os
import re
from pathlib import Path

import tomli as tomllib

dep_kind = os.environ["DEP_KIND"]
pyproject_file = Path(os.environ["PYPROJECT_FILE"])
data = tomllib.loads(pyproject_file.read_text())

deps = []
if dep_kind == "base":
    deps = data.get("project", {}).get("dependencies", [])
    skip = {
        "torch",
        "vllm",
        "torchstore",
        "torchtitan",
        "torchmonarch",
    }
    def name_of(req):
        return re.split(r"[<=>!~ \\[]", req, 1)[0].strip()
    deps = [d for d in deps if name_of(d) not in skip]
elif dep_kind == "dev":
    deps = data.get("project", {}).get("optional-dependencies", {}).get("dev", [])
else:
    raise SystemExit(f"Unknown dep kind: {dep_kind}")

if deps:
    print("\n".join(deps))
PY
)
    fi

    if [ -n "$output" ]; then
        printf '%s\n' "$output"
    fi
}

install_forge() {
    log_info "Installing Forge from source (no deps)"
    python -m pip install -e "${REPO_ROOT}[dev]" --no-deps

    log_info "Installing Forge dependencies from pyproject.toml"
    # XPU avoids CUDA-only pins like torchmonarch-nightly by installing deps explicitly.
    readarray -t base_deps < <(read_project_deps base)
    if [ "${#base_deps[@]}" -gt 0 ]; then
        python -m pip install "${base_deps[@]}"
    fi

    readarray -t dev_deps < <(read_project_deps dev)
    if [ "${#dev_deps[@]}" -gt 0 ]; then
        python -m pip install "${dev_deps[@]}"
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
                echo "  --use-sudo      Use system package manager instead of conda for system packages"
                echo "  -h, --help      Show this help message"
                echo ""
                echo "By default, system packages are installed via conda for better isolation."
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

    echo "Forge XPU Installation"
    echo "======================="
    echo ""
    echo "Note: Run this from the root of the torchforge repository"
    if [ "$USE_SUDO" = "true" ]; then
        echo "System packages will be installed via system package manager (requires sudo)"
        check_sudo
    else
        echo "System packages will be installed via conda (default, safer)"
    fi
    echo ""

    check_conda_env
    check_python_version
    check_command git
    check_command python
    check_command pip
    check_command conda

    mkdir -p "$FORGE_DEPS_DIR"

    # Install build prerequisites
    install_system_packages "$USE_SUDO"
    setup_xpu_env

    # vLLM installs PyTorch + triton-xpu, fixes triton conflict, creates constraints
    install_vllm_xpu
    verify_pytorch_xpu

    # Everything below is protected by PIP_CONSTRAINT
    install_torchstore
    install_torchtitan
    ensure_rust
    install_monarch
    install_forge

    # Test installation
    log_info "Testing installation..."
    python -c "import torch; print(f'PyTorch {torch.__version__} (XPU: {torch.xpu.is_available()})')"
    python -c "import vllm; print('vLLM imported successfully')"

    # Test other imports if possible
    if python -c "import torchtitan" 2>/dev/null; then
        echo "torchtitan imported successfully"
    fi
    if python -c "import monarch" 2>/dev/null; then
        echo "monarch imported successfully"
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
