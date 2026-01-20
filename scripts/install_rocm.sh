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
if [ -z "${PYTORCH_VERSION:-}" ]; then
    log_error "PYTORCH_VERSION not set in $VERSIONS_FILE"
    exit 1
fi
if [ -z "${VLLM_VERSION:-}" ]; then
    log_error "VLLM_VERSION not set in $VERSIONS_FILE"
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

# Defaults (override via environment variables)
FORGE_DEPS_DIR="${FORGE_DEPS_DIR:-$HOME/.cache/torchforge}"
PYTORCH_CHANNEL="${PYTORCH_CHANNEL:-auto}" # auto|stable|nightly

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

detect_rocm_version() {
    if [ -n "${ROCM_VERSION:-}" ]; then
        echo "$ROCM_VERSION"
        return 0
    fi

    local version=""
    if [ -f /opt/rocm/.info/version ]; then
        version=$(head -n 1 /opt/rocm/.info/version | awk '{print $1}' | awk -F. '{print $1"."$2}')
    fi

    if [ -z "$version" ] && command -v rocminfo >/dev/null 2>&1; then
        version=$(rocminfo 2>/dev/null | grep -m1 -E "ROCm.*[Vv]ersion" | grep -oE "[0-9]+\.[0-9]+" | head -n 1 || true)
    fi

    if [ -z "$version" ] && command -v rocm-smi >/dev/null 2>&1; then
        version=$(rocm-smi --showdriverversion 2>/dev/null | grep -oE "[0-9]+\.[0-9]+" | head -n 1 || true)
    fi

    if [ -z "$version" ]; then
        log_error "Unable to detect ROCm version. Set ROCM_VERSION=6.4 (or your ROCm version) and retry."
        exit 1
    fi

    echo "$version"
}

detect_rocm_arch() {
    if [ -n "${PYTORCH_ROCM_ARCH:-}" ]; then
        echo "$PYTORCH_ROCM_ARCH"
        return 0
    fi

    local archs=""
    if command -v rocminfo >/dev/null 2>&1; then
        archs=$(rocminfo 2>/dev/null | grep -oE "gfx[0-9a-f]+" | sort -u | paste -sd ";" - || true)
    fi

    if [ -z "$archs" ] && command -v rocm-smi >/dev/null 2>&1; then
        archs=$(rocm-smi --showproductname 2>/dev/null | grep -oE "gfx[0-9a-f]+" | sort -u | paste -sd ";" - || true)
    fi

    if [ -z "$archs" ]; then
        log_error "Unable to auto-detect PYTORCH_ROCM_ARCH. Set PYTORCH_ROCM_ARCH=gfx942 (example) and retry."
        exit 1
    fi

    echo "$archs"
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

install_amdsmi() {
    if [ "${SKIP_AMDSMI:-false}" = "true" ]; then
        log_info "Skipping amdsmi installation as requested"
        return
    fi

    if [ -n "${AMDSMI_VERSION:-}" ]; then
        log_info "Installing amdsmi==$AMDSMI_VERSION"
        python -m pip install "amdsmi==${AMDSMI_VERSION}"
        return
    fi

    local amdsmi_spec=""
    if [[ "$ROCM_VERSION" =~ ^[0-9]+\.[0-9]+$ ]]; then
        amdsmi_spec="==${ROCM_VERSION}.*"
    else
        amdsmi_spec="~=${ROCM_VERSION}"
    fi

    log_info "Installing amdsmi${amdsmi_spec} (pinned to ROCm ${ROCM_VERSION})"
    if ! python -m pip install "amdsmi${amdsmi_spec}"; then
        log_warning "Failed to install amdsmi${amdsmi_spec}; retrying without version pin (set AMDSMI_VERSION to override)"
        python -m pip install amdsmi || log_warning "amdsmi install failed; continuing"
    fi
}

install_pytorch() {
    local rocm_major="${ROCM_VERSION%%.*}"
    local channel="$PYTORCH_CHANNEL"

    if [ "$channel" = "auto" ]; then
        if [ "$rocm_major" -ge 7 ]; then
            channel="nightly"
        else
            channel="stable"
        fi
    fi

    if [ "$channel" = "stable" ]; then
        log_info "Installing stable PyTorch ${PYTORCH_VERSION} for ROCm ${ROCM_VERSION}"
        python -m pip install \
            "torch==${PYTORCH_VERSION}+rocm${ROCM_VERSION}" \
            torchvision torchaudio \
            --index-url "https://download.pytorch.org/whl/rocm${ROCM_VERSION}" \
            --force-reinstall
    elif [ "$channel" = "nightly" ]; then
        log_warning "Installing nightly PyTorch for ROCm ${ROCM_VERSION} (stable not available for this ROCm version)"
        python -m pip install \
            --pre torch torchvision torchaudio \
            --index-url "https://download.pytorch.org/whl/nightly/rocm${ROCM_VERSION}" \
            --force-reinstall
    else
        log_error "Unknown PYTORCH_CHANNEL: $PYTORCH_CHANNEL (expected auto|stable|nightly)"
        exit 1
    fi
}

install_vllm() {
    local vllm_dir="${FORGE_DEPS_DIR}/vllm"

    log_info "Installing vLLM ${VLLM_VERSION} from source (ROCm)"
    ensure_repo "https://github.com/vllm-project/vllm.git" "$vllm_dir" "$VLLM_VERSION"

    python -m pip install -r "${vllm_dir}/requirements/rocm.txt"
    python -m pip install --upgrade "cmake>=3.27" ninja
    install_amdsmi

    VLLM_TARGET_DEVICE=rocm PYTORCH_ROCM_ARCH="${PYTORCH_ROCM_ARCH}" \
        python -m pip install -e "$vllm_dir" --no-build-isolation
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

    # ROCm builds disable tensor_engine (RDMA/distributed tensor features).
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
        "torchmonarch-nightly",
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
        "torchmonarch-nightly",
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
    # ROCm avoids CUDA-only pins like torchmonarch-nightly by installing deps explicitly.
    readarray -t base_deps < <(read_project_deps base)
    if [ "${#base_deps[@]}" -gt 0 ]; then
        python -m pip install "${base_deps[@]}"
    fi

    readarray -t dev_deps < <(read_project_deps dev)
    if [ "${#dev_deps[@]}" -gt 0 ]; then
        python -m pip install "${dev_deps[@]}"
    fi
}

setup_rocm_env() {
    local conda_env_dir="${CONDA_PREFIX}"

    if [ -z "$conda_env_dir" ]; then
        log_error "Could not determine conda environment directory"
        exit 1
    fi

    mkdir -p "${conda_env_dir}/etc/conda/activate.d"
    mkdir -p "${conda_env_dir}/etc/conda/deactivate.d"

    local rocm_activation_script="${conda_env_dir}/etc/conda/activate.d/rocm_env.sh"
    cat > "$rocm_activation_script" << EOF
# ROCm environment for Forge
export PYTORCH_ROCM_ARCH="${PYTORCH_ROCM_ARCH}"
export VLLM_TARGET_DEVICE="rocm"
export USE_ROCM="1"
EOF

    cat > "${conda_env_dir}/etc/conda/deactivate.d/rocm_env.sh" << 'EOF'
# Clean up ROCm environment variables when deactivating
unset PYTORCH_ROCM_ARCH
unset VLLM_TARGET_DEVICE
unset USE_ROCM
EOF

    # DO NOT set LD_LIBRARY_PATH globally; use a Python-only shim like CUDA install.sh.
    local py_shim_activate="${conda_env_dir}/etc/conda/activate.d/python_ld_shim.sh"
    cat > "$py_shim_activate" << 'EOF'
# Python-only LD_LIBRARY_PATH shim for ROCm Torch libs.
TORCHFORGE_TORCH_LIB="$(python - <<'PY'
import os
import torch
print(os.path.join(os.path.dirname(torch.__file__), "lib"))
PY
)"
export TORCHFORGE_TORCH_LIB

python()  { LD_LIBRARY_PATH="${TORCHFORGE_TORCH_LIB}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" command python  "$@"; }
python3() { LD_LIBRARY_PATH="${TORCHFORGE_TORCH_LIB}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" command python3 "$@"; }

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
unset TORCHFORGE_TORCH_LIB
EOF

    log_info "Loading ROCm env for current session..."
    # shellcheck source=/dev/null
    source "$rocm_activation_script"
    # shellcheck source=/dev/null
    source "$py_shim_activate"
}

# Parse command line arguments
parse_args() {
    USE_SUDO=false
    SKIP_AMDSMI=false

    while [[ $# -gt 0 ]]; do
        case $1 in
            --use-sudo)
                USE_SUDO=true
                shift
                ;;
            --skip-amdsmi)
                SKIP_AMDSMI=true
                shift
                ;;
            -h|--help)
                echo "Usage: $0 [OPTIONS]"
                echo ""
                echo "Options:"
                echo "  --use-sudo      Use system package manager instead of conda for system packages"
                echo "  --skip-amdsmi   Skip installing amdsmi (optional ROCm helper library)"
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

    echo "Forge ROCm Installation"
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
    check_command git
    check_command python
    check_command pip
    check_command conda

    mkdir -p "$FORGE_DEPS_DIR"

    # Install build prerequisites
    install_system_packages "$USE_SUDO"

    ROCM_VERSION="$(detect_rocm_version)"
    PYTORCH_ROCM_ARCH="$(detect_rocm_arch)"
    export ROCM_VERSION
    export PYTORCH_ROCM_ARCH
    export VLLM_TARGET_DEVICE=rocm

    log_info "Detected ROCm version: ${ROCM_VERSION}"
    log_info "Detected PYTORCH_ROCM_ARCH: ${PYTORCH_ROCM_ARCH}"

    install_pytorch
    install_vllm
    install_torchstore
    install_torchtitan
    ensure_rust
    install_monarch
    install_forge
    setup_rocm_env

    # Test installation
    log_info "Testing installation..."
    python -c "import torch; print(f'PyTorch {torch.__version__} (HIP: {torch.version.hip})')"
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
