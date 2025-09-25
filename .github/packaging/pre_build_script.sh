#!/bin/bash
set -euxo pipefail

# Builds vLLM, Monarch and torchstore
# This script builds vLLM, Monarch and torchstore and places
# their wheels into dist/.

VLLM_BRANCH="v0.10.0"
MONARCH_COMMIT="265034a29ec3fb35919f4a9c23c65f2f4237190d"
TORCHTITAN_COMMIT="main"
TORCHSTORE_COMMIT="main"
BUILD_DIR="$HOME/forge-build"

# Push everything to the dist folder, so
# the final action pushes everything together
WHL_DIR="${REPOSITORY}/dist"

mkdir -p $BUILD_DIR
mkdir -p $WHL_DIR
echo "build dir is $BUILD_DIR"
echo "wheel dir is $WHL_DIR"

build_vllm() {
    cd "$BUILD_DIR"

    git clone https://github.com/vllm-project/vllm.git --branch $VLLM_BRANCH
    cd "$BUILD_DIR/vllm"

    python use_existing_torch.py
    pip install -r requirements/build.txt
    pip wheel --no-build-isolation --no-deps . -w "$WHL_DIR"
}

build_monarch() {
    # Get Rust build related pieces
    if ! command -v rustup &> /dev/null; then
        echo "getting rustup"
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
        export PATH="$HOME/.cargo/bin:$PATH"
        echo "$HOME/.cargo/bin" >> $GITHUB_PATH
    fi

    rustup toolchain install nightly
    rustup default nightly

    if command -v dnf &>/dev/null; then
        dnf install -y clang-devel \
            libibverbs rdma-core libmlx5 libibverbs-devel rdma-core-devel fmt-devel
    elif command -v apt-get &>/dev/null; then
        apt-get update
        apt-get install -y clang libunwind-dev \
            libibverbs-dev librdmacm-dev libfmt-dev
    fi

    cd "$BUILD_DIR"
    git clone https://github.com/meta-pytorch/monarch.git
    cd "$BUILD_DIR/monarch"
    git checkout $MONARCH_COMMIT

    pip install -r build-requirements.txt
    export USE_TENSOR_ENGINE=1
    export RUST_BACKTRACE=1
    export CARGO_TERM_VERBOSE=true
    export CARGO_TERM_COLOR=always
    pip wheel --no-build-isolation --no-deps . -w "$WHL_DIR"
}

build_torchtitan() {
    cd "$BUILD_DIR"
    git clone https://github.com/pytorch/torchtitan.git
    cd "$BUILD_DIR/torchtitan"
    git checkout $TORCHTITAN_COMMIT

    pip wheel --no-deps . -w "$WHL_DIR"
}

build_torchstore() {
    cd "$BUILD_DIR"
    if [ -d "torchstore" ]; then
        log_warn "torchstore directory exists, removing..."
        rm -rf torchstore
    fi

    git clone https://github.com/meta-pytorch/torchstore.git
    cd "$BUILD_DIR/torchstore"
    git checkout $TORCHSTORE_COMMIT

    pip wheel --no-deps . -w "$WHL_DIR"
}


append_date() {
    # Appends the current date and time to the Forge wheel
    version_file="assets/version.txt"
    init_file="src/forge/__init__.py"
    if [[ -n "$BUILD_VERSION" ]]; then
        # Update the version in version.txt
        echo "$BUILD_VERSION" > "$version_file"
        # Create a variable named __version__ at the end of __init__.py
        echo "__version__ = \"$BUILD_VERSION\"" >> "$init_file"
    else
        echo "Error: BUILD_VERSION environment variable is not set or empty."
        exit 1
    fi
}


# build_vllm
build_monarch
# build_torchstore
append_date
