#!/bin/bash

# Builds vLLM, Monarch and torchstore
# This script builds vLLM, Monarch and torchstore and places
# their wheels into dist/.

VLLM_BRANCH="v0.10.0"
MONARCH_COMMIT="main"
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
    cd "$BUILD_DIR"
    git clone https://github.com/meta-pytorch/monarch.git
    cd "$BUILD_DIR/monarch"
    git checkout $MONARCH_COMMIT

    pip install -r build-requirements.txt
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
build_torchstore
append_date
