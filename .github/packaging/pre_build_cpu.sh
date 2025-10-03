#!/bin/bash
set -euxo pipefail

# Builds vLLM, Monarch and torchstore
# This script builds vLLM, Monarch and torchstore and places
# their wheels into dist/.

VLLM_BRANCH="v0.10.0"
BUILD_DIR="$HOME/forge-build"

# Push other files to the dist folder
WHL_DIR="${GITHUB_WORKSPACE}/wheels/dist"

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
    export VERBOSE=1
    export CMAKE_VERBOSE_MAKEFILE=1
    export FORCE_CMAKE=1
    pip wheel -v --no-build-isolation --no-deps . -w "$WHL_DIR"
}

build_debug() {
    cd "$BUILD_DIR"

    git clone https://github.com/meta-pytorch/torchtune.git
    cd "$BUILD_DIR/torchtune"

    pip install -r requirements/build.txt
    pip wheel -v --no-build-isolation --no-deps . -w "$WHL_DIR"
}

build_debug
# build_vllm