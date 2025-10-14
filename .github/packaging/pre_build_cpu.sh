#!/bin/bash
set -euxo pipefail

# Builds vLLM
# This script builds vLLM and places its wheel into dist/.

# Script runs relative to forge root
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "current dir is $CURRENT_DIR"
VERSIONS_FILE="$CURRENT_DIR/../../assets/versions.sh"
echo "versions file is $VERSIONS_FILE"
source "$VERSIONS_FILE"

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

build_vllm