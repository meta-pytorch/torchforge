#!/bin/bash
set -euo pipefail

# Script runs relative to forge root
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERSIONS_FILE="$CURRENT_DIR/../../assets/versions.sh"
source "$VERSIONS_FILE"

echo "Installing nightly dependencies for forge build"
echo "PyTorch nightly already installed by test-infra via channel: nightly"

# 1. Verify PyTorch nightly is installed
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"

# 2. Install torchtitan nightly
echo "Installing torchtitan nightly..."
pip install --pre torchtitan \
  --index-url https://download.pytorch.org/whl/nightly/cu128

# 3. Install torchmonarch-nightly
echo "Installing torchmonarch-nightly..."
pip install torchmonarch-nightly

# 4. Install torchstore from main branch WITHOUT dependencies
# Following monarch_forge.sh:580-588 pattern
echo "Installing torchstore dependencies..."
pip install pygtrie

echo "Installing torchstore from main branch..."
git clone https://github.com/pytorch/torchstore.git /tmp/torchstore
cd /tmp/torchstore
git checkout main
pip install --no-deps .
cd -

# 5. Build vLLM from source (following internal pt2.sh:561-578 pattern)
# Note: Cannot use pip install vllm==0.10.0 because PyPI version requires torch==2.7.0
# vLLM has C++/CUDA extensions that must compile against our PyTorch nightly
echo "Building vLLM ${VLLM_VERSION} from source against PyTorch nightly..."
BUILD_DIR="/tmp/vllm-build"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

git clone https://github.com/vllm-project/vllm.git --branch "$VLLM_VERSION"
cd vllm

# Use existing torch (PyTorch nightly already installed)
# This script patches vLLM's setup.py to use the installed PyTorch instead of downloading
python use_existing_torch.py
pip install -r requirements/build.txt

# Clean up existing builds if needed
rm -rf build/ *.egg-info/

# Build and install vLLM (compiles C++/CUDA extensions against installed PyTorch)
pip install --no-build-isolation .

cd -

# 6. Set nightly version in __init__.py
echo "Setting nightly version..."
NIGHTLY_VERSION=$(date +%Y.%m.%d)
echo "__version__ = \"${NIGHTLY_VERSION}\"" > src/forge/__init__.py

echo "Nightly dependency installation complete!"
echo "Dependency versions:"
python -c "import torch, torchtitan, vllm; print(f'torch: {torch.__version__}, torchtitan: {torchtitan.__version__}, vllm: {vllm.__version__}')"
