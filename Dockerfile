# ================================================================================================
# Stage 1: Download vLLM wheel and build torchtitan wheel
# ================================================================================================
FROM ubuntu:22.04 AS wheel-downloader

# Install tools needed for downloading and building wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    jq \
    ca-certificates \
    git \
    python3.10 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Download vLLM wheel from GitHub releases
ARG GITHUB_REPO="meta-pytorch/forge"
ARG RELEASE_TAG="v0.0.0-93025"
ARG TORCHTITAN_COMMIT="0cfbd0b3c2d827af629a107a77a9e47229c31663"  # From assets/versions.sh - compatible with PyTorch 2.9

WORKDIR /tmp/download

# Download vLLM wheel
RUN echo "Fetching vLLM wheel from GitHub release ${RELEASE_TAG}..." && \
    # Get the release information from GitHub API
    RELEASE_INFO=$(curl -s "https://api.github.com/repos/${GITHUB_REPO}/releases/tags/${RELEASE_TAG}") && \
    # Extract the vLLM wheel download URL
    VLLM_URL=$(echo "$RELEASE_INFO" | jq -r '.assets[] | select(.name | contains("vllm")) | .browser_download_url' | head -1) && \
    VLLM_NAME=$(echo "$RELEASE_INFO" | jq -r '.assets[] | select(.name | contains("vllm")) | .name' | head -1) && \
    echo "Downloading: $VLLM_NAME" && \
    echo "URL: $VLLM_URL" && \
    # Download the wheel
    curl -L -o "/tmp/download/${VLLM_NAME}" "${VLLM_URL}" && \
    echo "vLLM download complete"

# Build torchtitan wheel from specific commit
RUN echo "Building torchtitan from commit ${TORCHTITAN_COMMIT}..." && \
    cd /tmp && \
    git clone https://github.com/pytorch/torchtitan.git && \
    cd torchtitan && \
    git checkout ${TORCHTITAN_COMMIT} && \
    python3 -m pip install --upgrade pip wheel && \
    pip wheel --no-deps . -w /tmp/download && \
    echo "torchtitan build complete: $(ls -lh /tmp/download/*.whl)"

# ================================================================================================
# Stage 2: Main application image
# ================================================================================================
FROM nvidia/cuda:12.9.1-base-ubuntu22.04

# Metadata labels
LABEL maintainer="PyTorch Team"
LABEL description="Forge - A PyTorch-native agentic RL library for post-training large language models"
LABEL cuda.version="12.9.1"
LABEL python.version="3.10"

# Set environment to avoid interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive

# ================================================================================================
# Install system dependencies
# ================================================================================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Build tools (needed for compiling Python extensions)
    build-essential \
    git \
    curl \
    ca-certificates \
    # RDMA/InfiniBand libraries for distributed training
    libibverbs1 \
    libibverbs-dev \
    rdma-core \
    libmlx5-1 \
    && rm -rf /var/lib/apt/lists/*

# ================================================================================================
# Set up CUDA environment variables
# ================================================================================================
# These environment variables match those set by cuda_env.sh in the installation script
ENV CUDA_VERSION=12.9
ENV CUDA_HOME=/usr/local/cuda-${CUDA_VERSION}
ENV NVCC=${CUDA_HOME}/bin/nvcc
ENV CUDA_NVCC_EXECUTABLE=${CUDA_HOME}/bin/nvcc
ENV CUDA_INCLUDE_DIRS=${CUDA_HOME}/include
ENV CUDA_CUDART_LIBRARY=${CUDA_HOME}/lib64/libcudart.so

# Add CUDA binaries to PATH
ENV PATH="${CUDA_HOME}/bin:${PATH}"

# Add CUDA compat libs to LD_LIBRARY_PATH
# This is critical for PyTorch and vLLM to find CUDA libraries
ENV LD_LIBRARY_PATH="${CUDA_HOME}/compat:${LD_LIBRARY_PATH}"

# Temporary flag required by Monarch
ENV MONARCH_HOST_MESH_V1_REMOVE_ME_BEFORE_RELEASE=1

# Create symlink if /usr/local/cuda-12.9 doesn't exist but /usr/local/cuda does
RUN if [ ! -d "/usr/local/cuda-12.9" ] && [ -d "/usr/local/cuda" ]; then \
    ln -s /usr/local/cuda /usr/local/cuda-12.9; \
    fi

# ================================================================================================
# Install uv package manager
# ================================================================================================
# Install uv - a fast Python package installer
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    # Make uv available in PATH
    ln -s /root/.local/bin/uv /usr/local/bin/uv && \
    ln -s /root/.local/bin/uvx /usr/local/bin/uvx

# Set working directory early so venv is created in project directory
WORKDIR /workspace

# Install Python 3.10 via uv (required by monarch wheel cp310)
RUN uv python install 3.10 && \
    # Create a virtual environment at project location using uv-managed Python 3.10
    uv venv --python 3.10 .venv

# Activate the virtual environment by adding it to PATH
ENV PATH="/workspace/.venv/bin:${PATH}"
ENV VIRTUAL_ENV="/workspace/.venv"

# Add uv-managed Python library path to LD_LIBRARY_PATH for monarch
# Monarch needs libpython3.10.so.1.0 which is provided by uv's Python installation
ENV LD_LIBRARY_PATH="/root/.local/share/uv/python/cpython-3.10.19-linux-x86_64-gnu/lib:${LD_LIBRARY_PATH}"

# ================================================================================================
# Install PyTorch nightly with uv
# ================================================================================================
# Install PyTorch nightly with CUDA 12.9 support
# This is a large download and should be in its own layer for caching
ARG PYTORCH_VERSION="2.9.0.dev20250905"
RUN uv pip install --no-cache \
    torch==${PYTORCH_VERSION} \
    --index-url https://download.pytorch.org/whl/nightly/cu129

# ================================================================================================
# Install pre-built wheels
# ================================================================================================
# Create temporary directory for wheels
RUN mkdir -p /tmp/wheels

# Copy local wheels from assets directory (excluding torchtitan - we build it fresh)
COPY assets/wheels/monarch*.whl assets/wheels/torchstore*.whl /tmp/wheels/

# Copy downloaded vLLM and built torchtitan wheels from stage 1
COPY --from=wheel-downloader /tmp/download/*.whl /tmp/wheels/

# Install all wheels using uv
# The wheels include: monarch, torchstore, freshly-built torchtitan, and vLLM
RUN uv pip install --no-cache /tmp/wheels/*.whl && \
    rm -rf /tmp/wheels

# ================================================================================================
# Install Forge
# ================================================================================================
# Copy the entire source tree
# .dockerignore will exclude unnecessary files
COPY . /workspace/

# Install Forge in production mode (not editable)
# This installs the forge package and its dependencies
RUN uv pip install --no-cache .

# ================================================================================================
# Final setup
# ================================================================================================
# Verify installations (basic import checks that don't require GPU)
# The virtual environment is activated via PATH, so python uses the venv
RUN python -c "import torch; print(f'PyTorch version: {torch.__version__}')" && \
    python -c "import vllm; print('vLLM imported successfully')" && \
    python -c "import forge; print('Forge imported successfully')"

# Set default command to bash for interactive use
# Users can override this when running the container
CMD ["/bin/bash"]
