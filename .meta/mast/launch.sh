#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# launch.sh - Launch MAST jobs with Forge
set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if config file is provided
if [ $# -eq 0 ]; then
    log_error "No config file provided"
    echo "Usage: $0 <config_file>"
    echo "Example: $0 .meta/mast/qwen3_1_7b_mast.yaml"
    exit 1
fi

CONFIG_FILE="$1"

# Generate a unique job name based on the config file name
BASENAME=$(basename "$CONFIG_FILE" .yaml)
RANDOM_SUFFIX=$(cat /dev/urandom | tr -dc 'a-z0-9' | fold -w 6 | head -n 1)
JOB_NAME="${BASENAME}-${RANDOM_SUFFIX}"
log_info "Generated job name: $JOB_NAME"

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Navigate to forge root (two levels up from .meta/mast/)
FORGE_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

log_info "Forge root directory: $FORGE_ROOT"
log_info "Config file: $CONFIG_FILE"

# Check if config file exists
if [ ! -f "$FORGE_ROOT/$CONFIG_FILE" ]; then
    log_error "Config file not found: $FORGE_ROOT/$CONFIG_FILE"
    exit 1
fi

# Navigate to forge root
cd "$FORGE_ROOT"
log_info "Changed to directory: $(pwd)"

# Reinstall forge package
log_info "Reinstalling forge package..."
pip install --force-reinstall --no-deps .
if [ $? -ne 0 ]; then
    log_error "Failed to reinstall forge package"
    exit 1
fi

log_info "Successfully reinstalled forge package"

# Launch the job
CHECKPOINT_FOLDER=/mnt/wsfuse/teamforge/forge_runs/$JOB_NAME
log_info "Launching MAST job..."

# Manually override the relevant checkpoint path(s)
# This unfortunately cannot be done in the YAML itself since this should be
# based on job name...
PYTHONPATH=. python .meta/mast/main.py --job-name $JOB_NAME --config $CONFIG_FILE trainer.checkpoint.folder=${CHECKPOINT_FOLDER} trainer.dcp_path=${CHECKPOINT_FOLDER}
