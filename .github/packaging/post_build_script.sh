#!/bin/bash
set -euxo pipefail

FORGE_WHEEL=${GITHUB_WORKSPACE}/${REPOSITORY}/dist/*.whl
WHL_DIR="${GITHUB_WORKSPACE}/wheels/dist"

echo "Uploading wheels to S3"
ls -l "${WHL_DIR}"
ls ${FORGE_WHEEL}
