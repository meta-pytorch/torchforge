#!/bin/bash
set -euxo pipefail

FORGE_WHEEL=${GITHUB_WORKSPACE}/${REPOSITORY}/dist/*.whl
WHL_DIR="${GITHUB_WORKSPACE}/wheels/dist"
DIST=dist/

echo "Uploading wheels to S3"
ls -l "${WHL_DIR}"
ls ${FORGE_WHEEL}
echo "Copying files from $WHL_DIR to $DIST"
mkdir -p $DIST && find "$WHL_DIR" -maxdepth 1 -type f -exec cp {} "$DIST/" \;
ls -l "${DIST}"
