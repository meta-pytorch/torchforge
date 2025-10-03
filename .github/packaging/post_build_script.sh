#!/bin/bash
set -euxo pipefail

FORGE_WHEEL=${GITHUB_WORKSPACE}/${REPOSITORY}/dist/*.whl
WHL_DIR="${GITHUB_WORKSPACE}/wheels/"
DIST=dist/

echo "Uploading wheels to S3"
ls -l "${WHL_DIR}"
ls ${FORGE_WHEEL}
echo "Copying files from $WHL_DIR to $DIST"
mkdir -p $DIST && cp -r $WHL_DIR $DIST
ls -l "${DIST}"