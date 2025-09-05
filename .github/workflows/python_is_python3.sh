# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Check if 'python' command exists
if ! command -v python &> /dev/null; then
    # If not, alias python to python3
    alias python=python3
    echo "Aliased 'python' to 'python3'"
else
    echo "'python' already exists: $(command -v python)"
fi
