#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Apply OpenEnv modifications for blackjack training."""

import subprocess
import sys
from pathlib import Path


def main():
    # Get script directory
    script_dir = Path(__file__).parent
    patch_file = script_dir / "openenv_blackjack.patch"

    if not patch_file.exists():
        print(f"Error: Patch file not found at {patch_file}")
        sys.exit(1)

    # Apply patch
    try:
        subprocess.run(
            ["git", "apply", str(patch_file)],
            check=True,
            capture_output=True,
            text=True,
        )
        print("✓ Patch applied successfully")
    except subprocess.CalledProcessError as e:
        print(f"Error applying patch: {e.stderr}")
        sys.exit(1)


if __name__ == "__main__":
    main()
