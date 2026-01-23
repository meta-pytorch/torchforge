# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""vLLM utility functions for version detection.

TEMPORARY: This module exists to support both vLLM v0 (< 0.13.0) and v1 (>= 0.13.0)
during the transition period. Once v0 support is dropped, this module can be removed.
"""

from functools import lru_cache


@lru_cache(maxsize=1)
def get_vllm_version() -> str | None:
    """Get the installed vLLM version.

    Returns:
        Version string (e.g., "0.13.0") or None if vLLM is not installed.
    """
    try:
        import vllm

        return vllm.__version__
    except ImportError:
        return None


def use_generator_v1() -> bool:
    """Check if Generator v1 should be used (vLLM >= 0.13.0).

    TEMPORARY: This function exists to support both v0 and v1 Generator implementations.
    Once v0 support is dropped, this function can be removed and v1 used unconditionally.

    Returns:
        True if vLLM >= 0.13.0, False otherwise (including if vLLM is not installed).
    """
    version = get_vllm_version()
    if version is None:
        return False
    return version >= "0.13.0"
