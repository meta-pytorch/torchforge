# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

KEY_DELIM = "."

# Alternate between two storage version IDs
# This reuses allocations instead of incrementing versions and deleting old ones
VERSION_A = 0
VERSION_B = 1


def get_storage_version(step: int) -> int:
    """Map incrementing step to ping-pong storage version (0 or 1)."""
    return VERSION_A if step % 2 == 0 else VERSION_B


def get_param_prefix(policy_version: int) -> str:
    storage_version = get_storage_version(policy_version)
    return f"policy_ver_{storage_version:010d}"


def get_param_key(policy_version: int, name: str) -> str:
    storage_version = get_storage_version(policy_version)
    return f"policy_ver_{storage_version:010d}{KEY_DELIM}{name}"


def extract_param_name(key: str) -> str:
    return KEY_DELIM.join(key.split(KEY_DELIM)[1:])
