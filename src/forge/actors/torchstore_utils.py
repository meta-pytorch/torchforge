# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

KEY_DELIM = "."


def get_param_prefix(policy_version: int) -> str:
    return f"policy_ver_{policy_version}"


def get_param_key(policy_version: int, name: str) -> str:
    return f"policy_ver_{policy_version}{KEY_DELIM}{name}"


def extract_param_name(key: str) -> str:
    return KEY_DELIM.join(key.split(KEY_DELIM)[1:])
