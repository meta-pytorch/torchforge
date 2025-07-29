# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from .collate import collate_packed

CROSS_ENTROPY_IGNORE_IDX = -100

__all__ = [
    "collate_packed",
]
