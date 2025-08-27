# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

__all__ = ["collate_packed", "CROSS_ENTROPY_IGNORE_IDX"]


def __getattr__(name):
    if name == "collate_packed":
        from .collate import collate_packed

        return collate_packed
    elif name == "CROSS_ENTROPY_IGNORE_IDX":
        from .utils import CROSS_ENTROPY_IGNORE_IDX

        return CROSS_ENTROPY_IGNORE_IDX
    else:
        raise AttributeError(f"module {__name__} has no attribute {name}")
