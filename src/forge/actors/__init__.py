# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

__all__ = ["Policy", "PolicyRouter"]


def __getattr__(name):
    if name == "Policy":
        from .policy import Policy

        return Policy
    if name == "PolicyRouter":
        from .policy import PolicyRouter

        return PolicyRouter
    raise AttributeError("fmodule {__name__} has no attribute {name}")
