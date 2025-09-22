# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import TypeVar, Optional

T = TypeVar('T')

def assert_not_none(value: Optional[T], name: str = "Value") -> T:
    """
    Asserts that a value is not None and returns the value.
    Raises a ValueError if the value is None.
    """
    if value is None:
        raise ValueError(f"{name} cannot be None.")
    return value
