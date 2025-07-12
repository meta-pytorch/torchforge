# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass


# This is basically empty, but I figure we'll want a lot of configuration classes.
@dataclass
class CollectorConfig:
    max_collector_steps: int
