# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch


def get_world_size_and_rank() -> tuple[int, int]:
    """Function that gets the current world size (aka total number
    of ranks) and rank number of the current process in the default process group.

    Returns:
        tuple[int, int]: world size, rank
    """
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_world_size(), torch.distributed.get_rank()
    else:
        return 1, 0
