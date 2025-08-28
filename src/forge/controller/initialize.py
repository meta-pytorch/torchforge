# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Forge-wide initializations."""

import logging

# from monarch.actor import proc_mesh


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def _get_gpu_allocator():
    pass


async def spawn_gpu_allocator():
    pass


async def spawn_service_registry():
    pass


async def initialize():
    pass


async def shutdown():
    pass
