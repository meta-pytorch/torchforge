# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Implements an actor responsible for tracking and assigning GPU devices on HostMesh."""

import logging

# from monarch.actor import proc_mesh

from forge.controller import ForgeActor

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class GpuManager(ForgeActor):
    def __init__(self):
        self._host_resource_map = {}


def _get_gpu_manager() -> GpuManager:
    pass


def _spawn_gpu_manager() -> GpuManager:
    pass
