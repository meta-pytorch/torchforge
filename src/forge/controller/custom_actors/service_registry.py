# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Implements an actor that tracks all services runinng in the workload."""

from monarch.actor import endpoint

from forge.controller import ForgeActor


class ServiceRegistry(ForgeActor):
    def __init__(self):
        pass

    @endpoint
    def register(self):
        pass
