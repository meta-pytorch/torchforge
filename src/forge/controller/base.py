# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any

from monarch.actor import ProcMesh


class BaseLauncher:
    async def initialize(self) -> None:
        pass

    async def get_allocator(self, name: str, num_hosts: int) -> tuple[Any, Any, str]:
        pass

    async def remote_setup(self, procs: ProcMesh) -> None:
        pass
