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
        """Get an allocator for the given mesh.

        Returns:
            A tuple of (allocation_resource, allocation_handle, allocation_name)
            - allocation_resource: The resource to use (e.g., HostMesh, allocator, etc.)
            - allocation_handle: Opaque handle for cleanup, passed back to cleanup_allocation
            - allocation_name: String name for tracking/logging
        """
        pass

    async def remote_setup(self, procs: ProcMesh) -> None:
        pass

    async def cleanup_all(self) -> None:
        """Clean up all allocations managed by this launcher."""
        pass
