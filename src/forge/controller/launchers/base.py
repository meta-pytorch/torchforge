# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from monarch.actor import ProcMesh
from monarch.job import JobState, JobTrait


class BaseLauncher:
    async def initialize(self) -> tuple[JobTrait, JobState]:
        """Initialize the launcher and allocate resources.

        This method is called once during provisioner initialization to set up
        the launcher and pre-allocate any required resources (e.g., compute nodes).

        Note this should just be a thin wrapper over the Monarch Jobs API. Since different
        scheduler's have slightly different APIs for allocating resources, we need this thin wrapper on top.
        """
        pass

    async def remote_setup(self, procs: ProcMesh) -> None:
        """Perform any launcher-specific setup for remote processes.

        Args:
            procs: The process mesh to configure.
        """
        pass
