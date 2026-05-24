# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Slurm Launcher"""


import atexit
import logging

from .base import BaseLauncher
from forge.types import LauncherConfig
from monarch.actor import ProcMesh

from monarch.job import JobState, JobTrait, SlurmJob

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class Slurmlauncher(BaseLauncher):
    def __init__(
        self,
        cfg: LauncherConfig,
    ):
        self.cfg = cfg

    async def initialize(self) -> tuple[JobTrait, JobState]:
        """Initialize the launcher and create a single SlurmJob for all resources.

        This pre-allocates all meshes defined in the config in one Slurm job.

        Returns:
            A tuple of (job, job_state) containing the SlurmJob handle and its state.
        """
        # Collect all mesh requirements from config
        meshes = self.cfg.get_meshes()

        # If no remote resources needed, skip job creation
        if not meshes:
            return

        # Build slurm_args from config
        slurm_args = [f"--{key}={value}" for key, value in self.cfg.slurm_args.items()]

        # Create a single SlurmJob with all meshes
        logger.info(f"Creating SlurmJob with meshes: {meshes}")
        job = SlurmJob(
            meshes=meshes,  # e.g., {"generator_0": 1, "generator_1": 1, "trainer": 2}
            slurm_args=slurm_args,
            job_name=self.cfg.job_name + "_workers" or "forge_job",
            time_limit="72:00:00",  # Default to 72 hours
            gpus_per_node=self.cfg.gpus_per_node,
            cpus_per_task=self.cfg.cpus_per_task,
            mem=self.cfg.mem,
        )

        # Apply the job to allocate resources
        logger.info("Submitting SlurmJob...")
        job.apply()
        logger.info("SlurmJob submitted, waiting for allocation...")

        # Register cleanup handler
        atexit.register(job.kill)

        # Wait for job allocation
        logger.info("Getting job state (this will block until nodes are allocated)...")
        job_state = job.state(cached_path=None)

        logger.info("SlurmLauncher initialization complete.")
        return job, job_state

    async def remote_setup(self, procs: ProcMesh) -> None:
        return
