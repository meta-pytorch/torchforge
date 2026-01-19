# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Launcher specific logic (i.e. SLURM, k8s when supported, etc.)"""

import atexit
import logging

from forge.controller.base import BaseLauncher
from forge.types import Launcher, LauncherConfig
from monarch._rust_bindings.monarch_hyperactor.channel import ChannelTransport
from monarch._rust_bindings.monarch_hyperactor.config import configure
from monarch.actor import ProcMesh
from monarch.job import JobState, JobTrait, SlurmJob

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


JOB_NAME_KEY = "job_name"
LAUNCHER_KEY = "launcher"


def get_meshes_from_config(cfg: LauncherConfig) -> dict[str, int]:
    """Extract mesh requirements from launcher config.

    Args:
        cfg: The launcher configuration

    Returns:
        Dictionary mapping mesh names to number of hosts required
    """
    meshes: dict[str, int] = {}

    # Add services that need remote hosts
    for service_name, service_cfg in cfg.services.items():
        hosts = getattr(service_cfg, "hosts", None)
        if hosts and hosts > 0:
            mesh_name = service_cfg.mesh_name or service_name
            meshes[mesh_name] = hosts

    # Add actors that need remote hosts
    for actor_name, actor_cfg in cfg.actors.items():
        hosts = getattr(actor_cfg, "hosts", None)
        if hosts and hosts > 0:
            mesh_name = actor_cfg.mesh_name or actor_name
            meshes[mesh_name] = hosts

    return meshes


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
        # HostMesh currently requires explicit configuration
        # of the underlying transport from client to mesh.
        # This can be removed in the future once this has been removed.
        configure(default_transport=ChannelTransport.TcpWithHostname)

        # Collect all mesh requirements from config
        meshes = get_meshes_from_config(self.cfg)

        # If no remote resources needed, skip job creation
        if not meshes:
            return

        # Build slurm_args from config
        slurm_args = [f"--{key}={value}" for key, value in self.cfg.slurm_args.items()]

        # Create a single SlurmJob with all meshes
        logger.info(f"Creating SlurmJob with meshes: {meshes}")
        job = SlurmJob(
            meshes=meshes,  # e.g., {"generator": 1, "trainer": 2, "ref_model": 1}
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


def get_launcher(cfg: LauncherConfig | None = None) -> BaseLauncher | None:
    if not cfg:
        return None
    if cfg.launcher == Launcher.SLURM:
        return Slurmlauncher(cfg)
    elif cfg.launcher == Launcher.MAST:
        try:
            from forge.fb.mast_launcher import MastLauncher

            return MastLauncher(cfg, detached=False)
        except ImportError as err:
            raise ValueError("MAST is not available, cannot launch MAST jobs.") from err

    else:
        raise ValueError(f"Unsupported config provided, got {cfg}")
