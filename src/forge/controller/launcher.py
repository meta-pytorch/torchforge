# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Launcher specific logic (i.e. SLURM, k8s when supported, etc.)"""

import atexit
import logging
from typing import Any

from forge.controller.base import BaseLauncher
from forge.types import Launcher, LauncherConfig
from monarch._rust_bindings.monarch_hyperactor.channel import ChannelTransport
from monarch._rust_bindings.monarch_hyperactor.config import configure
from monarch.actor import HostMesh, ProcMesh
from monarch.job import SlurmJob

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


JOB_NAME_KEY = "job_name"
LAUNCHER_KEY = "launcher"


class Slurmlauncher(BaseLauncher):
    def __init__(
        self,
        cfg: LauncherConfig,
    ):
        self.cfg = cfg
        self._job: SlurmJob | None = None
        self._host_meshes: dict[str, HostMesh] = {}  # mesh_name -> HostMesh

    async def initialize(self) -> None:
        """Initialize the launcher and create a single SlurmJob for all resources.

        This pre-allocates all meshes defined in the config in one Slurm job.
        """
        # HostMesh currently requires explicit configuration
        # of the underlying transport from client to mesh.
        # This can be removed in the future once this has been removed.
        configure(default_transport=ChannelTransport.TcpWithHostname)

        # Collect all mesh requirements from config
        meshes = {}

        # Add services that need remote hosts
        for service_name, service_cfg in self.cfg.services.items():
            # Use getattr to safely access hosts (might not be defined)
            hosts = getattr(service_cfg, "hosts", None)
            if hosts and hosts > 0:
                mesh_name = service_cfg.mesh_name or service_name
                meshes[mesh_name] = hosts

        # Add actors that need remote hosts
        for actor_name, actor_cfg in self.cfg.actors.items():
            # Use getattr to safely access hosts (might not be defined)
            hosts = getattr(actor_cfg, "hosts", None)
            if hosts and hosts > 0:
                mesh_name = actor_cfg.mesh_name or actor_name
                meshes[mesh_name] = hosts

        # If no remote resources needed, skip job creation
        if not meshes:
            return

        # Prepare slurm_args from config (only for args without dedicated parameters)
        slurm_args = []
        if self.cfg.account:
            slurm_args.append(f"--account={self.cfg.account}")
        if self.cfg.qos:
            slurm_args.append(f"--qos={self.cfg.qos}")

        # Prepare resource parameters
        # Convert memMB to format expected by SlurmJob (string like "500G" or "2047962M")
        mem = None
        if hasattr(self.cfg, "memMB") and self.cfg.memMB:
            mem = f"{self.cfg.memMB}M"

        cpus_per_task = None
        if hasattr(self.cfg, "cpu") and self.cfg.cpu:
            cpus_per_task = self.cfg.cpu

        # Create a single SlurmJob with all meshes
        logger.info(f"Creating SlurmJob with meshes: {meshes}")
        self._job = SlurmJob(
            meshes=meshes,  # e.g., {"generator": 1, "trainer": 2, "ref_model": 1}
            gpus_per_node=self.cfg.gpu,
            cpus_per_task=cpus_per_task,
            mem=mem,
            time_limit="72:00:00",  # Default to 72 hours
            job_name=self.cfg.job_name + "_workers" or "forge_job",
            slurm_args=slurm_args,
        )

        # Apply the job to allocate resources
        logger.info("Submitting SlurmJob...")
        self._job.apply()
        logger.info("SlurmJob submitted, waiting for allocation...")

        # Register cleanup handler
        atexit.register(self._job.kill)

        # Get the job state and extract all HostMeshes
        logger.info("Getting job state (this will block until nodes are allocated)...")
        job_state = self._job.state(cached_path=None)
        logger.info(
            f"Job state received! Extracting HostMeshes for {list(meshes.keys())}"
        )

        # Store all HostMeshes by name (like node0_host, node1_host in the example)
        for mesh_name in meshes.keys():
            host_mesh: HostMesh = getattr(job_state, mesh_name)
            self._host_meshes[mesh_name] = host_mesh
            logger.info(f"HostMesh '{mesh_name}' extracted and stored")

        logger.info(
            f"SlurmLauncher initialization complete. {len(self._host_meshes)} HostMeshes ready."
        )

    async def get_allocator(self, name: str, num_hosts: int) -> tuple[Any, Any, str]:
        """Return a pre-allocated HostMesh for the given mesh name.

        Args:
            name: The name of the mesh (may include replica suffix like "generator_0")
            num_hosts: Expected number of hosts (for validation)

        Returns:
            A tuple of (HostMesh, SlurmJob, job_name) where:
            - HostMesh is the pre-allocated resource
            - SlurmJob is the allocation handle for cleanup
            - job_name is for tracking/logging
        """
        # Strip replica suffix (e.g., "generator_0" -> "generator")
        # Services append _{replica_idx} to mesh names
        base_name = name
        if "_" in name:
            parts = name.rsplit("_", 1)
            if len(parts) == 2 and parts[1].isdigit():
                base_name = parts[0]

        if base_name not in self._host_meshes:
            raise RuntimeError(
                f"Mesh '{name}' (base: '{base_name}') was not pre-allocated. "
                f"Available meshes: {list(self._host_meshes.keys())}. "
                f"Make sure the mesh is defined in the launcher config."
            )

        host_mesh = self._host_meshes[base_name]

        # Return (HostMesh, SlurmJob handle, job_name)
        return host_mesh, self._job, self.cfg.job_name or "forge_job"

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
