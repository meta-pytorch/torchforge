# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""SSHLauncher"""

# from .launcher import get_meshes_from_config
import atexit
import collections
import os
import re
import sys
from dataclasses import dataclass, field
from typing import List

from forge.types import LauncherConfig
from monarch._src.job.job import SSHJob
from monarch.actor import HostMesh, ProcMesh

from .base import BaseLauncher


def fetch_hostfile(hostfile_path):
    # e.g., worker-0 slots=16
    with open(hostfile_path, "r") as fd:
        hostfile_text = fd.readlines()

    return parse_hostfile(hostfile_text)


def parse_hostfile(hostfile_lines):
    # Regex matches one or more non-whitespace characters (\S+) at the start of
    # the line, followed by one or more whitespace characters (\s+), followed
    # by the string "slots=", followed by one or more digits (\d+).
    pattern = r"^(\S+)\s+slots=(\d+)"

    resource_pool = collections.OrderedDict()

    for line in hostfile_lines:
        line = line.strip()
        match = re.search(pattern, line)
        if line.startswith("#") or line == "":
            # hostfile comment or empty line, ignore
            continue
        elif match:
            host = match.group(1)
            num_slots = int(match.group(2))
            if host in resource_pool:
                raise ValueError(
                    f"Hostfile contains multiple entries for {host}, unable to proceed with launching"
                )
            resource_pool[host] = num_slots
        else:
            raise ValueError(
                f"Hostfile contains a bad entry: {line}, unable to proceed with launching"
            )

    if len(resource_pool) == 0:
        raise ValueError(
            "Hostfile is empty or not formatted correctly, unable to proceed with launching."
        )

    return resource_pool


@dataclass
class SSHForgeProc:
    is_actor: bool = True
    is_colocate: bool = False
    num_hosts: int = 0


@dataclass
class SSHActor:
    name: str | None = None
    host_mesh: HostMesh | None = None

    def get_host_mesh(self):
        return self.host_mesh


@dataclass
class SSHService:
    name: str | None = None
    host_mesh_list: List[HostMesh] = field(default_factory=list)
    current_idx: int = 0

    def get_host_mesh(self):
        host_mesh = self.host_mesh_list[self.current_idx]
        self.current_idx = (self.current_idx + 1) % len(self.host_mesh_list)
        return host_mesh


class SSHLauncher(BaseLauncher):
    def __init__(self, cfg: LauncherConfig):
        self.cfg = cfg
        self.job = None
        self.state = None
        self.ssh_hostfile = cfg.ssh_hostfile
        self.host_pool = fetch_hostfile(self.ssh_hostfile)
        self.host_ips = list(self.host_pool.keys())
        self.monarch_port = cfg.monarch_port

        self.proc_meshes = self.cfg.get_meshes()
        self.colocate_procs = [p for p in self.cfg.colocate if p in self.proc_meshes]
        self.forge_proc_map = self._create_forge_proc_map(cfg)

        required_host_count = sum(
            [
                proc.num_hosts
                for proc in self.forge_proc_map.values()
                if not proc.is_colocate
            ]
        )
        colocated_host_count = 0
        if len(self.colocate_procs) > 0:
            colocated_host_count = max(
                [
                    proc.num_hosts
                    for proc in self.forge_proc_map.values()
                    if proc.is_colocate
                ]
            )
            required_host_count += colocated_host_count

        self.host_ip_map = {}
        slice_start = 0
        for key, proc in self.forge_proc_map.items():
            if proc.is_colocate:
                # colocated procs share the last block of hosts
                self.host_ip_map[key] = self.host_ips[-colocated_host_count:]
            else:
                self.host_ip_map[key] = self.host_ips[slice_start : proc.num_hosts]
                slice_start += proc.num_hosts

        self.host_mesh_map = dict()

    def _create_forge_proc_map(self, cfg: LauncherConfig) -> dict:
        proc_map = {}
        for key in self.proc_meshes.keys():
            is_colocate = key in self.colocate_procs
            if key in cfg.services:
                proc_map[key] = SSHForgeProc(
                    is_actor=False,
                    is_colocate=is_colocate,
                    num_hosts=cfg.services[key].hosts,
                )
            else:
                proc_map[key] = SSHForgeProc(
                    is_actor=True,
                    is_colocate=is_colocate,
                    num_hosts=cfg.actors[key].hosts,
                )

        return proc_map

    async def _setup_job(self):
        job = SSHJob(
            python_exe=os.getenv("PYTHON_EXE", sys.executable),
            ssh_args=[
                "-o",
                "StrictHostKeyChecking=no",
                "-o",
                "BatchMode=yes",
            ],
            monarch_port=self.monarch_port,  # make sure this is open
        )

        for proc_key in self.proc_meshes.keys():
            job.add_mesh(f"{proc_key}", self.host_ip_map[proc_key])

        return job

    async def get_host_mesh(self, name: str, num_hosts: int) -> tuple[HostMesh, str]:
        for key in self.host_mesh_map.keys():
            if key in name:
                host_mesh = self.host_mesh_map[key].get_host_mesh()
                return host_mesh, None

        return None, None

    def _create_actor_or_service(self, mesh_name, job_host_mesh):
        if self.forge_proc_map[mesh_name].is_actor:
            return SSHActor(name=mesh_name, host_mesh=job_host_mesh)

        return SSHService(
            name=mesh_name,
            host_mesh_list=[
                job_host_mesh.slice(hosts=slice(i, i + 1))
                for i in range(len(self.host_ip_map[mesh_name]))
            ],
        )

    async def initialize(self) -> None:
        self.job = await self._setup_job()
        self.state = self.job.state()

        for proc_key in self.forge_proc_map.keys():
            self.host_mesh_map[proc_key] = self._create_actor_or_service(
                mesh_name=proc_key, job_host_mesh=getattr(self.state, proc_key)
            )

        # Register cleanup handler
        atexit.register(self.job.kill)

        print(f"ssh_launcher[init]: {self.ssh_hostfile=} {self.host_ips=}")
        print(f"ssh_launcher[init]: {self.host_ip_map=}")
        return self.job, self.state

    async def remote_setup(self, procs: ProcMesh) -> None:
        return

    @classmethod
    def validate_configuration(cls, cfg: LauncherConfig) -> tuple[bool, str]:

        error_msg_prefix = "SSH Launcher configuration error:"
        # Check that a valid hostfile is specified.
        if cfg.ssh_hostfile is None or not os.path.isfile(cfg.ssh_hostfile):
            error_msg = f"{error_msg_prefix} Invalid hostfile path {cfg.ssh_hostfile}."
            return False, error_msg

        # Check that all services/actors which use gpus also specify host count
        gpu_host_count = {
            name: service.hosts
            for name, service in cfg.services.items()
            if service.with_gpus
        }
        gpu_host_count.update(
            {name: actor.hosts for name, actor in cfg.actors.items() if actor.with_gpus}
        )
        missing_hosts = [
            name for name, count in gpu_host_count.items() if count is None or count < 1
        ]
        if len(missing_hosts) > 0:
            error_msg = f"{error_msg_prefix} missing host counts for services/actors that need gpus {missing_hosts}"
            return False, error_msg

        # Check that sufficient hosts available
        available_host_count = len(fetch_hostfile(cfg.ssh_hostfile))
        colocate_hosts = [
            host for host in gpu_host_count.keys() if host in cfg.colocate
        ]
        required_host_count = sum(list(gpu_host_count.values()))
        required_host_count = sum(
            [
                count
                for name, count in gpu_host_count.items()
                if name not in cfg.colocate
            ]
        )
        if len(colocate_hosts) > 0:
            required_host_count += max(
                [
                    count
                    for name, count in gpu_host_count.items()
                    if name in cfg.colocate
                ]
            )

        if required_host_count > available_host_count:
            error_msg = f"{error_msg_prefix} {required_host_count=} exceeds {available_host_count=}"
            return False, error_msg

        return True, None
