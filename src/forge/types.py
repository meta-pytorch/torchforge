# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, TypedDict, Union


class Message(TypedDict):
    role: str
    content: str | dict[str, Any]
    tools: dict[str, Any] | None


@dataclass(kw_only=True)
class Observation:
    """Base class for environment observations.

    Contract:
    - Should contain all information needed by an agent to make decisions
    - Should be serializable/deserializable
    - Should be immutable (or treated as such)
    Args:
        done: Whether the episode/conversation is complete
        reward: Optional reward signal (can be boolean, int, or float)
        metadata: Additional data that doesn't affect agent decisions but may be useful
                 for transforms, logging, evaluation, etc.
    """

    done: bool = False
    reward: bool | int | float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class Launcher(Enum):
    MAST = "mast"
    SLURM = "slurm"


@dataclass
class ProcessConfig:
    """A configuration for allocating Monarch ProcMeshes.

    Args:
        procs (int): Number of processes to launch for each replica of the service.
        with_gpus (bool, optional): Whether to allocate GPUs for the service processes.
        hosts (int | None, optional): Number of hosts to allocate for each replica.
            If this is set to None, it will use the local host.
            If this is set to a positive integer, it will run on a remote host.
        mesh_name (str | None, optional): Name of the mesh to use for the proc_mesh.

    """

    procs: int = 1
    with_gpus: bool = False
    hosts: int | None = None
    mesh_name: str | None = None


@dataclass
class ServiceConfig:
    """The configuration for a Forge service.

    Args:
        procs (int): Number of processes to launch for each replica of the service.
        num_replicas (int): Number of replicas to launch for the service.
        with_gpus (bool, optional): Whether to allocate GPUs for the service processes.
        hosts (int | None, optional): Number of hosts to allocate for each replica.
            If this is set to None, it will use the local host.
            If this is set to a positive integer, it will run on a remote host.
        health_poll_rate (float, optional): Frequency (in seconds) to poll for health status.
        replica_max_concurrent_requests (int, optional): Maximum number of concurrent requests per replica.
        return_first_rank_result (bool, optional): Whether to auto-unwrap ValueMesh to the first rank's result.
    """

    procs: int
    num_replicas: int
    with_gpus: bool = False
    hosts: int | None = None
    health_poll_rate: float = 0.2
    replica_max_concurrent_requests: int = 10
    return_first_rank_result: bool = True
    mesh_name: str | None = None

    def to_process_config(self) -> ProcessConfig:
        """Extract ProcessConfig from this ServiceConfig.

        Maps procs to procs for ProcessConfig.
        """
        return ProcessConfig(
            procs=self.procs,
            with_gpus=self.with_gpus,
            hosts=self.hosts,
            mesh_name=self.mesh_name,
        )


Scalar = Union[int, float]


@dataclass
class LauncherConfig:
    """A launcher config for the scheduler."""

    launcher: Launcher
    job_name: str = ""
    services: dict[str, ServiceConfig] = field(default_factory=dict)
    actors: dict[str, ProcessConfig] = field(default_factory=dict)
    slurm_args: dict[str, str] = field(default_factory=dict)
    cpus_per_task: int | None = None  # CPUs per node (SLURM param, can get with sinfo)
    mem: int | None = (  # noqa: N815
        None  # Memory per node (SLURM param, can get with sinfo)
    )
    gpus_per_node: int = 8  # GPUs per node (SLURM param, can get with sinfo)

    def __post_init__(self):
        if isinstance(self.launcher, str):
            self.launcher = Launcher(self.launcher)


@dataclass
class ProvisionerConfig:
    """A config for the forge provisioner."""

    launcher_config: LauncherConfig
