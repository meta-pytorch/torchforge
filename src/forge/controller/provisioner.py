# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Remote and local resource manager for allocation and provisioning."""

import asyncio
import logging

import os
import socket
import uuid

import torch

from forge.controller.launcher import BaseLauncher, get_launcher
from forge.env import all_env_vars, FORGE_DISABLE_METRICS
from forge.types import ProcessConfig, ProvisionerConfig

from monarch._src.actor.actor_mesh import ActorMesh

from monarch.actor import (
    Actor,
    endpoint,
    HostMesh,
    ProcMesh,
    shutdown_context,
    this_host,
)

from monarch.utils import setup_env_for_distributed

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def _get_port() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("localhost", 0))
        addr = s.getsockname()
        port = addr[1]
        return str(port)


class _RemoteInfoFetcher(Actor):
    """An actor responsible for getting remote host information."""

    @endpoint
    def get_info(self) -> tuple[str, str]:
        """Returns hostname and port."""
        return socket.gethostname(), _get_port()

    @endpoint
    def get_gpu_count(self) -> int:
        """Returns the number of GPUs available on this host."""
        try:
            gpu_count = torch.cuda.device_count()
        except Exception:
            # If torch is not available or CUDA is not available, assume no GPUs
            gpu_count = 0
        return gpu_count


class EnvSetter(Actor):
    """Actor to set environment variables on each proc in a mesh.

    Ideally, this is handled in spawn_procs's bootstrap call which
    essentially does the same thing as we're doing here.

    However, Monarch's SetupActor currently fails to stop on shutdown
    which leads to zombie messages sent to the SetupActor. This is a
    known issue, and we will move back to bootstrap once it's fixed.

    We are able to avoid this here by properly awaiting the spawning
    of the actor.

    """

    @endpoint
    def set_env(self, env_vars: dict[str, str]):
        """Set environment variables on this proc.

        Args:
            env_vars: Dictionary of environment variables to set
        """
        import os
        import socket

        # Set VLLM_HOST_IP (required for vLLM on multiple nodes)
        os.environ["VLLM_HOST_IP"] = socket.gethostbyname(socket.getfqdn())

        # Set user-provided environment variables
        for k, v in env_vars.items():
            os.environ[k] = v


async def get_remote_info(host_mesh: HostMesh) -> tuple[str, str]:
    """Returns the host name and port of the host mesh."""
    throwaway_procs = host_mesh.spawn_procs(per_host={"procs": 1})
    fetcher = throwaway_procs.spawn("_fetcher", _RemoteInfoFetcher)

    # This will reduce something like extent = {"hosts": 2, "procs": 1} to
    # {"hosts": 1, "procs": 1}.
    singleton_slice = {k: slice(0, 1) for k in fetcher.extent.keys()}
    fetcher = fetcher.slice(**singleton_slice)
    # Fetcher should be a singleton at this point - call_one() will fail otherwise
    host, port = await fetcher.get_info.call_one()

    # Stopping this proc is the right thing to do, but Monarch does not yet handle manual stops well.
    # await throwaway_procs.stop()
    return host, port


async def get_host_gpus(host_mesh: HostMesh) -> int:
    """Returns the number of GPUs available on the host mesh."""
    throwaway_procs = host_mesh.spawn_procs(per_host={"procs": 1})
    fetcher = throwaway_procs.spawn("_gpu_counter", _RemoteInfoFetcher)

    # Reduce to a singleton
    singleton_slice = {k: slice(0, 1) for k in fetcher.extent.keys()}
    fetcher = fetcher.slice(**singleton_slice)

    gpu_count = await fetcher.get_gpu_count.call_one()
    return gpu_count


async def set_environment(proc_mesh: ProcMesh, env_vars: dict[str, str]):
    """Set environment variables on a proc mesh using EnvSetter actor.

    This replaces the old bootstrap approach to avoid Monarch's SetupActor
    mesh failures on shutdown.

    Args:
        proc_mesh: The proc mesh to set environment variables on
        env_vars: Dictionary of environment variables to set
    """
    env_setter = proc_mesh.spawn("_env_setter", EnvSetter)
    await env_setter.set_env.call(env_vars)


class GpuManager:
    """Tracks and assigns GPU devices on a host.

    Args:
        available_devices: Set of GPU device IDs to manage. If None, uses all devices from 0 to max_device_count-1.
        max_device_count: Maximum number of GPU devices on this host. Defaults to 8.

    """

    def __init__(
        self, available_devices: set[int] | None = None, max_device_count: int = 8
    ):
        if available_devices is None:
            available_devices = set(range(0, max_device_count))
        else:
            # Validate types first
            assert all(
                isinstance(x, int) for x in available_devices
            ), f"All device IDs must be integers, got: {available_devices}"
            # When available_devices is provided (e.g., from CUDA_VISIBLE_DEVICES),
            # adjust max_device_count to accommodate the highest device ID
            if available_devices:
                max_device_count = max(max(available_devices) + 1, max_device_count)

        assert all(
            isinstance(x, int) for x in available_devices
        ), f"All device IDs must be integers, got: {available_devices}"
        assert all(
            x >= 0 for x in available_devices
        ), f"All device IDs must be non-negative, got: {available_devices}"
        self.available_gpus = available_devices
        self.max_device_count = max_device_count

    def get_available_gpus(self) -> list[str]:
        """Returns a list of available GPU devices."""
        return [str(gpu) for gpu in self.available_gpus]

    def get_gpus(self, num_gpus: int) -> list[str]:
        """Assigns GPU devices."""
        if num_gpus > len(self.available_gpus):
            raise RuntimeError("Not enough GPUs available")
        gpus = list(self.available_gpus)[:num_gpus]
        self.available_gpus -= set(gpus)
        return [str(gpu) for gpu in gpus]

    def release_gpus(self, gpu_ids: list[str]) -> None:
        """Releases the given GPU devices."""
        for gpu_id in gpu_ids:
            self.available_gpus.add(int(gpu_id))


class Provisioner:
    """A global resource provisioner."""

    def __init__(self, cfg: ProvisionerConfig | None = None):
        self._lock = asyncio.Lock()

        self._job: JobTrait | None = None
        # _job_state contains all the HostMeshes that were allocated as attributes, accessible by their name
        self._job_state: JobState | None = None

        # HostMeshes are currently not hashable, so
        # we generate a hash per HostMesh. We'll
        # remove this once this is supported in Monarch.
        self._this_host_id = uuid.uuid1()

        # For the local host, we may want to set CUDA_VISIBLE_DEVICES
        # for small scale testing. We inherit the environment's
        # CUDA_VISIBLE_DEVICES **only for the local host** and not
        # for remote hosts.
        available_local_devices = None
        cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", None)
        if cuda_visible_devices is not None and cuda_visible_devices.strip():
            try:
                available_local_devices = set(
                    int(x.strip()) for x in cuda_visible_devices.split(",") if x.strip()
                )
            except ValueError as e:
                raise ValueError(
                    f"Invalid CUDA_VISIBLE_DEVICES format: '{cuda_visible_devices}'. "
                    f"Expected comma-separated integers (e.g., '0,1,2'). Error: {e}"
                ) from e

        # Get the actual GPU count for the local host
        try:
            local_gpu_count = torch.cuda.device_count()
        except Exception:
            # If torch is not available or CUDA is not available, assume no GPUs
            local_gpu_count = 0

        self._host_gpu_map = {
            self._this_host_id: GpuManager(
                available_local_devices, max_device_count=local_gpu_count
            ),
        }
        self._proc_host_map = {}
        self.launcher: BaseLauncher | None = get_launcher(
            cfg.launcher_config if cfg is not None else None
        )
        if not self.launcher:
            logger.warning("Launcher not provided, allocations must run locally.")

        self._registered_actors: list["ForgeActor"] = []
        self._registered_services: list["ServiceInterface"] = []

    async def initialize(self):
        """Call this after creating the instance"""
        if self.launcher is not None:
            self._job, self._job_state = await self.launcher.initialize()

    async def get_host_mesh(self, name: str) -> HostMesh:
        """Get the pre-allocated HostMesh from launcher.

        Args:
            name: Name for the host mesh

        Returns:
            the allocated HostMesh ready to use
        """
        # no need to lock here because this is already locked behind `get_proc_mesh`
        if not self.launcher:
            raise RuntimeError(
                "You tried to create a remote allocation by specifying the number of hosts on an actor or service, "
                "but no launcher was specified."
            )
        logger.debug(f"Creating remote host mesh for {name}")

        # Strip replica suffix (e.g., "generator_0" -> "generator")
        # Services append _{replica_idx} to mesh names
        base_name = name
        if "_" in name:
            parts = name.rsplit("_", 1)
            if len(parts) == 2 and parts[1].isdigit():
                base_name = parts[0]

        # _job_state contains all the HostMeshes that were allocated as attributes, accessible by their name
        try:
            host_mesh = getattr(self._job_state, base_name)
        except AttributeError as e:
            raise RuntimeError(
                f"Mesh '{name}' (base: '{base_name}') was not pre-allocated. "
                "Make sure the mesh is defined in the launcher config."
            ) from e

        return host_mesh

    async def get_proc_mesh(
        self,
        num_procs: int,
        mesh_name: str,
        with_gpus: bool = False,
        num_hosts: int | None = None,
        host_mesh: HostMesh | None = None,
        env_vars: dict[str, str] | None = None,
        addr: str | None = None,
        port: str | None = None,
    ):
        """Gets a proc mesh.

        Args:
            num_procs: The number of processes to allocate.
            mesh_name: Name of the pre-allocated mesh to use.
                Must match a mesh name defined in the launcher config.
            with_gpus: Whether to include GPU allocations.
                This only adds the CUDA_VISIBLE_DEVICES environment variable.
            num_hosts: The number of hosts to allocate.
                If this is set, a remote allocation is created.
                If this is None, it uses the local host.
                This behavior may change in the future.
            mesh_name: Name of the pre-allocated mesh to use.
                Required for remote allocations (when num_hosts > 0).
                Must match a mesh name defined in the launcher config.
            host_mesh: The host mesh to allocate the process on.
                If None, will use the pre-allocated host mesh corresponding to mesh_name.
            env_vars: Additional environment variables to set for the spawned processes.
                These will be merged with auto-detected env vars (CUDA_VISIBLE_DEVICES, MASTER_ADDR, etc.).
            addr: The distributed address to use.
                If None, an address will be detected.
            port: The distributed port to use.
                If None, a port will be detected.

        Returns:
            A ProcMesh.

        """
        if env_vars is None:
            env_vars = {}

        is_remote = num_hosts is not None and num_hosts > 0

        async with self._lock:
            if is_remote:
                if host_mesh is None:
                    host_mesh = await self.get_host_mesh(
                        name=mesh_name,
                    )
                    host_id = uuid.uuid1()
                    # Get the GPU count from the remote host
                    remote_gpu_count = await get_host_gpus(host_mesh)
                    gpu_manager = GpuManager(max_device_count=remote_gpu_count)
                    self._host_gpu_map[host_id] = gpu_manager
                    host_mesh._host_id = host_id
                else:
                    host_id = host_mesh._host_id
                    gpu_manager = self._host_gpu_map[host_id]
            else:
                # fallback to local
                host_mesh = this_host()
                gpu_manager = self._host_gpu_map[self._this_host_id]
                host_mesh._host_id = self._this_host_id

            if with_gpus:
                if not addr or not port:
                    addr, port = await get_remote_info(host_mesh)
                gpu_ids = gpu_manager.get_gpus(num_procs)

                env_vars["MASTER_ADDR"] = addr
                env_vars["MASTER_PORT"] = port

                # Set the PTD world size
                world_size = num_procs * (num_hosts or 1)
                env_vars["WORLD_SIZE"] = str(world_size)
                env_vars["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)

                # Inherit Forge-relevant environment variables from the system
                for env_var in all_env_vars():
                    env_vars[env_var.name] = str(env_var.get_value())

            # Spawn procs without bootstrap to avoid SetupActor mesh failures
            procs = host_mesh.spawn_procs(
                per_host={"procs": num_procs},
                name=mesh_name,
            )

            # Set up environment variables (replaces old bootstrap)
            if env_vars:
                await set_environment(procs, env_vars)

            # Set up PyTorch distributed environment if using GPUs
            if with_gpus:
                await setup_env_for_distributed(
                    procs,
                    master_addr=addr,
                    master_port=int(port),
                )

            if is_remote:
                await self.launcher.remote_setup(procs)

            # Tag the proc mesh with additional metadata for our own cleanup later
            if with_gpus:
                # Applies any launcher specific remote setup.
                procs._gpu_ids = gpu_ids

            procs._host = host_mesh

            self._proc_host_map[procs] = host_mesh

        # Spawn LocalFetcherActor for this ProcMesh and register with GlobalLoggingActor.
        # When called, the LocalFetcherActor is broadcast by Monarch to all ranks in the ProcMesh.
        if not FORGE_DISABLE_METRICS.get_value():
            from forge.observability.metric_actors import get_or_create_metric_logger

            _ = await get_or_create_metric_logger(procs, process_name=mesh_name)
        return procs

    async def host_mesh_from_proc(self, proc_mesh: ProcMesh):
        if proc_mesh not in self._proc_host_map:
            raise ValueError(
                "The proc mesh was not allocated with an associated hostmesh."
            )
        return self._proc_host_map[proc_mesh]

    async def stop_proc_mesh(self, proc_mesh: ProcMesh):
        """Stops a proc mesh."""
        if proc_mesh not in self._proc_host_map:
            logger.warning(
                f"proc mesh {proc_mesh} was requested to be stopped, but was either already stopped or "
                "was never registered with the provisioner."
            )
            return
        async with self._lock:
            # Deregister LocalFetcherActor from GlobalLoggingActor
            if hasattr(proc_mesh, "_local_fetcher") and hasattr(proc_mesh, "_uid"):
                from forge.observability.metric_actors import (
                    get_or_create_metric_logger,
                )

                global_logger = await get_or_create_metric_logger(proc_mesh)
                await global_logger.deregister_fetcher.call_one(proc_mesh._uid)

            if hasattr(proc_mesh, "_gpu_ids"):
                gpu_manager = self._host_gpu_map[proc_mesh._host._host_id]
                gpu_manager.release_gpus(proc_mesh._gpu_ids)
            await proc_mesh.stop()

            del self._proc_host_map[proc_mesh]

    def register_service(self, service: "ServiceInterface") -> None:
        """Registers a service allocation for cleanup."""
        # Import ServiceInterface here instead of at top-level to avoid circular import
        from forge.controller.service import ServiceInterface

        if not isinstance(service, ServiceInterface):
            raise TypeError(
                f"register_service expected ServiceInterface, got {type(service)}"
            )

        self._registered_services.append(service)

    def register_actor(self, actor: "ForgeActor") -> None:
        """Registers a single actor allocation for cleanup."""

        if not isinstance(actor, ActorMesh):
            raise TypeError(f"register_actor expected ActorMesh, got {type(actor)}")

        self._registered_actors.append(actor)

    async def shutdown_all_allocations(self):
        """Gracefully shut down all tracked actors and services."""
        logger.info(
            f"Shutting down {len(self._registered_services)} service(s) and {len(self._registered_actors)} actor(s)..."
        )
        # --- ServiceInterface ---
        for service in reversed(self._registered_services):
            try:
                await service.shutdown()

            except Exception as e:
                logger.warning(f"Failed to shut down {service}: {e}")

        # --- Actor instance (ForgeActor or underlying ActorMesh) ---
        for actor in reversed(self._registered_actors):
            try:
                # Get the class to call shutdown on (ForgeActor or its bound class)
                actor_cls = getattr(actor, "_class", None) or actor.__class__
                await actor_cls.shutdown(actor)

            except Exception as e:
                logger.warning(f"Failed to shut down {actor}: {e}")

        self._registered_actors.clear()
        self._registered_services.clear()

        if self.launcher:
            self._job.kill()
        try:
            await shutdown_context()
        except Exception as e:
            logger.warning(f"Failed to shutdown context: {e}")

    async def shutdown(self):
        """Tears down all remaining remote allocations."""
        await self.shutdown_all_allocations()


_provisioner: Provisioner | None = None


async def init_provisioner(cfg: ProvisionerConfig | None = None):
    global _provisioner
    if not _provisioner:
        _provisioner = Provisioner(cfg)
        await _provisioner.initialize()
    return _provisioner


async def _get_provisioner():
    if not _provisioner:
        await init_provisioner()
    return _provisioner


async def get_proc_mesh(
    process_config: ProcessConfig,
    host_mesh: HostMesh | None = None,
    env_vars: dict[str, str] | None = None,
    port: str | None = None,
    addr: str | None = None,
) -> ProcMesh:
    """Returns a proc mesh from the provisioner.

    Args:
        process_config: The process config.
        host_mesh: The host mesh to allocate the process on.
            If None, a new host mesh will be created.
        port: The distributed port to use.
            If None, a port will be detected.
        addr: The distributed address to use.
            If None, an address will be detected.

    Returns:
        A proc mesh.

    """
    provisioner = await _get_provisioner()
    return await provisioner.get_proc_mesh(
        num_procs=process_config.procs,
        with_gpus=process_config.with_gpus,
        num_hosts=process_config.hosts,
        mesh_name=process_config.mesh_name,
        host_mesh=host_mesh,
        env_vars=env_vars,
        port=port,
        addr=addr,
    )


async def host_mesh_from_proc(proc_mesh: ProcMesh):
    """Returns the host mesh that allocated the original proc_mesh.

    This functionality will be enabled in Monarch, so this is a temporary
    API.

    """
    provisioner = await _get_provisioner()
    return await provisioner.host_mesh_from_proc(proc_mesh)


async def register_service(service: "ServiceInterface") -> None:
    """Registers a service allocation with the global provisioner."""
    provisioner = await _get_provisioner()
    provisioner.register_service(service)


async def register_actor(actor: "ForgeActor") -> None:
    """Registers an actor allocation with the global provisioner."""
    provisioner = await _get_provisioner()
    provisioner.register_actor(actor)


async def stop_proc_mesh(proc_mesh: ProcMesh):
    provisioner = await _get_provisioner()
    return await provisioner.stop_proc_mesh(proc_mesh=proc_mesh)


async def shutdown_metric_logger():
    """Shutdown the global metric logger and all its backends."""
    from forge.observability.metric_actors import get_or_create_metric_logger

    logger.info("Shutting down metric logger...")
    try:
        mlogger = await get_or_create_metric_logger()
        await mlogger.shutdown.call_one()
    except Exception as e:
        logger.warning(f"Failed to shutdown metric logger: {e}")


async def shutdown():
    await shutdown_metric_logger()

    logger.info("Shutting down provisioner..")

    provisioner = await _get_provisioner()
    result = await provisioner.shutdown()

    logger.info("Shutdown completed successfully")
    return result
