# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Remote and local resource manager for allocation and provisioning."""
import asyncio
import functools
import logging

import os
import socket
import uuid

from monarch._src.actor.actor_mesh import ActorMesh
from monarch._src.actor.shape import Extent

from monarch.actor import Actor, endpoint, HostMesh, ProcMesh, this_host

from monarch.tools import commands

from forge.controller.launcher import BaseLauncher, get_launcher

from forge.env import all_env_vars, FORGE_DISABLE_METRICS
from forge.types import ProcessConfig, ProvisionerConfig

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
        return socket.gethostname(), _get_port()


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
    await throwaway_procs.stop()
    return host, port


class GpuManager:
    """Tracks and assigns GPU devices on a host.

    This currently mimics the `gpu_manager` in system_controllers - we will
    consolidate as part of the "proper HostMesh integration" work.

    """

    def __init__(self, available_devices: set[int] | None = None):
        if available_devices is None:
            available_devices = set(range(0, 8))
        assert all(isinstance(x, int) for x in available_devices)
        assert all(x >= 0 and x < 8 for x in available_devices)
        self.available_gpus = available_devices

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
        self._server_names = []
        self._proc_server_map = {}
        self._lock = asyncio.Lock()

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
        self._host_gpu_map = {
            self._this_host_id: GpuManager(available_local_devices),
        }
        self._proc_host_map = {}
        self._host_mesh_map = {}
        self.launcher: BaseLauncher | None = get_launcher(
            cfg.launcher_config if cfg is not None else None
        )
        if not self.launcher:
            logger.warning("Launcher not provided, remote allocations will not work.")

        self._registered_actors: list["ForgeActor"] = []
        self._registered_services: list["ServiceInterface"] = []

    async def initialize(self):
        """Call this after creating the instance"""
        if self.launcher is not None:
            await self.launcher.initialize()

    async def create_host_mesh(self, name: str, num_hosts: int) -> HostMesh:
        """Creates a remote server and a HostMesh on it."""
        # no need to lock here because this is already locked behind `get_proc_mesh`
        if not self.launcher:
            raise RuntimeError(
                "You tried to create a remote allocation by specifying the number of hosts on an actor or service, "
                "but no launcher was specified."
            )
        logger.debug(f"Creating remote server for alloc {name}")
        alloc, alloc_constraints, server_name = await self.launcher.get_allocator(
            name, num_hosts
        )

        # We are asking Monarch to allocate a single process on
        # every host, reflected in the Extent we provide below.

        # Technically, this is ["hosts", "procs"] but to reduce
        # confusion on its relationship with procs elsewhere,
        # we call it "no_dim".

        # TODO - remove this once Monarch supports HostMesh without it.
        host_mesh = HostMesh.allocate_nonblocking(
            name=name,
            extent=Extent(["hosts", "no_dim"], [num_hosts, 1]),
            allocator=alloc,
            alloc_constraints=alloc_constraints,
        )
        return host_mesh, server_name

    def get_host_mesh(self, name: str) -> HostMesh:
        """Returns the host mesh given its associated name.

        This is currently an experimental API for HostMesh v1 and
        should not be relied on longer term.

        """
        return self._host_mesh_map[name]

    async def get_proc_mesh(
        self,
        num_procs: int,
        with_gpus: bool = False,
        num_hosts: int | None = None,
        mesh_name: str | None = None,
        host_mesh: HostMesh | None = None,
        env_vars: dict[str, str] | None = None,
        addr: str | None = None,
        port: str | None = None,
    ):
        """Gets a proc mesh.

        Args:
            num_procs: The number of processes to allocate.
            with_gpus: Whether to include GPU allocations.
                This only adds the CUDA_VISIBLE_DEVICES environment variable.
            num_hosts: The number of hosts to allocate.
                If this is set, a remote allocation is created.
                If this is None, it uses the local host.
                This behavior may change in the future.
            host_mesh: The host mesh to allocate the process on.
                If None, a new host mesh will be created.
            port: The distributed port to use.
                If None, a port will be detected.
            addr: The distributed address to use.
                If None, an address will be detected.

        Returns:
            A ProcMesh.

        """
        if env_vars is None:
            env_vars = {}

        is_remote = num_hosts is not None and num_hosts > 0

        async with self._lock:
            server_name = None
            if is_remote:
                if mesh_name is None:
                    created_hosts = len(self._server_names)
                    mesh_name = f"alloc_{created_hosts}"
                if host_mesh is None:
                    host_mesh, server_name = await self.create_host_mesh(
                        name=mesh_name,
                        num_hosts=num_hosts,
                    )
                    host_id = uuid.uuid1()
                    gpu_manager = GpuManager()
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

            def bootstrap(env: dict[str, str]):
                """Runs on process startup.

                We use this to set environment variables like CUDA, etc.
                We prefer to pass in environment variables to bootstrap,
                but there are occasionally host-specific environments that can
                only be set once the process is alive on remote hosts.

                """
                # bootstrap is run on all processes. We use this
                # to set environment variables like CUDA etc.
                import os

                # vLLM requires this environment variable when spawning model servers
                # across multiple nodes.
                os.environ["VLLM_HOST_IP"] = socket.gethostbyname(socket.getfqdn())

                for k, v in env.items():
                    os.environ[k] = v

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

            procs = host_mesh.spawn_procs(
                per_host={"procs": num_procs},
                bootstrap=functools.partial(bootstrap, env=env_vars),
            )

            if is_remote:
                await self.launcher.remote_setup(procs)

            # Tag the proc mesh with additional metadata for our own cleanup later
            if with_gpus:
                # Applies any launcher specific remote setup.
                procs._gpu_ids = gpu_ids

            self._host_mesh_map[mesh_name] = host_mesh
            procs._host = host_mesh

            # If we created a server, track so we can tear it down later.
            if server_name:
                self._server_names.append(server_name)
                self._proc_server_map[procs] = server_name

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
            if proc_mesh in self._proc_server_map:
                server_name = self._proc_server_map[proc_mesh]
                commands.kill(server_name)
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

    async def shutdown(self):
        """Tears down all remaining remote allocations."""
        await self.shutdown_all_allocations()
        async with self._lock:
            for server_name in self._server_names:
                commands.kill(server_name)


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
