# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging

import math
import sys
from typing import Any, Type, TypeVar

from monarch.actor import Actor, current_rank, current_size, endpoint

from forge.controller.proc_mesh import get_proc_mesh, stop_proc_mesh

from forge.types import ProcessConfig, ServiceConfig

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
T = TypeVar("T", bound="ForgeActor")


class ForgeActor(Actor):
    procs: int = 1
    hosts: int | None = None
    with_gpus: bool = False
    num_replicas: int = 1
    _extra_config: dict[str, Any] = {}

    def __init__(self, *args, **kwargs):
        if not hasattr(self, "_rank"):
            self._rank = current_rank().rank
        if not hasattr(self, "_size"):
            self._size = math.prod(current_size().values())

        # Custom formatter that includes rank/size info with blue prefix
        BLUE = "\033[34m"
        RESET = "\033[0m"
        formatter = logging.Formatter(
            f"{BLUE}[{self.__class__.__name__}-{self._rank}/{self._size}] %(asctime)s %(levelname)s{RESET} %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setLevel(logging.INFO)
        stdout_handler.setFormatter(formatter)

        self._proc_mesh = None
        self.logger.root.setLevel(logging.INFO)
        self.logger.root.addHandler(stdout_handler)
        super().__init__(*args, **kwargs)

    @classmethod
    def options(
        cls: Type[T],
        *,
        procs: int = 1,
        hosts: int | None = None,
        with_gpus: bool = False,
        num_replicas: int = 1,
        mesh_name: str | None = None,
        **kwargs,
    ) -> Type[T]:
        """
        Returns a version of ForgeActor with configured resource attributes.

        This method allows you to pre-configure an actor class before spawning it with
        `.as_actor()` or `.as_service()`. Each call creates a separate subclass, so
        multiple different configurations can coexist without interfering with each other.

        ---- Usage Examples ----

        # Pre-configure a service with multiple replicas
        service = await MyForgeActor.options(num_replicas=2, procs=2).as_service(...)
        await service.shutdown()

        # Default usage without calling options
        service = await MyForgeActor.as_service(...)
        await service.shutdown()

        # Pre-configure a single actor
        actor = await MyForgeActor.options(procs=1, hosts=1).as_actor(...)
        await actor.shutdown()

        # Default usage without calling options
        actor = await MyForgeActor.as_actor(...)
        await actor.shutdown()
        """

        attrs = {
            "procs": procs,
            "hosts": hosts,
            "with_gpus": with_gpus,
            "num_replicas": num_replicas,
            "mesh_name": mesh_name,
            "_extra_config": kwargs,
        }

        return type(cls.__name__, (cls,), attrs)

    @classmethod
    async def as_service(
        cls: Type[T], *actor_args, **actor_kwargs
    ) -> "ServiceInterface":
        """
        Spawns this actor as a Service using the configuration stored in `.options()`,
        or defaults if `.options()` was not called.

        The configuration values stored in the subclass returned by `.options()` (like
        `procs` and `num_replicas`) are used to construct a ServiceConfig instance.
        If no configuration was stored, defaults to a single replica with one process.
        """
        # Lazy import to avoid top-level dependency issues
        from forge.controller.service import Service, ServiceInterface

        cfg_kwargs = {
            "procs": cls.procs,
            "hosts": cls.hosts,
            "with_gpus": cls.with_gpus,
            "num_replicas": cls.num_replicas,
            "mesh_name": cls.mesh_name,
            **cls._extra_config,  # all extra fields
        }
        cfg = ServiceConfig(**cfg_kwargs)

        logger.info("Spawning Service for %s", cls.__name__)
        service = Service(cfg, cls, actor_args, actor_kwargs)
        await service.__initialize__()
        return ServiceInterface(service, cls)

    @endpoint
    async def setup(self):
        """Sets up the actor.

        We assume a specific setup function for all actors. The
        best practice for actor deployment is to:
        1. Pass all data to the actor via the constructor.
        2. Call setup() to for heavy weight initializations.

        This is to ensure that any failures during initialization
        can be propagated back to the caller.

        """
        pass

    @endpoint
    async def set_env(self, addr: str, port: str):
        """A temporary workaround to set master addr/port.

        TODO - issues/144. This should be done in proc_mesh creation.
        The ideal path:
        - Create a host mesh
        - Grab a host from host mesh, from proc 0 spawn an actor that
          gets addr/port
        - Spawn procs on the HostMesh with addr/port, setting the
          addr/port in bootstrap.

        We can't currently do this because HostMesh only supports single
        proc_mesh creation at the moment. This will be possible once
        we have "proper HostMesh support".

        """
        import os

        os.environ["MASTER_ADDR"] = addr
        os.environ["MASTER_PORT"] = port

    @classmethod
    async def launch(cls, *args, **kwargs) -> "ForgeActor":
        """Provisions and deploys a new actor.

        This method is used by `Service` to provision a new replica.

        We implement it this way because special actors like inference servers
        may be composed of multiple actors spawned across multiple processes.
        This allows you to specify how your actor gets launched together.

        This implementation is basic, assuming that we're spawning
        a homogeneous set of actors on a single proc mesh.

        """
        # Build process config
        cfg = ProcessConfig(
            procs=cls.procs,
            hosts=cls.hosts,
            with_gpus=cls.with_gpus,
            mesh_name=cls.mesh_name,
        )

        proc_mesh = await get_proc_mesh(process_config=cfg)

        actor_name = kwargs.pop("name", cls.__name__)
        actor = proc_mesh.spawn(actor_name, cls, *args, **kwargs)
        actor._proc_mesh = proc_mesh

        if hasattr(proc_mesh, "_hostname") and hasattr(proc_mesh, "_port"):
            host, port = proc_mesh._hostname, proc_mesh._port
            await actor.set_env.call(addr=host, port=port)
        await actor.setup.call()
        return actor

    @classmethod
    async def as_actor(cls: Type[T], *args, **actor_kwargs) -> T:
        """
        Spawns a single actor using the configuration stored in `.options()`, or defaults.

        The configuration values stored in the subclass returned by `.options()` (like
        `procs`) are used to construct a ProcessConfig instance.
        If no configuration was stored, defaults to a single process with no GPU.
        """
        logger.info("Spawning single actor %s", cls.__name__)
        actor = await cls.launch(*args, **actor_kwargs)
        return actor

    @classmethod
    async def shutdown(cls, actor: "ForgeActor"):
        """Shuts down an actor.
        This method is used by `Service` to teardown a replica.
        """
        if actor._proc_mesh is None:
            raise AssertionError("Called shutdown on a replica with no proc_mesh.")
        await stop_proc_mesh(actor._proc_mesh)
