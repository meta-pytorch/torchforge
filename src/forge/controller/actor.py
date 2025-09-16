# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging

import math
import sys
from typing import Type

from monarch.actor import Actor, current_rank, current_size, endpoint

from forge.controller.proc_mesh import get_proc_mesh, stop_proc_mesh

from forge.types import ProcessConfig, ServiceConfig

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class ForgeActor(Actor):
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
        cls,
        *,
        service_config: ServiceConfig | None = None,
        num_replicas: int | None = None,
        procs_per_replica: int | None = None,
        **service_kwargs,
    ) -> Type["ConfiguredService"]:
        """
        Returns a ConfiguredService class that wraps this ForgeActor in a Service.

        Usage (choose ONE of the following forms):
            # Option A: construct ServiceConfig implicitly
            service = await MyForgeActor.options(
                num_replicas=1,
                procs_per_replica=2,
            ).as_service(...)
            await service.shutdown()

            # Option B: provide an explicit ServiceConfig
            cfg = ServiceConfig(num_replicas=1, procs_per_replica=2, scheduling="round_robin")
            service = await MyForgeActor.options(service_config=cfg).as_service(...)
            await service.shutdown()

        """
        from forge.controller.service import Service, ServiceInterface

        if service_config is not None:
            # Use the provided config directly
            cfg = service_config
        else:
            if num_replicas is None or procs_per_replica is None:
                raise ValueError(
                    "Must provide either `service_config` or (num_replicas + procs_per_replica)."
                )
            cfg = ServiceConfig(
                num_replicas=num_replicas,
                procs_per_replica=procs_per_replica,
                **service_kwargs,
            )

        class ConfiguredService:
            """
            A wrapper around Service that binds a ForgeActor class.
            Provides:
              - as_service(): spawns the actor inside the service
              - shutdown(): stops the service
            """

            _actor_def = cls
            _service_interface: ServiceInterface | None

            def __init__(self) -> None:
                self._service_interface = None

            @classmethod
            async def as_service(cls, **actor_kwargs) -> "ConfiguredService":
                """
                Spawn the actor inside a Service with the given configuration.

                Args:
                    **actor_kwargs: arguments to pass to the ForgeActor constructor

                Returns:
                    self: so that methods like .shutdown() can be called
                """
                self = cls()
                logger.info("Spawning Service Actor for %s", self._actor_def.__name__)
                service = Service(cfg, self._actor_def, actor_kwargs)
                await service.__initialize__()
                self._service_interface = ServiceInterface(service, self._actor_def)
                return self

            async def shutdown(self):
                """
                Gracefully stops the service if it has been started.
                """
                if self._service_interface is None:
                    raise RuntimeError("Service not started yet")
                await self._service_interface._service.stop()
                self._service_interface = None

            def __getattr__(self, item):
                """
                Delegate attribute access to the ServiceInterface instance.
                This makes ConfiguredService behave like a ServiceInterface.
                """
                if self._service_interface is None:
                    raise AttributeError(
                        f"Service not started yet; cannot access '{item}'"
                    )
                return getattr(self._service_interface, item)

        return ConfiguredService

    @classmethod
    async def as_service(cls, **actor_kwargs) -> "ConfiguredService":
        """
        Spawn this ForgeActor inside a Service with default configuration.
        Defaults: num_replicas=1, procs_per_replica=1

        Usage:
            service = await MyForgeActor.as_service(...)
            await service.shutdown()
        """
        return await cls.options(num_replicas=1, procs_per_replica=1).as_service(
            **actor_kwargs
        )

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
    async def launch(cls, *, process_config: ProcessConfig, **kwargs) -> "ForgeActor":
        """Provisions and deploys a new actor.

        This method is used by `Service` to provision a new replica.

        We implement it this way because special actors like inference servers
        may be composed of multiple actors spawned across multiple processes.
        This allows you to specify how your actor gets launched together.

        This implementation is basic, assuming that we're spawning
        a homogeneous set of actors on a single proc mesh.

        """
        proc_mesh = await get_proc_mesh(process_config=process_config)

        # TODO - expand support so name can stick within kwargs
        actor_name = kwargs.pop("name", cls.__name__)
        actor = await proc_mesh.spawn(actor_name, cls, **kwargs)
        actor._proc_mesh = proc_mesh

        if hasattr(proc_mesh, "_hostname") and hasattr(proc_mesh, "_port"):
            host, port = proc_mesh._hostname, proc_mesh._port
            await actor.set_env.call(addr=host, port=port)
        await actor.setup.call()
        return actor

    @classmethod
    async def shutdown(cls, actor: "ForgeActor"):
        """Shuts down an actor.

        This method is used by `Service` to teardown a replica.
        """
        if actor._proc_mesh is None:
            raise AssertionError("Called shutdown on a replica with no proc_mesh.")
        await stop_proc_mesh(actor._proc_mesh)
