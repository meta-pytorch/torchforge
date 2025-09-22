# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging

import math
import sys
from typing import Type, TypeVar

from monarch.actor import Actor, current_rank, current_size, endpoint

from forge.controller.proc_mesh import get_proc_mesh, stop_proc_mesh

from forge.types import ProcessConfig, ServiceConfig

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
T = TypeVar("T", bound="ForgeActor")


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
        cls: Type[T],
        *,
        service_config: ServiceConfig | None = None,
        process_config: ProcessConfig | None = None,
        num_replicas: int | None = None,
        procs: int | None = None,
        **service_kwargs,
    ) -> Type[T]:
        """
        Returns a subclass of this ForgeActor with a bound ServiceConfig.
        The returned subclass can later be launched via `.as_service()`.

        Usage modes:

        ---- Service Mode (default) ----
        Use when deploying a replicated service (multiple replicas, each with N procs).

            # Option A: construct ServiceConfig implicitly
            service = await MyForgeActor.options(
                num_replicas=1,
                procs=2,
            ).as_service(...)
            await service.shutdown()

            # Option B: provide an explicit ServiceConfig
            cfg = ServiceConfig(num_replicas=1, procs=2, ..)
            service = await MyForgeActor.options(service_config=cfg).as_service(...)
            await service.shutdown()

            # Option C: skip options, use the default service config with num_replicas=1, procs=1
            service = await MyForgeActor.as_service(...)
            await service.shutdown()

        ---- Single Actor Mode ----
        Use when launching just one actor directly (without Service abstraction).
        Must provide a ProcessConfig.

            cfg = ProcessConfig(...)
            actor = await MyForgeActor.options(process_config=cfg).as_actor(...)
            await actor.shutdown()

        ---- Notes ----
        - If `process_config` is passed, we bind to an actor configuration
          and expect `.as_actor(...)` to be called later.
        - Otherwise (default), we bind to a service configuration and expect
          `.as_service(...)` to be called later.
        - Passing both `service_config` and `process_config` is invalid.
        """

        if service_config is not None and process_config is not None:
            raise ValueError(
                "Cannot pass both `service_config` and `process_config`. "
                "Use either `service_config` for service mode or `process_config` for single actor mode."
            )

        if process_config is not None:
            return type(
                f"{cls.__name__}Actor",
                (cls,),
                {"_process_config": process_config},
            )
        elif service_config is not None:
            cfg = service_config
            return type(
                f"{cls.__name__}Service",
                (cls,),
                {"_service_config": cfg},
            )
        else:
            if num_replicas is None or procs is None:
                raise ValueError(
                    "Must provide either `service_config` or (num_replicas + procs)."
                )
            cfg = ServiceConfig(
                num_replicas=num_replicas,
                procs=procs,
                **service_kwargs,
            )

            return type(
                f"{cls.__name__}Service",
                (cls,),
                {"_service_config": cfg},
            )

    @classmethod
    async def as_service(cls: Type[T], **actor_kwargs) -> "ServiceInterface":
        """
        Convenience method to spawn this actor as a Service using default configuration.
        If `.options()` was called, it will use the bound ServiceConfig;
        otherwise defaults to 1 replica, 1 proc.
        """
        # Lazy import to avoid top-level dependency issues
        from forge.controller.service import Service, ServiceInterface

        # Use _service_config if already set by options(), else default
        cfg = getattr(cls, "_service_config", None)
        if cfg is None:
            cfg = ServiceConfig(num_replicas=1, procs=1)
            # dynamically create a configured subclass for consistency
            cls = type(f"{cls.__name__}Service", (cls,), {"_service_config": cfg})

        logger.info("Spawning Service Actor for %s", cls.__name__)
        service = Service(cfg, cls, actor_kwargs)
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
    async def as_actor(cls: Type[T], **actor_kwargs) -> T:
        """
        Spawns a single actor using the ProcessConfig bound in `.options()`.
        Example:
            cfg = ProcessConfig(...)
            actor = await MyForgeActor.options(process_config=cfg).as_actor(...)
        """
        cfg = getattr(cls, "_process_config", None)
        if cfg is None:
            raise ValueError(
                "No process_config found. Use `.options(process_config=...)` before calling `.as_actor()`."
            )
        logger.info("Spawning single actor %s", cls.__name__)
        actor = await cls.launch(process_config=cfg, **actor_kwargs)
        return actor

    @classmethod
    async def shutdown(cls, actor: "ForgeActor"):
        """Shuts down an actor.

        This method is used by `Service` to teardown a replica.
        """
        if actor._proc_mesh is None:
            raise AssertionError("Called shutdown on a replica with no proc_mesh.")
        await stop_proc_mesh(actor._proc_mesh)
