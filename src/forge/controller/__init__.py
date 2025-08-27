# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

__all__ = [
    "Service",
    "ServiceConfig",
    "ServiceInterface",
    "Session",
    "SessionContext",
    "spawn_service",
    "spawn_actors",
    "get_proc_mesh",
    "ForgeActor",
]


def __getattr__(name):
    if name == "Service":
        from .service import Service

        return Service
    elif name == "ServiceConfig":
        from .service import ServiceConfig

        return ServiceConfig
    elif name == "ServiceInterface":
        from .interface import ServiceInterface

        return ServiceInterface
    elif name == "Session":
        from .interface import Session

        return Session
    elif name == "SessionContext":
        from .interface import SessionContext

        return SessionContext
    elif name == "spawn_service":
        from .spawn import spawn_service

        return spawn_service
    elif name == "spawn_actors":
        from .proc_mesh import spawn_actors

        return spawn_actors
    elif name == "get_proc_mesh":
        from .proc_mesh import get_proc_mesh

        return get_proc_mesh
    elif name == "ForgeActor":
        from .actor import ForgeActor

        return ForgeActor
    else:
        raise AttributeError(f"module {__name__} has no attribute {name}")
