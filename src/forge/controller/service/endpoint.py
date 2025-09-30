# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Service endpoint management for the Forge framework.
"""

from typing import Any, Generic, List, TypeVar

from monarch._src.actor.endpoint import EndpointProperty

from typing_extensions import ParamSpec

from .router import RoundRobinRouter, Router

P = ParamSpec("P")
R = TypeVar("R")
Propagator = Any


class ServiceEndpoint(Generic[P, R]):
    """
    This extends Monarch's actor APIs for service endpoints.
    - `route(*args, **kwargs)`: Routes the request to a single replica.
    - `fanout(*args, **kwargs)`: Broadcasts the request to all healthy replicas.

    Monarch's native actor APIs do not apply for services.
    """

    def __init__(
        self,
        service,
        endpoint_name: str,
    ):
        self.service = service
        self.endpoint_name = endpoint_name

    async def route(self, *args: P.args, **kwargs: P.kwargs) -> R:
        """Chooses a replica to call based on context and load balancing strategy."""
        # Extract sess_id from kwargs if present
        sess_id = kwargs.pop("sess_id", None)
        return await self.service._route(sess_id, self.endpoint_name, *args, **kwargs)

    async def fanout(self, *args: P.args, **kwargs: P.kwargs) -> List[R]:
        """Broadcasts a request to all healthy replicas and returns the results as a list."""
        result = await self.service._fanout(self.endpoint_name, *args, **kwargs)
        return result

    async def choose(self, *args: P.args, **kwargs: P.kwargs) -> R:
        raise NotImplementedError(
            "You tried to use choose() on a service, not an actor. "
            "Services only support route() and fanout()."
        )

    async def call(self, *args: P.args, **kwargs: P.kwargs) -> List[R]:
        raise NotImplementedError(
            "You tried to use call() on a service, not an actor. "
            "Services only support route() and fanout()."
        )

    async def call_one(self, *args: P.args, **kwargs: P.kwargs) -> R:
        raise NotImplementedError(
            "You tried to use a call_one() on a service, not an actor. "
            "Services only support route() and fanout()."
        )

    async def broadcast(self, *args: P.args, **kwargs: P.kwargs) -> List[R]:
        raise NotImplementedError(
            "You tried to use broadcast() on a service, not an actor. "
            "Services only support route() and fanout()."
        )

    async def generate(self, *args: P.args, **kwargs: P.kwargs):
        raise NotImplementedError(
            "You tried to use generate() on a service, not an actor. "
            "Services only support route() and fanout()."
        )


class ServiceEndpointV2(Generic[P, R]):
    """An endpoint object specific to services.

    This loosely mimics the Endpoint APIs exposed in Monarch, with
    a few key differences:
    - Only choose and call are retained (dropping stream and call_one)
    - Call returns a list directly rather than a ValueMesh.

    These changes are made with Forge use cases in mind, but can
    certainly be expanded/adapted in the future.

    """

    def __init__(self, actor_mesh, endpoint_name: str):
        self.actor_mesh = actor_mesh
        self.endpoint_name = endpoint_name

    async def route(self, *args: P.args, **kwargs: P.kwargs) -> R:
        """Chooses a replica to call based on context and load balancing strategy."""
        # Extract sess_id from kwargs if present
        sess_id = kwargs.pop("sess_id", None)
        return await self.actor_mesh.call.call_one(
            sess_id, self.endpoint_name, *args, **kwargs
        )

    async def fanout(self, *args: P.args, **kwargs: P.kwargs) -> List[R]:
        """Broadcasts a request to all healthy replicas and returns the results as a list."""
        result = await self.actor_mesh.call_all.call_one(
            self.endpoint_name, *args, **kwargs
        )
        return result

    async def choose(self, *args: P.args, **kwargs: P.kwargs) -> R:
        raise NotImplementedError(
            "You tried to use choose() on a service, not an actor. "
            "Services only support route() and fanout()."
        )

    async def call(self, *args: P.args, **kwargs: P.kwargs) -> List[R]:
        raise NotImplementedError(
            "You tried to use call() on a service, not an actor. "
            "Services only support route() and fanout()."
        )

    async def call_one(self, *args: P.args, **kwargs: P.kwargs) -> R:
        raise NotImplementedError(
            "You tried to use a call_one() on a service, not an actor. "
            "Services only support route() and fanout()."
        )

    async def broadcast(self, *args: P.args, **kwargs: P.kwargs) -> List[R]:
        raise NotImplementedError(
            "You tried to use broadcast() on a service, not an actor. "
            "Services only support route() and fanout()."
        )

    async def generate(self, *args: P.args, **kwargs: P.kwargs):
        raise NotImplementedError(
            "You tried to use generate() on a service, not an actor. "
            "Services only support route() and fanout()."
        )


class ServiceEndpointProperty(EndpointProperty, Generic[P, R]):
    """
    Extension of EndpointProperty that carries service-specific
    routing and batching configuration.
    """

    def __init__(
        self,
        method: Any,
        propagator: Propagator,
        explicit_response_port: bool,
        *,
        router: Router = RoundRobinRouter(),
        batch_size: int = 1,
        batch_timeout: float = 0.01,
    ) -> None:
        super().__init__(method, propagator, explicit_response_port)
        self.router = router
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout


def service_endpoint(
    *,
    router: Router = RoundRobinRouter(),
    batch_size: int = 1,
    batch_timeout: float = 0.01,
    propagate=None,
    explicit_response_port=False,
):
    """
    Marks an actor method as a service endpoint with batching routing support.

    Example:
        class MyForgeActor(ForgeActor):
            @service_endpoint(router=RoundRobinRouter(), batch_size=16, batch_timeout=0.05)
            async def predict(self, x): ...
    """

    def decorator(method) -> ServiceEndpointProperty:
        return ServiceEndpointProperty(
            method,
            propagator=propagate,
            explicit_response_port=explicit_response_port,
            router=router,
            batch_size=batch_size,
            batch_timeout=batch_timeout,
        )

    return decorator
