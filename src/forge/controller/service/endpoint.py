# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Service endpoint management for the Forge framework.
"""

import asyncio
from typing import Generic, List, TypeVar

from monarch.actor import endpoint

from typing_extensions import ParamSpec

from .router import LeastLoadedRouter, RoundRobinRouter, Router, SessionRouter

P = ParamSpec("P")
R = TypeVar("R")


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
        router: str = "round_robin",
        batch_size: int = 1,
        batch_timeout: float = 0.1,
    ):
        self.service = service
        self.endpoint_name = endpoint_name

        self.router = self._resolve_router(router)
        self.session_router = SessionRouter(fallback_router=self.router)

    def _resolve_router(self, router_name: str) -> Router:
        """Convert a router name into a router object.

        Args:
            router_name (str): a router name. Supported routers: "round_robin", "leastloaded".

        Returns:
            Router: A Router object.
        """
        if router_name == "round_robin":
            return RoundRobinRouter()
        if router_name == "leastloaded":
            return LeastLoadedRouter()
        raise ValueError(
            f"Unknown router name: {router_name}. Supported routers: 'round_robin', 'leastloaded'."
        )

    async def route(self, *args: P.args, **kwargs: P.kwargs) -> R:
        """Chooses a replica to call based on context and load balancing strategy."""
        # Extract sess_id from kwargs if present
        sess_id = kwargs.pop("sess_id", None)
        return await self.service._call(sess_id, self.endpoint_name, *args, **kwargs)

    async def fanout(self, *args: P.args, **kwargs: P.kwargs) -> List[R]:
        """Broadcasts a request to all healthy replicas and returns the results as a list."""
        result = await self.service.call_all(self.endpoint_name, *args, **kwargs)
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


class BatchedServiceEndpoint(ServiceEndpoint[P, R]):
    """
    A ServiceEndpoint that supports request batch routing.

    Args:
        router: The underlying Router instance used to make routing decisions
        session_router: The fallback Router for session-based routing.
        batch_max_size: Maximum number of requests to collect in a single batch (default: 8)
        batch_max_wait_s: Maximum time to wait before processing a batch in seconds (default: 0.01)

    Features:
    - Maintains a batch queue
    - Spawns a background task to group requests into batches
    """

    def __init__(
        self,
        service,
        endpoint_name: str,
        router: str = "round_robin",
        session_router: str = "leastloaded",
        batch_size: int = 1,
        batch_timeout: float = 0.1,
    ):

        super().__init__(service, endpoint_name)

        self.router = self._resolve_router(router)
        self.session_router = SessionRouter(
            fallback_router=self._resolve_router(session_router)
        )

        self.batch_size = batch_size
        self.batch_timeout = batch_timeout

        self.batch_queue: asyncio.Queue = asyncio.Queue()
        self.running_batch_loop = False
        # if self.batch_size > 1:
        #     self.running_batch_loop = True
        #     self.batch_task = asyncio.create_task(self._batch_loop())

    # async def _batch_loop(self):
    #     while self.running_batch_loop:
    #         batch_futs = []

    #         fut = await self.batch_queue.get()
    #         batch_futs.append(fut)
    #         start_time = time.monotonic()

    #         while True:
    #             try:
    #                 timeout = max(
    #                     0, self.batch_timeout - (time.monotonic() - start_time)
    #                 )
    #                 fut = await asyncio.wait_for(self.batch_queue.get(), timeout)
    #                 batch_futs.append(fut)
    #                 if len(batch_futs) >= self.batch_size:
    #                     break
    #             except asyncio.TimeoutError:
    #                 break

    #         healthy_replicas = [r for r in self.service._replicas if r.healthy]
    #         replica = self.router.get_replica(healthy_replicas)

    #         for fut in batch_futs:
    #             fut.set_result(replica)

    # async def route(self, *args: P.args, **kwargs: P.kwargs) -> R:
    #     sess_id = kwargs.pop("sess_id", None)
    #     if sess_id:
    #         healthy_replicas = [r for r in self.service._replicas if r.healthy]
    #         replica = self.session_router.get_replica(healthy_replicas, sess_id)
    #     else:
    #         if self.batch_size > 1:
    #             fut = asyncio.Future()
    #             self.batch_queue.put_nowait(fut)
    #             replica = await fut
    #         else:
    #             healthy_replicas = [r for r in self.service._replicas if r.healthy]
    #             replica = self.router.get_replica(healthy_replicas)

    #     request = ServiceRequest(
    #         session_id=sess_id,
    #         function=self.endpoint_name,
    #         args=args,
    #         kwargs=kwargs,
    #         future=asyncio.Future(),
    #     )
    #     await replica.enqueue_request(request)
    #     return await request.future


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

    async def choose(self, *args: P.args, **kwargs: P.kwargs) -> R:
        """Chooses a replica to call based on context and load balancing strategy."""
        # Extract sess_id from kwargs if present
        sess_id = kwargs.pop("sess_id", None)
        return await self.actor_mesh.call.call_one(
            sess_id, self.endpoint_name, *args, **kwargs
        )

    async def call(self, *args: P.args, **kwargs: P.kwargs) -> List[R]:
        """Broadcasts a request to all healthy replicas and returns the results as a list."""
        result = await self.actor_mesh.call_all.call_one(
            self.endpoint_name, *args, **kwargs
        )
        return result


def service_endpoint(
    *,
    router="round_robin",
    session_router="leastloaded",
    batch_size=1,
    batch_timeout=0.01,
    propagate=None,
    explicit_response_port=False,
):
    """
    Marks an actor method as a service endpoint with batching routing support.

    Example:
        class MyForgeActor(ForgeActor):
            @service_endpoint(router="round_robin", batch_size=16, batch_timeout=0.05)
            async def predict(self, x): ...
    """

    def decorator(method):
        # First wrap in EndpointProperty (so actor has a proper endpoint)
        ep = endpoint(
            method, propagate=propagate, explicit_response_port=explicit_response_port
        )
        ep._service_endpoint_config = dict(
            router=router,
            session_router=session_router,
            batch_size=batch_size,
            batch_timeout=batch_timeout,
        )
        return ep

    return decorator
