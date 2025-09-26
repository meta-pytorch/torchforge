# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Service endpoint management for the Forge framework.
"""

import asyncio
import time
from typing import Generic, List, TypeVar

from monarch.actor import endpoint
from typing_extensions import ParamSpec

from .replica import Replica

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

        self.batch_size = batch_size
        self.batch_timeout = batch_timeout
        self._running_batch_loop = False
        self._batch_queue: asyncio.Queue = asyncio.Queue()
        if self.batch_size > 1:
            self._running_batch_loop = True
            self.batch_task = asyncio.create_task(self._batch_loop())

        self.max_attempts = 1  # number of tries for routing = initial + retries

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

        for attempt in range(self.max_attempts):
            replica = await self._choose_replica(sess_id)

            # Wait for the result
            try:
                return await self.service._call(
                    replica, sess_id, self.endpoint_name, *args, **kwargs
                )
            except Exception as e:
                # If the replica failed, try to retry
                if not replica.healthy and attempt < self.max_attempts - 1:
                    # Clear sticky mapping before retry
                    if (
                        sess_id is not None
                        and sess_id in self.service._session_replica_map
                    ):
                        del self.service._session_replica_map[sess_id]
                    continue  # retry with a fresh replica
                raise

    async def _choose_replica(self, sess_id: str | None) -> "Replica":
        """Get a replica for the given session ID."""

        # Stateful routing always uses session router
        if sess_id:
            healthy = self.service._get_healthy_replicas()
            return self.session_router.get_replica(
                healthy, sess_id, self.service._session_replica_map
            )
        # Stateless: batching
        if self.batch_size > 1:
            fut = asyncio.Future()
            self._batch_queue.put_nowait(fut)
            return await fut

        # No batching, pick immediately
        healthy = self.service._get_healthy_replicas()
        return self.router.get_replica(healthy)

    async def _batch_loop(self):
        """Background task that continuously processes batches of routing requests.

        This is the core batching logic that runs in a separate asyncio task.
        It collects requests from the queue and processes them in batches based
        on size and time constraints.

        The loop follows these steps:
        1. Wait for the first request to start a new batch
        2. Collect additional requests until batch_size or batch_timeout is reached
        3. Make a single routing decision for the entire batch
        4. Fulfill all futures with the selected replica

        This process repeats indefinitely until the task is cancelled.
        """
        while self._running_batch_loop:
            batch_futs = []

            # Wait for first request
            fut = await self._batch_queue.get()
            batch_futs.append(fut)
            start_time = time.monotonic()

            while True:
                try:
                    timeout = max(
                        0, self.batch_timeout - (time.monotonic() - start_time)
                    )
                    fut = await asyncio.wait_for(
                        self._batch_queue.get(), timeout
                    )  # wait for timeout or until self._queue.get() finishes
                    batch_futs.append(fut)

                    if len(batch_futs) >= self.batch_size:
                        break
                except asyncio.TimeoutError:
                    break

            healthy_replicas = self.service._get_healthy_replicas()

            # One routing decision for the whole batch
            replica = self.router.get_replica(
                healthy_replicas, None, self.service._session_replica_map
            )

            # Fulfill all futures with the chosen replica
            for fut in batch_futs:
                fut.set_result(replica)

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

    async def stop(self):
        """Stop the batching loop."""
        self._running_batch_loop = False


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
