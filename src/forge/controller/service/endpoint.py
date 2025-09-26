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
    ServiceEndpoint provides the basic, non-batched routing API for Forge services.

    Args:
        service: The underlying service object that owns replicas.
        endpoint_name (str): The name of the endpoint method.
        router (str, optional): Routing strategy for stateless requests.
                 Supported values:
                   - "round_robin": cycle through replicas in order.
                   - "leastloaded": pick the replica with the lowest load.
                 Default: "round_robin".

    Supported methods:
        - `route`: Send a request to a single replica, chosen by the configured router
        (e.g. round-robin, least-loaded).
        - `fanout`: Broadcasts the request to all healthy replicas.

    Notes:
        - Support `@endpoint()` and `@service_endpoint(router='..')` decorators.
        - To specify router, use `@service_endpoint(router='..')`.
        - Retry logic: If `max_attempts > 1`, failed calls may be retried on a different replica
        if the first one becomes unhealthy.
        - Session-aware routing: If a `sess_id` is provided, requests are routed via
        `SessionRouter` for sticky session behavior.
        - Monarch's native actor APIs do not apply for services.
    """

    def __init__(
        self,
        service,
        endpoint_name: str,
        router: str = "round_robin",
    ):
        self.service = service
        self.endpoint_name = endpoint_name

        # Primary router (stateless routing)
        self.router = self._resolve_router(router)

        # Session-aware router for sticky sessions
        self.session_router = SessionRouter(fallback_router=self.router)

        # Number of routing attempts (initial + retries)
        self.max_attempts = 1

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
        """
        Route a single request to one replica.

        Retries up to `self.max_attempts` times if the chosen replica fails
        and is marked unhealthy. Sticky session mapping is cleared on retry.
        """
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
        """
        Select a replica to handle the request.

        - If `sess_id` is provided, use the session router for sticky sessions.
        - Otherwise, use the stateless router to pick among healthy replicas.
        """

        # Stateful routing always uses session router
        if sess_id:
            healthy = self.service._get_healthy_replicas()
            return self.session_router.get_replica(
                healthy, sess_id, self.service._session_replica_map
            )

        # Use router to choose a replica
        healthy_replicas = self.service._get_healthy_replicas()
        return self.router.get_replica(
            healthy_replicas, None, self.service._session_replica_map
        )

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
        """Stop the service endpoint.

        For plain ServiceEndpoint (non-batched), this is a no-op.
        """
        return


class BatchedServiceEndpoint(ServiceEndpoint[P, R]):
    """
    A ServiceEndpoint variant that supports request batching.

    Args:
        service: The underlying service object that owns replicas.
        endpoint_name (str): The name of the endpoint method.
        router (str, optional): Routing strategy for stateless requests.
                 Supported values:
                   - "round_robin": cycle through replicas in order.
                   - "leastloaded": pick the replica with the lowest load.
                 Default: "round_robin".
        batch_size (int, optional): Maximum number of requests to group together
                 in a single batch before dispatching. Default: 8.
        batch_timeout (float, optional): Maximum time (in seconds) to wait before
                 dispatching a batch. Default: 0.01.

    Key features:
        - Collects requests into batches of up to `batch_size`.
        - Uses a background asyncio task (`_batch_loop`) to manage the queue.
        - Makes one routing decision per batch, and assigns the chosen replica
        to all requests in that batch.
        - Provides the same API (`route`, `fanout`, `stop`) as ServiceEndpoint.

    Usage:
        class MyForgeActor(ForgeActor):
            @service_endpoint(router="round_robin", batch_size=16, batch_timeout=0.05)
            async def forward(self, x): ...
    """

    def __init__(
        self,
        service,
        endpoint_name: str,
        router: str = "round_robin",
        batch_size: int = 8,
        batch_timeout: float = 0.01,
    ):
        super().__init__(service, endpoint_name, router=router)

        self.batch_size = batch_size
        self.batch_timeout = batch_timeout
        self._batch_queue: asyncio.Queue = asyncio.Queue()
        self._running_batch_loop = True
        self.batch_task = asyncio.create_task(self._batch_loop())

    async def _choose_replica(self, sess_id: str | None) -> "Replica":
        """
        Overridden to support batching.

        - Session requests bypass batching and use sticky session router.
        - Stateless requests are enqueued; the batch loop will fulfill their
          Future with a chosen replica.
        """

        # Stateful routing always uses session router
        if sess_id:
            healthy = self.service._get_healthy_replicas()
            return self.session_router.get_replica(
                healthy, sess_id, self.service._session_replica_map
            )
        # Stateless: batching
        fut = asyncio.Future()
        self._batch_queue.put_nowait(fut)
        return await fut

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

            # Wait for the first request to start a batch
            fut = await self._batch_queue.get()
            batch_futs.append(fut)
            start_time = time.monotonic()

            # Collect additional requests until batch size or timeout
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

            # Make one routing decision for the batch
            healthy_replicas = self.service._get_healthy_replicas()
            replica = self.router.get_replica(
                healthy_replicas, None, self.service._session_replica_map
            )

            # Fulfill all futures with the chosen replica
            for fut in batch_futs:
                fut.set_result(replica)

    async def stop(self):
        """Stop the batching loop."""
        self._running_batch_loop = False
        if hasattr(self, "batch_task"):
            self.batch_task.cancel()


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
            batch_size=batch_size,
            batch_timeout=batch_timeout,
        )
        return ep

    return decorator
