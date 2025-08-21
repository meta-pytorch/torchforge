# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""
Distributed Actor Service Controller

This module provides a robust service orchestration system for managing distributed
actor-based workloads with automatic scaling, fault tolerance, and intelligent load balancing.

The main Service class acts as a singleton controller that handles:
- Fault tolerance with automatic replica recovery
- Autoscaling based on real-time metrics
- Load balancing across healthy replicas
- Session management with context propagation
- Comprehensive metrics collection and monitoring

Example:
    Basic service setup:

    >>> config = ServiceConfig(
    ...     gpus_per_replica=1,
    ...     num_replicas=3
    ... )
    >>> service = Service(config, MyActorClass, *args, **kwargs)
    >>> await service.__initialize__()

    Session-based usage:

    >>> async with service.session():
    ...     result = await service.my_endpoint(arg1, arg2)
"""


import asyncio
import contextvars
import logging
import pprint
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, Dict, List, Optional

from monarch._src.actor.endpoint import EndpointProperty
from monarch.actor import ProcMesh

from forge.controller import RecoverableProcMesh
from forge.controller.replica import Replica, ServiceRequest
from forge.types import ServiceConfig

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


# TODO - tie this into metric logger when it exists
@dataclass
class ReplicaMetrics:
    """
    Metrics collection for a single replica instance.

    Tracks request counts, timing metrics, current state, and session assignments
    for performance monitoring and autoscaling decisions.

    Attributes:
        replica_idx: Unique identifier for this replica
        total_requests: Total number of requests processed
        successful_requests: Number of successfully completed requests
        failed_requests: Number of failed requests
        request_times: Sliding window of request start timestamps
        request_latencies: Sliding window of request completion latencies
        active_requests: Currently processing requests
        queue_depth: Number of pending requests in queue
        assigned_sessions: Number of sessions assigned to this replica
    """

    replica_idx: int
    # Request metrics
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    # Timing metrics (sliding window)
    request_times: deque = field(default_factory=lambda: deque(maxlen=100))
    request_latencies: deque = field(default_factory=lambda: deque(maxlen=100))
    # Current state
    active_requests: int = 0
    queue_depth: int = 0
    # Session metrics
    assigned_sessions: int = 0

    def add_request_start(self, timestamp: float):
        """Record when a request starts processing."""
        self.request_times.append(timestamp)
        self.total_requests += 1

    def add_request_completion(self, start_time: float, success: bool):
        """Record when a request completes."""
        latency = time.time() - start_time
        self.request_latencies.append(latency)
        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1

    def get_request_rate(self, window_seconds: float = 60.0) -> float:
        """Get requests per second over the last window_seconds."""
        now = time.time()
        cutoff = now - window_seconds
        recent_requests = [t for t in self.request_times if t >= cutoff]
        return len(recent_requests) / window_seconds if window_seconds > 0 else 0.0

    def get_avg_latency(self, window_requests: int = 50) -> float:
        """Get average latency over the last N requests."""
        if not self.request_latencies:
            return 0.0
        recent_latencies = list(self.request_latencies)[-window_requests:]
        return sum(recent_latencies) / len(recent_latencies)

    def get_capacity_utilization(self, max_concurrent: int) -> float:
        """Get current capacity utilization (0.0 to 1.0)."""
        return self.active_requests / max_concurrent if max_concurrent > 0 else 0.0


@dataclass
class ServiceMetrics:
    """
    Aggregated metrics collection for the entire service.

    Provides service-wide visibility into performance, health, and scaling metrics
    by aggregating data from all replica instances.

    Attributes:
        replica_metrics: Per-replica metrics indexed by replica ID
        total_sessions: Number of active sessions across all replicas
        healthy_replicas: Number of currently healthy replicas
        total_replicas: Total number of replicas (healthy + unhealthy)
        last_scale_event: Timestamp of the last scaling operation
    """

    # Replica metrics
    replica_metrics: Dict[int, ReplicaMetrics] = field(default_factory=dict)
    # Service-level metrics
    total_sessions: int = 0
    healthy_replicas: int = 0
    total_replicas: int = 0
    # Time-based metrics
    last_scale_event: float = 0.0

    def get_total_request_rate(self, window_seconds: float = 60.0) -> float:
        """Get total requests per second across all replicas."""
        return sum(
            metrics.get_request_rate(window_seconds)
            for metrics in self.replica_metrics.values()
        )

    def get_avg_queue_depth(self) -> float:
        """Get average queue depth across all healthy replicas."""
        healthy_metrics = [
            m
            for m in self.replica_metrics.values()
            if m.replica_idx < self.healthy_replicas
        ]
        if not healthy_metrics:
            return 0.0
        return sum(m.queue_depth for m in healthy_metrics) / len(healthy_metrics)

    def get_avg_capacity_utilization(self, replicas: List) -> float:
        """Get average capacity utilization across all healthy replicas."""
        healthy_replicas = [r for r in replicas if r.healthy]
        if not healthy_replicas:
            return 0.0

        utilizations = []
        for replica in healthy_replicas:
            if replica.idx in self.replica_metrics:
                metrics = self.replica_metrics[replica.idx]
                utilization = metrics.get_capacity_utilization(
                    replica.max_concurrent_requests
                )
                utilizations.append(utilization)

        return sum(utilizations) / len(utilizations) if utilizations else 0.0

    def get_sessions_per_replica(self) -> float:
        """Get average sessions per healthy replica."""
        if self.healthy_replicas == 0:
            return 0.0
        return self.total_sessions / self.healthy_replicas


@dataclass
class Session:
    session_id: str


# Global context variable for session state
# This is used to propagate session state across async tasks
_session_context: contextvars.ContextVar[dict | None] = contextvars.ContextVar(
    "session_context", default=None
)


class SessionContext:
    """Context manager for service sessions using context variables."""

    def __init__(self, service: "Service", **session_kwargs):
        self.service = service
        self.session_id: str | None = None
        self.session_kwargs = session_kwargs
        self._token = None

    async def __aenter__(self):
        """Start a session and set context variables."""
        self.session_id = await self.service.start_session()
        # Set context for this async task
        context_value = {"session_id": self.session_id, "kwargs": self.session_kwargs}
        self._token = _session_context.set(context_value)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Terminate the session and restore context."""
        if self._token:
            _session_context.reset(self._token)
        if self.session_id:
            await self.service.terminate_session(self.session_id)
            self.session_id = None


class Service:
    """
    Distributed Actor Service Controller

    A sophisticated service orchestration system that manages multiple replicas of actor-based
    services with automatic scaling, fault tolerance, and intelligent load balancing.

    The Service acts as a unified interface for distributed workloads, automatically handling:
    - **Fault Tolerance**: Health monitoring, automatic replica recovery, request migration
    - **Load Balancing**: Round-robin, least-loaded, and session-affinity routing
    - **Session Management**: Stateful session handling with context propagation
    - **Metrics Collection**: Comprehensive performance and health monitoring

    Args:
        cfg: Service configuration including number of replicas, GPUs per replica, and health polling rate
        actor_def: Actor class definition to instantiate on each replica
        *actor_args: Positional arguments passed to actor constructor
        **actor_kwargs: Keyword arguments passed to actor constructor

    Example:
        Basic setup with autoscaling:

        >>> config = ServiceConfig(
        ...     gpus_per_replica=1,
        ...     num_replicas=3,
        ... )
        >>> service = Service(config, MyActorClass, model_path="/path/to/model")
        >>> await service.__initialize__()

        Session-based usage:

        >>> async with service.session():
        ...     result1 = await service.my_endpoint(arg1, arg2)
        ...     result2 = await service.another_endpoint(arg3)

        Stateless usage:

        >>> result = await service.my_endpoint(arg1, arg2)  # Uses round-robin

    Attributes:
        _cfg: Service configuration
        _replicas: List of managed replica instances
        _active_sessions: Currently active sessions
        _metrics: Aggregated service and replica metrics
        _endpoints: Dynamically registered actor endpoints
    """

    def __init__(self, cfg: ServiceConfig, actor_def, *actor_args, **actor_kwargs):
        self._cfg = cfg
        self._replicas = []
        self._actor_def = actor_def
        self._actor_args = actor_args
        self._actor_kwargs = actor_kwargs

        self._active_sessions = []
        self._id_session_map = {}
        self._session_replica_map: Dict[str, int] = {}
        self._next_replica_idx = 0  # For round-robin load balancing

        # Initialize metrics collection
        self._metrics = ServiceMetrics()
        self._health_task = None
        self._shutdown_requested = False

        # Replica initialization queue
        self._replicas_to_init = []

        # For all endpoints within the actor_def, create an interface from it
        self._endpoints = []
        for func_name in dir(actor_def):
            func = getattr(actor_def, func_name)
            if isinstance(func, EndpointProperty):
                logger.debug("Registering endpoint %s", func_name)
                self._endpoints.append(func_name)
                # Dynamically add this endpoint method to the Service class
                self._add_endpoint_method(func_name)

    async def __initialize__(self):
        logger.debug("Starting service up with %d replicas.", self._cfg.num_replicas)
        replicas = []
        num_replicas = self._cfg.num_replicas
        for i in range(num_replicas):
            mesh = RecoverableProcMesh(proc_config=self._cfg.to_process_config())
            replica = Replica(
                proc_mesh=mesh,
                actor=None,
                idx=len(self._replicas) + i,
                max_concurrent_requests=self._cfg.replica_max_concurrent_requests,
                return_first_rank_result=self._cfg.return_first_rank_result,
            )
            replicas.append(replica)

        # Initializing should only happen in the health_loop
        # and during the first initialization.
        # If multiple parts of the code try to initialize replicas at
        # the same time, it can cause nasty race conditions
        # (e.g., double initialization, inconsistent state, or resource conflicts).
        # By funneling all replica initialization through a single queue and the
        # health loop, we ensure safe, serialized initialization.
        logger.debug(
            "Queued %d replicas for initialization. Total replicas: %d",
            num_replicas,
            len(self._replicas),
        )
        self._replicas_to_init.extend(replicas)
        await self._maybe_init_replicas()
        self._replicas.extend(replicas)

        # Start the health loop in the background
        self._health_task = asyncio.create_task(
            self._health_loop(poll_rate_s=self._cfg.health_poll_rate)
        )

    def _add_endpoint_method(self, endpoint_name: str):
        """Dynamically adds an endpoint method to this Service instance."""

        async def endpoint_method(sess_id: str | None = None, *args, **kwargs):
            return await self._call(sess_id, endpoint_name, *args, **kwargs)

        # Set the method on this instance
        setattr(self, endpoint_name, endpoint_method)

    async def _call(self, sess_id: str | None, function: str, *args, **kwargs):
        """
        Routes a function call to the appropriate replica with load balancing and fault tolerance.

        This is the core routing method that handles:
        - Session-based routing for stateful calls
        - Round-robin load balancing for stateless calls
        - Custom routing based on context hints
        - Automatic retry on replica failures
        - Request queuing and processing

        Args:
            sess_id: Optional session ID for stateful routing
            function: Name of the actor endpoint to call
            *args: Positional arguments to pass to the endpoint
            **kwargs: Keyword arguments to pass to the endpoint

        Returns:
            The result from the actor endpoint execution

        Raises:
            RuntimeError: If no healthy replicas are available
            Exception: Any exception raised by the actor endpoint
        """
        # Check context variables for session state if no explicit sess_id
        if sess_id is None:
            ctx = _session_context.get()
            if ctx:
                sess_id = ctx["session_id"]
                routing_hints = ctx["kwargs"]
            else:
                routing_hints = {}
        else:
            routing_hints = {}

        replica = await self._get_replica(sess_id, **routing_hints)

        # Create a ServiceRequest object to queue
        request = ServiceRequest(
            session_id=sess_id,
            function=function,
            args=args,
            kwargs=kwargs,
            future=asyncio.Future(),
        )

        # Queue the request using replica's method
        await replica.enqueue_request(request)

        # Start the replica processing loop if not already running
        if not replica._running:
            asyncio.create_task(replica.run())

        # Wait for the result
        try:
            return await request.future
        except Exception as e:
            # If the replica failed, try to retry once
            if not replica.healthy:
                logger.debug(
                    "Replica %d failed during request, retrying on healthy replica",
                    replica.idx,
                )
                return await self._retry_request_on_healthy_replica(
                    sess_id, function, *args, **kwargs
                )
            raise

    async def _retry_request_on_healthy_replica(
        self, sess_id: str | None, function: str, *args, **kwargs
    ):
        """Retries a failed request on a healthy replica."""
        # Force reassignment to a healthy replica (only for session-based calls)
        if sess_id is not None and sess_id in self._session_replica_map:
            del self._session_replica_map[sess_id]

        # Retry the call (this will assign to a new healthy replica)
        return await self._call(sess_id, function, *args, **kwargs)

    async def _migrate_remaining_requests(self, failed_replica: Replica):
        """Migrates remaining requests from a failed replica to healthy replicas."""
        migrated_requests = []

        # Collect all remaining requests
        while not failed_replica.request_queue.empty():
            try:
                request = failed_replica.request_queue.get_nowait()
                migrated_requests.append(request)
            except asyncio.QueueEmpty:
                break

        if not migrated_requests:
            return

        logger.debug(
            "Migrating %d requests from failed replica %d",
            len(migrated_requests),
            failed_replica.idx,
        )

        # Find healthy replicas
        healthy_replicas = [
            r for r in self._replicas if r.healthy and r != failed_replica
        ]

        if not healthy_replicas:
            # No healthy replicas, fail all requests
            for request in migrated_requests:
                request.future.set_exception(
                    RuntimeError("No healthy replicas available")
                )
            return

        # Distribute requests among healthy replicas
        for i, request in enumerate(migrated_requests):
            target_replica = healthy_replicas[i % len(healthy_replicas)]
            await target_replica.enqueue_request(request)

            # Start replica processing if not running
            if not target_replica._running:
                asyncio.create_task(target_replica.run())

            # Update session mapping if needed
            sess_id = request.session_id
            if (
                sess_id in self._session_replica_map
                and self._session_replica_map[sess_id] == failed_replica.idx
            ):
                self._session_replica_map[sess_id] = target_replica.idx

    async def start_session(self) -> str:
        """
        Starts a new session for stateful request handling.

        Sessions enable request affinity to specific replicas, maintaining state
        consistency for workloads that require it. Each session gets a unique ID
        and is automatically assigned to the least loaded replica.

        Returns:
            str: Unique session identifier for use in subsequent requests

        Example:
            >>> session_id = await service.start_session()
            >>> result = await service.my_endpoint(session_id, arg1, arg2)
            >>> await service.terminate_session(session_id)
        """
        sess_id = str(uuid.uuid4())
        session = Session(session_id=sess_id)
        self._active_sessions.append(session)

        # Update metrics
        self._update_service_metrics()

        return sess_id

    def session(self, **kwargs) -> SessionContext:
        """Returns a context manager for session-based calls."""
        return SessionContext(self, **kwargs)

    def _update_service_metrics(self):
        """Updates service-level metrics."""
        self._metrics.total_sessions = len(self._active_sessions)
        self._metrics.total_replicas = len(self._replicas)
        self._metrics.healthy_replicas = sum(1 for r in self._replicas if r.healthy)

        # Update queue depths for all replicas
        for replica in self._replicas:
            if replica.idx not in self._metrics.replica_metrics:
                self._metrics.replica_metrics[replica.idx] = ReplicaMetrics(replica.idx)

            replica_metrics = self._metrics.replica_metrics[replica.idx]
            replica_metrics.queue_depth = replica.request_queue.qsize()
            replica_metrics.active_requests = replica.active_requests

        # Update session assignments per replica
        session_counts = defaultdict(int)
        for sess_id, replica_idx in self._session_replica_map.items():
            session_counts[replica_idx] += 1

        for replica_idx, count in session_counts.items():
            if replica_idx in self._metrics.replica_metrics:
                self._metrics.replica_metrics[replica_idx].assigned_sessions = count

    def get_metrics(self) -> ServiceMetrics:
        """
        Get comprehensive service metrics for monitoring and analysis.

        Returns detailed metrics including per-replica performance data,
        service-wide aggregations, and health status information.

        Returns:
            ServiceMetrics: Complete metrics object with replica and service data

        Example:
            >>> metrics = service.get_metrics()
            >>> print(f"Request rate: {metrics.get_total_request_rate():.1f} req/s")
            >>> print(f"Queue depth: {metrics.get_avg_queue_depth():.1f}")
        """
        self._update_service_metrics()
        return self._metrics

    def get_metrics_summary(self) -> dict:
        """
        Get a summary of key metrics for monitoring and debugging.

        Provides a structured summary of service and replica metrics in a format
        suitable for monitoring dashboards, logging, or debugging purposes.

        Returns:
            dict: Structured metrics summary with service and per-replica data

        Example:
            >>> summary = service.get_metrics_summary()
            >>> print(f"Healthy replicas: {summary['service']['healthy_replicas']}")
            >>> for idx, metrics in summary['replicas'].items():
            ...     print(f"Replica {idx}: {metrics['request_rate']:.1f} req/s")
        """
        self._update_service_metrics()

        summary = {
            "service": {
                "total_sessions": self._metrics.total_sessions,
                "healthy_replicas": self._metrics.healthy_replicas,
                "total_replicas": self._metrics.total_replicas,
                "total_request_rate": self._metrics.get_total_request_rate(),
                "avg_queue_depth": self._metrics.get_avg_queue_depth(),
                "avg_capacity_utilization": self._metrics.get_avg_capacity_utilization(
                    self._replicas
                ),
                "sessions_per_replica": self._metrics.get_sessions_per_replica(),
            },
            "replicas": {},
        }

        for replica_idx, metrics in self._metrics.replica_metrics.items():
            summary["replicas"][replica_idx] = {
                "total_requests": metrics.total_requests,
                "successful_requests": metrics.successful_requests,
                "failed_requests": metrics.failed_requests,
                "request_rate": metrics.get_request_rate(),
                "avg_latency": metrics.get_avg_latency(),
                "active_requests": metrics.active_requests,
                "queue_depth": metrics.queue_depth,
                "assigned_sessions": metrics.assigned_sessions,
                "capacity_utilization": metrics.get_capacity_utilization(10),
            }

        return summary

    async def terminate_session(self, sess_id: str):
        """
        Terminates an active session and cleans up associated resources.

        Removes the session from active tracking, clears replica assignments,
        and updates service metrics. Sessions should be terminated when no
        longer needed to free up resources.

        Args:
            sess_id: The unique session identifier to terminate

        Example:
            >>> session_id = await service.start_session()
            >>> # ... use session for requests ...
            >>> await service.terminate_session(session_id)
        """
        logger.debug("Terminating session %s", sess_id)

        # Remove from active sessions
        self._active_sessions = [
            s for s in self._active_sessions if s.session_id != sess_id
        ]

        # Remove from session-replica mapping
        if sess_id in self._session_replica_map:
            del self._session_replica_map[sess_id]

        # Update metrics
        self._update_service_metrics()

    async def _health_loop(self, poll_rate_s: float):
        """Runs the health loop to monitor and recover replicas.

        This loop continuously checks the health of replicas and recovers
        failed replicas by reinitializing their proc_meshes. It also
        periodically updates service metrics to reflect the current state.

        """
        while not self._shutdown_requested:
            # Process any replicas that need initialization
            await self._maybe_init_replicas()

            # Check for failed replicas and recover them
            failed_replicas = []
            for replica in self._replicas:
                if replica.proc_mesh.failed:
                    failed_replicas.append(replica)

            if any(failed_replicas):
                logger.debug(
                    "[HEALTH LOOP] Detected %d failed replicas: %s",
                    len(failed_replicas),
                    pprint.pformat(failed_replicas),
                )
                self._replicas_to_init.extend(failed_replicas)

            await asyncio.sleep(poll_rate_s)

    async def _custom_replica_routing(
        self, sess_id: str | None, **kwargs
    ) -> Optional[Replica]:
        """Hook for custom routing logic. Override in subclasses to implement custom routing."""
        return None

    def _get_next_replica(self) -> "Replica":
        """Get the next replica using round-robin selection."""
        healthy_replicas = [r for r in self._replicas if r.healthy]
        if not healthy_replicas:
            raise RuntimeError("No healthy replicas available for load balancing")

        # Simple round-robin
        self._next_replica_idx = (self._next_replica_idx + 1) % len(healthy_replicas)
        return healthy_replicas[self._next_replica_idx]

    def _get_least_loaded_replica(self) -> "Replica":
        """Get the replica with the lowest load."""
        healthy_replicas = [r for r in self._replicas if r.healthy]
        if not healthy_replicas:
            raise RuntimeError("No healthy replicas available for session assignment")

        # Load = active_requests + queue_depth
        def get_load(replica: "Replica") -> int:
            return replica.active_requests + replica.request_queue.qsize()

        return min(healthy_replicas, key=get_load)

    async def _get_replica(self, sess_id: str | None, **kwargs) -> "Replica":
        """Get a replica for the given session ID, with optional custom routing hints."""
        # Try custom routing first if hints are provided
        if kwargs:
            custom_result = await self._custom_replica_routing(sess_id, **kwargs)
            if custom_result is not None:
                return custom_result

        # Default routing logic
        if sess_id is None:
            # No session, use round-robin load balancing
            replica = self._get_next_replica()
            return replica

        # Session-based routing
        if sess_id in self._session_replica_map:
            replica_idx = self._session_replica_map[sess_id]
            # Find the replica with this index
            for replica in self._replicas:
                if replica.idx == replica_idx and replica.healthy:
                    return replica
            # If the replica is no longer healthy, remove from session map and reassign
            del self._session_replica_map[sess_id]

        # New session, assign to least loaded replica
        replica = self._get_least_loaded_replica()
        self._session_replica_map[sess_id] = replica.idx
        logger.debug("Assigning session %s to replica %d", sess_id, replica.idx)
        return replica

    async def stop(self):
        logger.debug("Stopping service...")
        # Signal shutdown to health loop
        self._shutdown_requested = True

        # Wait for health loop to finish gracefully
        if self._health_task is not None:
            try:
                await asyncio.wait_for(self._health_task, timeout=5.0)
                logger.info("Health loop stopped gracefully.")
            except asyncio.TimeoutError:
                logger.warning("Health loop didn't stop gracefully, cancelling...")
                self._health_task.cancel()
                try:
                    await self._health_task
                except asyncio.CancelledError:
                    logger.info("Health loop task cancelled.")

        # Stop all replicas using their stop method
        await asyncio.gather(
            *[replica.stop() for replica in self._replicas],
            return_exceptions=True,
        )

    async def _maybe_init_replicas(self):
        """Initializes replicas that are queued for initialization."""
        if not self._replicas_to_init:
            return

        logger.debug("Init replicas: %s", pprint.pformat(self._replicas_to_init))

        def _recover_hook(
            replica: Replica,
        ) -> Callable[[ProcMesh], Coroutine[Any, Any, None]]:
            async def inner_hook(proc_mesh: ProcMesh) -> None:
                if "name" in self._actor_kwargs:
                    actor_name = self._actor_kwargs.pop("name")
                else:
                    actor_name = self._actor_def.__name__
                # TODO - expand support so name can stick within kwargs
                actor = await proc_mesh.spawn(
                    actor_name,
                    self._actor_def,
                    *self._actor_args,
                    **self._actor_kwargs,
                )
                replica.actor = actor
                # Use replica's setup method instead of inline setup
                await replica.setup()

            return inner_hook

        await asyncio.gather(
            *[
                replica.proc_mesh.spawn(_recover_hook(replica))
                for replica in self._replicas_to_init
            ]
        )
        self._replicas_to_init.clear()

    async def _migrate_replica_workload(self, replica_to_remove: Replica):
        """Migrates all workload from a replica that's being removed."""
        # Migrate queued requests
        await self._migrate_remaining_requests(replica_to_remove)

        # Reassign sessions to other replicas
        sessions_to_reassign = [
            sess_id
            for sess_id, replica_idx in self._session_replica_map.items()
            if replica_idx == replica_to_remove.idx
        ]

        for sess_id in sessions_to_reassign:
            del self._session_replica_map[sess_id]
            logger.debug("Session %s will be reassigned on next request", sess_id)

    def __repr__(self):
        return f"Service(actor={self._actor_def.__name__})"
