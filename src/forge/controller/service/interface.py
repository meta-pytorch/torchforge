# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""
Service interface and session management.

This module provides the user-facing API for interacting with distributed services,
including session management, context propagation, and dynamic endpoint registration.
"""

import contextvars
from dataclasses import dataclass

from monarch._src.actor.endpoint import EndpointProperty

from .endpoint import ServiceEndpoint, ServiceEndpointV2


@dataclass
class Session:
    """Simple session data holder."""

    session_id: str


# Context variable for session state
_session_context = contextvars.ContextVar("session_context")


class SessionContext:
    """
    Async context manager for stateful service sessions with automatic lifecycle management.

    Provides a convenient way to maintain stateful connections to replicas across multiple
    requests. Sessions ensure that all requests within the context are routed to the same
    replica, enabling stateful interactions while handling session lifecycle automatically.

    Example:

        >>> async with service.session() as session:
        ...     # All calls within this block use the same replica
        ...     result1 = await service.my_endpoint(arg1)
        ...     result2 = await service.another_endpoint(result1)

    """

    def __init__(self, service: "ServiceInterface"):
        self.service = service
        self.session_id: str | None = None
        self._token = None

    async def __aenter__(self):
        """Start a session and set context variables."""
        self.session_id = await self.service.start_session()
        # Set context for this async task
        context_value = {"session_id": self.session_id}
        self._token = _session_context.set(context_value)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Terminate the session and restore context."""
        if self._token:
            _session_context.reset(self._token)
        if self.session_id:
            await self.service.terminate_session(self.session_id)
            self.session_id = None


class ServiceInterface:
    """
    A lightweight interface to the base Service class.

    This is a temporary workaround until Monarch supports nested
    actors.

    """

    def __init__(self, _service, actor_def):
        self._service = _service
        self.actor_def = actor_def

        # Dynamically create ServiceEndpoint objects for user's actor endpoints
        # Inspect the actor_def directly to find endpoints
        for attr_name in dir(actor_def):
            attr_value = getattr(actor_def, attr_name)
            if hasattr(attr_value, "_service_endpoint_config"):
                # Decorated with @service_endpoint
                # Create a ServiceEndpoint with batch routing config
                cfg = attr_value._service_endpoint_config
                endpoint = ServiceEndpoint(
                    self._service,
                    attr_name,
                    router=cfg["router"],
                    batch_size=cfg["batch_size"],
                    batch_timeout=cfg["batch_timeout"],
                )

            elif isinstance(attr_value, EndpointProperty):
                # Decorated with @endpoint
                # Create a ServiceEndpoint that will route through the Service Actor
                endpoint = ServiceEndpoint(self._service, attr_name)
            else:
                # Not decorated with @endpoint or @service_endpoint
                raise ValueError(
                    f"Attribute '{attr_name}' is not a valid service endpoint"
                )
            setattr(self, attr_name, endpoint)

    # Session management methods - handled by ServiceInterface
    async def start_session(self) -> str:
        """Starts a new session for stateful request handling."""
        return await self._service.start_session()

    async def terminate_session(self, sess_id: str):
        """Terminates an active session and cleans up associated resources."""
        return await self._service.terminate_session(sess_id)

    async def shutdown(self) -> None:
        """
        Shut down the underlying Service and all endpoints.
        """
        for attr in dir(self):
            ep = getattr(self, attr)
            if isinstance(ep, ServiceEndpoint):
                await ep.stop()
        await self._service.stop()

    def session(self) -> "SessionContext":
        """Returns a context manager for session-based calls."""
        return SessionContext(self)

    async def get_metrics(self):
        """Get comprehensive service metrics for monitoring and analysis."""
        return self._service.get_metrics()

    async def get_metrics_summary(self):
        """Get a summary of key metrics for monitoring and debugging."""
        return self._service.get_metrics_summary()

    # Testing method - forwarded to Service Actor
    async def _get_internal_state(self):
        """
        Get comprehensive internal state for testing purposes.

        Returns:
            dict: Complete internal state including sessions, replicas, and metrics
        """
        return await self._service._get_internal_state()

    def __getattr__(self, name: str):
        """Forward all other attribute access to the underlying Service Actor."""
        _service = object.__getattribute__(self, "_service")
        # Forward everything else to the _service
        if hasattr(_service, name):
            return getattr(_service, name)
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'"
        )


class ServiceInterfaceV2:
    """
    A lightweight interface to a Service Actor running on a single-node mesh.

    This interface holds references to the proc_mesh and actor_mesh (both of size 1)
    and exposes its user-defined actor endpoints as ServiceEndpoint objects that
    route through the Service Actor's _call and _call_all endpoints.

    The ServiceInterface acts as the handle that is returned to end clients,
    providing a simple interface that makes actual calls to the Service Actor.

    This is also needed to simplify serializing a handle to the service, in case
    we want to pass this to other actors in the future.

    """

    def __init__(self, _proc_mesh, _service, actor_def):
        self._proc_mesh = _proc_mesh
        self._service = _service
        self.actor_def = actor_def

        # Dynamically create ServiceEndpoint objects for user's actor endpoints
        # Inspect the actor_def directly to find endpoints
        for attr_name in dir(actor_def):
            attr_value = getattr(actor_def, attr_name)
            if isinstance(attr_value, EndpointProperty):
                # Create a ServiceEndpoint that will route through the Service Actor
                endpoint = ServiceEndpointV2(self._service, attr_name)
                setattr(self, attr_name, endpoint)

    # Session management methods - handled by ServiceInterface
    async def start_session(self) -> str:
        """Starts a new session for stateful request handling."""
        return await self._service.start_session.call_one()

    async def terminate_session(self, sess_id: str):
        """Terminates an active session and cleans up associated resources."""
        return await self._service.terminate_session.call_one(sess_id)

    def session(self) -> "SessionContext":
        """Returns a context manager for session-based calls."""
        return SessionContext(self)

    # Metrics methods - forwarded to Service Actor
    async def get_metrics(self):
        """Get comprehensive service metrics for monitoring and analysis."""
        return await self._service.get_metrics.call_one()

    async def get_metrics_summary(self):
        """Get a summary of key metrics for monitoring and debugging."""
        return await self._service.get_metrics_summary.call_one()

    # Testing method - forwarded to Service Actor
    async def _get_internal_state(self):
        """
        Get comprehensive internal state for testing purposes.

        Returns:
            dict: Complete internal state including sessions, replicas, and metrics
        """
        return await self._service._get_internal_state.call_one()

    def __getattr__(self, name: str):
        """Forward all other attribute access to the underlying Service Actor."""
        _service = object.__getattribute__(self, "_service")
        # Forward everything else to the _service
        if hasattr(_service, name):
            return getattr(_service, name)
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'"
        )
