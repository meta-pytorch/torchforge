# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""
Orchestrator interface and session management.

This module provides the user-facing API for interacting with distributed orchestrators,
including session management, context propagation, and dynamic endpoint registration.
"""

import contextvars
import logging
from dataclasses import dataclass
from typing import Generic, List, ParamSpec, TypeVar

from monarch._src.actor.endpoint import EndpointProperty

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

P = ParamSpec("P")
R = TypeVar("R")


@dataclass
class Session:
    """Simple session data holder."""

    session_id: str


# Context variable for session state
_session_context = contextvars.ContextVar("session_context")


class SessionContext:
    """
    Async context manager for stateful orchestrator sessions with automatic lifecycle management.

    Provides a convenient way to maintain stateful connections to replicas across multiple
    requests. Sessions ensure that all requests within the context are routed to the same
    replica, enabling stateful interactions while handling session lifecycle automatically.

    Example:

        >>> async with orchestrator.session() as session:
        ...     # All calls within this block use the same replica
        ...     result1 = await orchestrator.my_endpoint(arg1)
        ...     result2 = await orchestrator.another_endpoint(result1)

    """

    def __init__(self, orchestrator: "OrchestratorInterface"):
        self.orchestrator = orchestrator
        self.session_id: str | None = None
        self._token = None

    async def __aenter__(self):
        """Start a session and set context variables."""
        self.session_id = await self.orchestrator.start_session()
        # Set context for this async task
        context_value = {"session_id": self.session_id}
        self._token = _session_context.set(context_value)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Terminate the session and restore context."""
        if self._token:
            _session_context.reset(self._token)
        if self.session_id:
            await self.orchestrator.terminate_session(self.session_id)
            self.session_id = None


class OrchestratorEndpoint(Generic[P, R]):
    """An endpoint object specific to orchestrators.

    This loosely mimics the Endpoint APIs exposed in Monarch, with
    a few key differences:
    - Only choose and call are retained (dropping stream and call_one)
    - Call returns a list directly rather than a ValueMesh.

    These changes are made with Forge use cases in mind, but can
    certainly be expanded/adapted in the future.

    """

    def __init__(self, orchestrator, endpoint_name: str):
        self.orchestrator = orchestrator
        self.endpoint_name = endpoint_name

    async def choose(self, *args: P.args, **kwargs: P.kwargs) -> R:
        """Chooses a replica to call based on context and load balancing strategy."""
        # Extract sess_id from kwargs if present
        sess_id = kwargs.pop("sess_id", None)
        return await self.orchestrator._call(
            sess_id, self.endpoint_name, *args, **kwargs
        )

    async def call(self, *args: P.args, **kwargs: P.kwargs) -> List[R]:
        """Broadcasts a request to all healthy replicas and returns the results as a list."""
        result = await self.orchestrator.call_all(self.endpoint_name, *args, **kwargs)
        return result


class OrchestratorEndpointV2(Generic[P, R]):
    """An endpoint object specific to orchestrators.

    This loosely mimics the Endpoint APIs exposed in Monarch, with
    a few key differences:
    - Only choose and call are retained (dropping stream and call_one)
    - Call returns a list directly rather than a ValueMesh.

    These changes are made with Forge use cases in mind, but can
    certainly be expanded/adapted in the future.

    """

    def __init__(self, orchestrator_actor, endpoint_name: str):
        self.orchestrator_actor = orchestrator_actor
        self.endpoint_name = endpoint_name

    async def choose(self, *args: P.args, **kwargs: P.kwargs) -> R:
        """Chooses a replica to call based on context and load balancing strategy."""
        # Extract sess_id from kwargs if present
        sess_id = kwargs.pop("sess_id", None)
        return await self.orchestrator_actor.call.call_one(
            sess_id, self.endpoint_name, *args, **kwargs
        )

    async def call(self, *args: P.args, **kwargs: P.kwargs) -> List[R]:
        """Broadcasts a request to all healthy replicas and returns the results as a list."""
        result = await self.orchestrator_actor.call_all.call_one(
            self.endpoint_name, *args, **kwargs
        )
        return result


class OrchestratorInterface:
    """
    A lightweight interface to the base Orchestrator class.

    This is a temporary workaround until Monarch supports nested
    actors.

    """

    def __init__(self, _orchestrator, actor_def):
        self._orchestrator = _orchestrator
        self.actor_def = actor_def

        # Dynamically create OrchestratorEndpoint objects for user's actor endpoints
        # Inspect the actor_def directly to find endpoints
        for attr_name in dir(actor_def):
            attr_value = getattr(actor_def, attr_name)
            if isinstance(attr_value, EndpointProperty):
                # Create a OrchestratorEndpoint that will route through the Orchestrator
                endpoint = OrchestratorEndpoint(self._orchestrator, attr_name)
                setattr(self, attr_name, endpoint)

    # Session management methods - handled by OrchestratorInterface
    async def start_session(self) -> str:
        """Starts a new session for stateful request handling."""
        return await self._orchestrator.start_session()

    async def terminate_session(self, sess_id: str):
        """Terminates an active session and cleans up associated resources."""
        return await self._orchestrator.terminate_session(sess_id)

    def session(self) -> "SessionContext":
        """Returns a context manager for session-based calls."""
        return SessionContext(self)

    async def get_metrics(self):
        """Get comprehensive orchestrator metrics for monitoring and analysis."""
        return self._orchestrator.get_metrics()

    async def get_metrics_summary(self):
        """Get a summary of key metrics for monitoring and debugging."""
        return self._orchestrator.get_metrics_summary()

    # Testing method - forwarded to Orchestrator
    async def _get_internal_state(self):
        """
        Get comprehensive internal state for testing purposes.

        Returns:
            dict: Complete internal state including sessions, replicas, and metrics
        """
        return await self._orchestrator._get_internal_state()

    def __getattr__(self, name: str):
        """Forward all other attribute access to the underlying Orchestrator."""
        _orchestrator = object.__getattribute__(self, "_orchestrator")
        # Forward everything else to the _orchestrator
        if hasattr(_orchestrator, name):
            return getattr(_orchestrator, name)
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'"
        )


class OrchestratorInterfaceV2:
    """
    A lightweight interface to a Orchestrator Actor running on a single-node mesh.

    This interface holds references to the proc_mesh and actor_mesh (both of size 1)
    and exposes its user-defined actor endpoints as OrchestratorEndpoint objects that
    route through the Orchestrator Actor's _call and _call_all endpoints.

    The OrchestratorInterface acts as the handle that is returned to end clients,
    providing a simple interface that makes actual calls to the Orchestrator Actor.

    This is also needed to simplify serializing a handle to the orchestrator, in case
    we want to pass this to other actors in the future.

    """

    def __init__(self, _proc_mesh, _orchestrator_actor, actor_def):
        self._proc_mesh = _proc_mesh
        self._orchestrator_actor = _orchestrator_actor
        self.actor_def = actor_def

        # Dynamically create OrchestratorEndpoint objects for user's actor endpoints
        # Inspect the actor_def directly to find endpoints
        for attr_name in dir(actor_def):
            attr_value = getattr(actor_def, attr_name)
            if isinstance(attr_value, EndpointProperty):
                # Create a OrchestratorEndpoint that will route through the Orchestrator Actor
                endpoint = OrchestratorEndpointV2(self._orchestrator_actor, attr_name)
                setattr(self, attr_name, endpoint)

    # Session management methods - handled by OrchestratorInterface
    async def start_session(self) -> str:
        """Starts a new session for stateful request handling."""
        return await self._orchestrator_actor.start_session.call_one()

    async def terminate_session(self, sess_id: str):
        """Terminates an active session and cleans up associated resources."""
        return await self._orchestrator_actor.terminate_session.call_one(sess_id)

    def session(self) -> "SessionContext":
        """Returns a context manager for session-based calls."""
        return SessionContext(self)

    # Metrics methods - forwarded to Orchestrator Actor
    async def get_metrics(self):
        """Get comprehensive orchestrator metrics for monitoring and analysis."""
        return await self._orchestrator_actor.get_metrics.call_one()

    async def get_metrics_summary(self):
        """Get a summary of key metrics for monitoring and debugging."""
        return await self._orchestrator_actor.get_metrics_summary.call_one()

    # Testing method - forwarded to Orchestrator Actor
    async def _get_internal_state(self):
        """
        Get comprehensive internal state for testing purposes.

        Returns:
            dict: Complete internal state including sessions, replicas, and metrics
        """
        return await self._orchestrator_actor._get_internal_state.call_one()

    def __getattr__(self, name: str):
        """Forward all other attribute access to the underlying Orchestrator."""
        _orchestrator_actor = object.__getattribute__(self, "_orchestrator_actor")
        # Forward everything else to the _orchestrator_actor
        if hasattr(_orchestrator_actor, name):
            return getattr(_orchestrator_actor, name)
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'"
        )
