# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
OpenEnv Actor for sandboxed code execution.

This actor works with ANY OpenEnv environment using only raw dictionaries,
without requiring environment-specific packages (like julia_env or coding_env).

Usage:
    from openenv import GenericEnvClient, GenericAction

    # Create actor for any environment - just specify the Docker image
    actor = OpenEnvActor(
        docker_image="julia-env:latest",
        env_name="julia",
    )
    await actor.setup()

    # Execute with GenericAction (just a dict wrapper)
    action = GenericAction(core_code="println('hello')", test_code="@test true")
    result = await actor.execute(action)  # Returns StepResult with dict observation

    await actor.teardown()
"""

import logging
from typing import Any, Dict, Optional, TYPE_CHECKING

from monarch.actor import endpoint

if TYPE_CHECKING:
    from openenv import GenericEnvClient
    from openenv.core.client_types import StepResult

from forge.controller import ForgeActor
from forge.observability.metrics import record_metric, Reduce
from forge.actors.openenv_utils import (
    ContainerConfig,
    ContainerManager,
    ConnectionPool,
    is_connection_error,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class OpenEnvActor(ForgeActor):
    """A generic sandboxed execution environment using GenericEnvClient.

    This actor manages WebSocket connections to Docker containers,
    with connection pooling for high concurrency.

    Args:
        docker_image: Docker image name (e.g., "julia-env:latest")
        env_name: Environment name for logging (e.g., "julia", "python")
        env_vars: Environment variables to pass to containers
        container_timeout_s: Timeout for container startup
        request_timeout_s: Timeout for individual requests
        port: Starting port for containers
        container_memory_gb: Memory limit per container in GB
        enable_zombie_cleanup: Whether to enable zombie process cleanup
        num_connections: Total WebSocket connections to create
        num_containers: Number of Docker containers

    Usage:
        >>> actor = OpenEnvActor(
        ...     docker_image="julia-env:latest",
        ...     env_name="julia",
        ...     num_connections=16,
        ...     num_containers=2,
        ... )
        >>> await actor.setup()
        >>> action = {"core_code": "...", "test_code": "..."}
        >>> result = await actor.execute(action)
        >>> await actor.teardown()
    """

    def __init__(
        self,
        docker_image: str,
        env_name: str = "openenv",
        env_vars: Optional[Dict[str, str]] = None,
        container_timeout_s: float = 180.0,
        request_timeout_s: float = 120.0,
        port: int = 8000,
        container_memory_gb: int = 4,
        enable_zombie_cleanup: bool = False,
        num_connections: int = 1,
        num_containers: int = 1,
    ):
        self.num_connections = num_connections
        self.num_containers = num_containers
        self.request_timeout_s = request_timeout_s
        self.enable_zombie_cleanup = enable_zombie_cleanup

        # Container management
        self._container_config = ContainerConfig(
            docker_image=docker_image,
            env_name=env_name,
            env_vars=env_vars or {},
            port=port,
            memory_gb=container_memory_gb,
            timeout_s=container_timeout_s,
        )
        self._container_manager = ContainerManager(self._container_config)

        # Connection pool
        self._pool = ConnectionPool(request_timeout_s=request_timeout_s)

        # Backward compatibility
        self.client = None
        self.actual_port = None

    @endpoint
    async def setup(self):
        """Initialize containers and create sync connection pool with thread pool executor."""
        logger.info(
            f"Setting up: {self.num_connections} sync connections "
            f"across {self.num_containers} containers (with thread pool)"
        )

        # Create containers
        container_urls = self._container_manager.create_containers(self.num_containers)

        # Initialize thread pool and create sync connections
        await self._pool.initialize(num_connections=self.num_connections)
        self._pool.create_connections(container_urls, self.num_connections)

        # Backward compatibility
        if self._pool.clients:
            self.client = self._pool.clients[0]
            self.actual_port = self._container_config.port

    @endpoint
    async def recreate(self):
        """Resets the environment to a clean state."""
        import asyncio

        if not self.client:
            raise RuntimeError("Client not initialized. Call setup() first.")
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(self._pool._executor, self.client.reset)

    @endpoint
    async def execute(self, action: Dict[str, Any]) -> "StepResult[Dict[str, Any]]":
        """Execute an action using an available connection from the pool.

        Args:
            action: Dictionary action with keys like core_code, test_code.

        Returns:
            StepResult with observation dict.
        """
        if not self._pool.clients:
            raise RuntimeError("Connection pool not initialized. Call setup() first.")

        client_idx, client = await self._pool.acquire(timeout=self.request_timeout_s)

        try:
            return await self._execute_with_retry(client_idx, client, action)
        finally:
            await self._pool.release(client_idx)

    async def _execute_with_retry(
        self, client_idx: int, client: "GenericEnvClient", action: Dict[str, Any]
    ) -> "StepResult[Dict[str, Any]]":
        """Execute action with retry logic for connection errors.

        Uses thread pool to run sync WebSocket calls without blocking event loop.
        """
        max_retries = 3

        for attempt in range(max_retries):
            try:
                # Execute in thread pool - doesn't block event loop
                result = await self._pool.execute_step(client_idx, action)
                record_metric("pool/execute_success", 1, Reduce.SUM)
                return result

            except Exception as e:
                is_conn_error, error_type = is_connection_error(str(e))

                if is_conn_error and attempt < max_retries - 1:
                    record_metric(f"pool/{error_type}_error_count", 1, Reduce.SUM)
                    logger.error(f"{error_type} error on client {client_idx}: {e}")

                    try:
                        client = await self._pool.reconnect(
                            client_idx, self._container_manager.container_urls
                        )
                        record_metric("pool/reconnect_success", 1, Reduce.SUM)
                        continue
                    except Exception as reconnect_error:
                        logger.error(f"Reconnect failed: {reconnect_error}")
                        record_metric("pool/reconnect_failure", 1, Reduce.SUM)

                if is_conn_error:
                    raise RuntimeError(
                        f"Client {client_idx} failed after {max_retries} attempts: {e}"
                    ) from e
                raise

        raise RuntimeError("Execution failed after all retry attempts")

    @endpoint
    async def health_check(self) -> Dict[str, Any]:
        """Check if the environment is healthy."""
        import asyncio

        if not self._pool.clients:
            return {"healthy": False, "error": "Pool not initialized"}

        healthy_count = 0
        pool_status = []
        loop = asyncio.get_event_loop()

        for i, client in enumerate(self._pool.clients):
            try:
                # Run sync state() in thread pool
                await loop.run_in_executor(self._pool._executor, client.state)
                pool_status.append({"index": i, "healthy": True})
                healthy_count += 1
            except Exception as e:
                pool_status.append({"index": i, "healthy": False, "error": str(e)})

        return {
            "healthy": healthy_count > 0,
            "healthy_count": healthy_count,
            "total_clients": len(self._pool.clients),
            "pool_status": pool_status,
        }

    @endpoint
    async def get_pool_status(self) -> Dict[str, Any]:
        """Get connection pool status."""
        status = self._pool.get_status()
        status["num_containers"] = len(self._container_manager.container_urls)
        status["container_urls"] = self._container_manager.container_urls
        return status

    @endpoint
    async def get_state(self) -> Dict[str, Any]:
        """Get current environment state."""
        import asyncio

        if not self._pool.clients:
            raise RuntimeError("Pool not initialized. Call setup() first.")

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._pool._executor,
            self._pool.clients[0].state
        )

    @endpoint
    async def teardown(self):
        """Clean up all connections and containers."""
        logger.debug("Tearing down...")
        await self._pool.close_all()
        self._container_manager.stop_all()
        self.client = None
        logger.debug("Teardown complete.")

    @endpoint
    async def restart_container(self) -> Dict[str, Any]:
        """Restart all containers and reconnect the pool."""
        import asyncio

        logger.warning("Restarting all containers...")

        try:
            # Cleanup existing
            await self._pool.close_all()
            self._container_manager.stop_all()
            await asyncio.sleep(2)

            # Recreate
            container_urls = self._container_manager.create_containers(self.num_containers)
            await self._pool.initialize(num_connections=self.num_connections)
            self._pool.create_connections(container_urls, self.num_connections)

            if self._pool.clients:
                self.client = self._pool.clients[0]

            logger.info(f"Restart complete: {len(self._pool.clients)} sync connections")
            return {
                "success": True,
                "num_containers": len(container_urls),
                "num_connections": len(self._pool.clients),
            }

        except Exception as e:
            logger.error(f"Restart failed: {e}")
            return {"success": False, "error": str(e)}

    def create_action(self, **kwargs) -> "GenericAction":
        """Create a GenericAction instance."""
        from openenv import GenericAction
        return GenericAction(**kwargs)

    # Expose internal state for backward compatibility
    @property
    def clients(self) -> list:
        return self._pool.clients

    @property
    def client_available(self) -> list:
        return self._pool.client_available

    @property
    def container_urls(self) -> list:
        return self._container_manager.container_urls

    @property
    def providers(self) -> list:
        return self._container_manager.providers
