# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Shared utilities for OpenEnv actors.

This module contains common utility functions used by OpenEnvActor.
"""

import logging
import socket

logger = logging.getLogger(__name__)


# Error keywords for connection issue detection
WEBSOCKET_ERROR_KEYWORDS = [
    "connectionclosederror",
    "keepalive ping timeout",
    "websocket",
    "connection closed",
    "connection reset",
    "broken pipe",
]

CONTAINER_ERROR_KEYWORDS = [
    "no such container",
    "container not found",
    "container is not running",
    "container has stopped",
    "container exited",
    "exec session",
    "state improper",
    "oci runtime error",
    "docker daemon",
    "cannot connect to docker",
    "connection refused",
]

HTTP_ERROR_KEYWORDS = [
    "connection timeout",
    "read timeout",
    "http error",
    "status code",
]


def is_connection_error(error_msg: str) -> tuple:
    """Check if error is a connection-related error.

    Args:
        error_msg: The error message to check.

    Returns:
        Tuple of (is_error, error_type) where error_type is 'websocket', 'container', or None.
    """
    error_lower = error_msg.lower()
    if any(kw in error_lower for kw in WEBSOCKET_ERROR_KEYWORDS):
        return True, "websocket"
    if any(kw in error_lower for kw in CONTAINER_ERROR_KEYWORDS):
        return True, "container"
    return False, None


def is_http_error(error_msg: str) -> bool:
    """Check if error is an HTTP-level error (not requiring container recreation)."""
    error_lower = error_msg.lower()
    return any(kw in error_lower for kw in HTTP_ERROR_KEYWORDS)


def is_port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    """
    Check if a port is already in use on the specified host.

    Args:
        port: Port number to check
        host: Host address to check (default: 127.0.0.1)

    Returns:
        True if port is in use, False if available
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, port))
            return False
        except OSError:
            return True


def find_available_port(
    preferred_port: int, min_port: int = 5000, max_attempts: int = 100
) -> int:
    """
    Find an available port starting from preferred_port and decrementing.

    Args:
        preferred_port: The preferred port to use
        min_port: Minimum port number to try (default: 5000)
        max_attempts: Maximum number of ports to try (default: 100)

    Returns:
        An available port number

    Raises:
        RuntimeError: If no available port is found after max_attempts
    """
    port = preferred_port
    attempts = 0

    while attempts < max_attempts:
        if port < min_port:
            raise RuntimeError(
                f"No available port found after trying {attempts} ports. "
                f"Reached minimum port {min_port}."
            )

        if not is_port_in_use(port):
            logger.info(f"Found available port: {port}")
            return port

        logger.debug(f"Port {port} is in use, trying {port - 1}")
        port -= 1
        attempts += 1

    raise RuntimeError(
        f"No available port found after trying {max_attempts} ports "
        f"(from {preferred_port} down to {port})."
    )


class ContainerConfig:
    """Configuration for container setup."""

    def __init__(
        self,
        docker_image: str,
        env_name: str = "openenv",
        env_vars: dict = None,
        port: int = 8000,
        memory_gb: int = 4,
        timeout_s: float = 180.0,
    ):
        self.docker_image = docker_image
        self.env_name = env_name
        self.env_vars = env_vars or {}
        self.port = port
        self.memory_gb = memory_gb
        self.timeout_s = timeout_s


class ContainerManager:
    """Manages Docker container lifecycle for OpenEnv environments."""

    def __init__(self, config: ContainerConfig):
        self.config = config
        self.providers = []
        self.container_urls = []
        self._logs_dir = None

    def _setup_logs_dir(self):
        """Create the logs directory for container output."""
        import os
        self._logs_dir = os.path.expanduser(f"~/{self.config.env_name}_container_logs")
        os.makedirs(self._logs_dir, exist_ok=True)
        return self._logs_dir

    def _build_container_env_vars(self, container_port: int) -> dict:
        """Build environment variables for a container."""
        env_vars = self.config.env_vars.copy()
        env_vars["PORT"] = str(container_port)

        env_name_upper = self.config.env_name.upper()
        log_filename = f"{self.config.env_name}_env_port_{container_port}.log"
        container_log_path = f"/tmp/{self.config.env_name}_logs/{log_filename}"
        env_vars[f"{env_name_upper}_LOG_FILE"] = container_log_path
        env_vars[f"{env_name_upper}_LOG_LEVEL"] = "DEBUG"

        return env_vars

    def _build_volumes(self) -> dict:
        """Build volume mappings for containers."""
        if not self._logs_dir:
            self._setup_logs_dir()
        return {self._logs_dir: f"/tmp/{self.config.env_name}_logs"}

    def create_containers(self, num_containers: int):
        """Create and start Docker containers.

        Args:
            num_containers: Number of containers to create.

        Returns:
            List of container base URLs.
        """
        from openenv.core.containers.runtime import LocalDockerProvider

        self._setup_logs_dir()
        self.providers = []
        self.container_urls = []

        for i in range(num_containers):
            try:
                container_port = find_available_port(self.config.port - i)
                logger.info(f"Creating container {i + 1}/{num_containers} on port {container_port}")

                env_vars = self._build_container_env_vars(container_port)
                volumes = self._build_volumes()

                provider = LocalDockerProvider()
                base_url = provider.start_container(
                    self.config.docker_image,
                    port=container_port,
                    env_vars=env_vars,
                    volumes=volumes,
                    memory_gb=self.config.memory_gb,
                )

                provider.wait_for_ready(base_url, timeout_s=self.config.timeout_s)

                self.providers.append(provider)
                self.container_urls.append(base_url)

                logger.info(f"Container {i + 1}/{num_containers} ready at {base_url}")

            except Exception as e:
                logger.error(f"Failed to create container {i + 1}: {e}")
                self.stop_all()
                raise

        return self.container_urls

    def stop_all(self):
        """Stop all managed containers."""
        import subprocess

        for i, provider in enumerate(self.providers):
            try:
                if hasattr(provider, 'container_id') and provider.container_id:
                    # Use docker kill directly for faster shutdown
                    # docker stop can hang on stuck processes
                    try:
                        subprocess.run(
                            ['docker', 'kill', provider.container_id],
                            timeout=5,
                            capture_output=True
                        )
                    except subprocess.TimeoutExpired:
                        logger.warning(f"docker kill timed out for container {i}")
                else:
                    provider.stop_container()
                logger.debug(f"Stopped container {i}")
            except Exception as e:
                logger.warning(f"Error stopping container {i}: {e}")

        self.providers = []
        self.container_urls = []


class ConnectionPool:
    """Manages a pool of sync WebSocket connections to OpenEnv containers.

    Uses thread pool execution to avoid blocking the asyncio event loop
    while maintaining simple sync WebSocket clients.
    """

    def __init__(self, request_timeout_s: float = 120.0, max_workers: int | None = None):
        self.request_timeout_s = request_timeout_s
        self._max_workers = max_workers  # If None, will be derived from num_connections
        self.clients = []
        self.client_available = []
        self._lock = None
        self._condition = None
        self._executor = None

    async def initialize(self, num_connections: int = 16):
        """Initialize async primitives and thread pool.

        Args:
            num_connections: Number of connections that will be created.
                Used to derive thread pool size if max_workers not set.
        """
        import asyncio
        from concurrent.futures import ThreadPoolExecutor

        self._lock = asyncio.Lock()
        self._condition = asyncio.Condition(self._lock)

        # Derive thread pool size: 2 threads per connection, capped at 64
        # This allows for concurrent execute + health check per connection
        max_workers = self._max_workers or min(64, max(4, num_connections * 2))
        logger.info(f"ConnectionPool: Creating thread pool with {max_workers} workers for {num_connections} connections")
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="ws_pool")

    def create_connections(self, container_urls: list, num_connections: int):
        """Create sync WebSocket connections distributed across containers.

        Args:
            container_urls: List of container base URLs.
            num_connections: Total number of connections to create.
        """
        from openenv import GenericEnvClient

        self.clients = []
        self.client_available = []
        num_containers = len(container_urls)

        for i in range(num_connections):
            try:
                container_idx = i % num_containers
                base_url = container_urls[container_idx]

                logger.debug(f"Creating sync connection {i + 1}/{num_connections} → container {container_idx}")

                client = GenericEnvClient(
                    base_url=base_url,
                    connect_timeout_s=10.0,
                    message_timeout_s=self.request_timeout_s,
                )
                client.reset()

                self.clients.append(client)
                self.client_available.append(True)

            except Exception as e:
                logger.error(f"Failed to create connection {i + 1}: {e}")
                self.close_all_sync()
                raise

        logger.info(f"Connection pool ready: {len(self.clients)} sync connections")

    async def acquire(self, timeout: float = 30.0) -> tuple:
        """Acquire an available client from the pool.

        Args:
            timeout: Maximum wait time in seconds.

        Returns:
            Tuple of (client_index, client).

        Raises:
            TimeoutError: If no client available within timeout.
        """
        import asyncio

        start_time = asyncio.get_event_loop().time()

        async with self._condition:
            while True:
                for i, available in enumerate(self.client_available):
                    if available:
                        self.client_available[i] = False
                        logger.debug(f"Acquired client {i} from pool")
                        return i, self.clients[i]

                elapsed = asyncio.get_event_loop().time() - start_time
                remaining = timeout - elapsed
                if remaining <= 0:
                    raise TimeoutError(
                        f"No client available after {timeout}s. "
                        f"All {len(self.clients)} clients busy."
                    )

                try:
                    await asyncio.wait_for(self._condition.wait(), timeout=remaining)
                except asyncio.TimeoutError as timeout_err:
                    raise TimeoutError(
                        f"No client available after {timeout}s. "
                        f"All {len(self.clients)} clients busy."
                    ) from timeout_err

    async def release(self, client_idx: int):
        """Release a client back to the pool."""
        async with self._condition:
            self.client_available[client_idx] = True
            logger.debug(f"Released client {client_idx}")
            self._condition.notify()

    async def reconnect(self, client_idx: int, container_urls: list) -> "GenericEnvClient":
        """Reconnect a failed client.

        Args:
            client_idx: Index of client to reconnect.
            container_urls: List of container URLs.

        Returns:
            New client instance.
        """
        from openenv import GenericEnvClient
        import asyncio

        num_containers = len(container_urls)
        container_idx = client_idx % num_containers
        base_url = container_urls[container_idx]

        logger.info(f"Reconnecting sync client {client_idx} to {base_url}")

        # Close old client in thread pool
        old_client = self.clients[client_idx]
        if old_client:
            try:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(self._executor, old_client.close)
            except Exception as e:
                logger.debug(f"Error closing old client: {e}")

        await asyncio.sleep(1)

        # Create new client in thread pool
        def create_client():
            client = GenericEnvClient(
                base_url=base_url,
                connect_timeout_s=10.0,
                message_timeout_s=self.request_timeout_s,
            )
            client.reset()
            return client

        loop = asyncio.get_event_loop()
        new_client = await loop.run_in_executor(self._executor, create_client)

        self.clients[client_idx] = new_client
        logger.info(f"Sync client {client_idx} reconnected")
        return new_client

    async def execute_step(self, client_idx: int, action: dict):
        """Execute step on client using thread pool to avoid blocking event loop.

        Args:
            client_idx: Index of client to use.
            action: Action dictionary to execute.

        Returns:
            StepResult from the client.
        """
        import asyncio

        client = self.clients[client_idx]
        loop = asyncio.get_event_loop()

        # Run sync WebSocket call in thread pool - doesn't block event loop
        return await loop.run_in_executor(
            self._executor,
            client.step,
            action
        )

    async def close_all(self):
        """Close all connections and shutdown thread pool."""
        self.close_all_sync()
        if self._executor:
            self._executor.shutdown(wait=False)
            self._executor = None

    def close_all_sync(self):
        """Close all connections synchronously."""
        for i, client in enumerate(self.clients):
            try:
                client.close()
                logger.debug(f"Closed sync client {i}")
            except Exception as e:
                logger.warning(f"Error closing client {i}: {e}")

        self.clients = []
        self.client_available = []

    def get_status(self) -> dict:
        """Get pool status."""
        return {
            "total": len(self.clients),
            "available": sum(1 for a in self.client_available if a),
            "busy": sum(1 for a in self.client_available if not a),
        }
