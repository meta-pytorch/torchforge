# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Monarch distributed executor for vLLM."""

from __future__ import annotations

import base64
import logging
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import cloudpickle
from monarch.actor import Actor, context, endpoint
from monarch.tools.network import get_ipaddr
from vllm.v1.executor.abstract import Executor
from vllm.v1.worker.worker_base import WorkerWrapperBase

logger = logging.getLogger(__name__)


def _get_host_ip() -> str:
    """Get this host's routable IP address using hostname resolution.

    Uses socket.gethostname() + DNS resolution, which works on internal
    networks where external IPs (like 8.8.8.8) are unreachable.
    """
    import socket

    if host_ip := os.environ.get("VLLM_HOST_IP"):
        return host_ip

    hostname = socket.gethostname()
    # Use Monarch's get_ipaddr which resolves hostname via DNS
    return get_ipaddr(hostname, 0)


def _get_free_port() -> int:
    """Get an available TCP port from the OS.

    Returns:
        int: Available port number
    """
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port


def _build_worker_configs(
    vllm_config,
    gpus_per_host: int,
    master_addr: str,
    master_port: int,
    gpu_ids: List[str] | None = None,
) -> tuple[list[dict[str, str]], list[dict]]:
    """Build environment variables and init kwargs for all workers.

    Args:
        vllm_config: vLLM configuration
        gpus_per_host: Number of GPUs per host
        master_addr: IP address of head node for distributed init
        master_port: Port for distributed init
        gpu_ids: List of allocated GPU IDs. If None, uses sequential indices.

    Returns:
        Tuple of (all_envs, all_kwargs) where:
        - all_envs: List of env var dicts, one per worker
        - all_kwargs: List of init kwargs dicts, one per worker
    """
    # Use allocated GPU IDs if provided, else fallback to sequential
    if gpu_ids:
        cuda_devices = ",".join(gpu_ids)
    else:
        cuda_devices = ",".join(str(i) for i in range(gpus_per_host))
    tp_size = vllm_config.parallel_config.tensor_parallel_size
    world_size = vllm_config.parallel_config.world_size

    all_envs = []
    all_kwargs = []
    for rank in range(world_size):
        local_rank = rank % gpus_per_host
        is_driver = rank % tp_size == 0  # First worker in each TP group

        # Environment variables for PyTorch distributed
        env_vars = {
            "MASTER_ADDR": master_addr,
            "MASTER_PORT": str(master_port),
            "WORLD_SIZE": str(world_size),
            "RANK": str(rank),
            "LOCAL_RANK": str(local_rank),
            "CUDA_VISIBLE_DEVICES": cuda_devices,
            "VLLM_HOST_IP": master_addr,
            # Disable symmetric memory for all-reduce - requires fd passing
            # between processes which doesn't work with Monarch
            "VLLM_ALLREDUCE_USE_SYMM_MEM": "0",
            # Disable ADDR2LINE to prevent fork crash
            "TORCH_DISABLE_ADDR2LINE": "1",
        }
        all_envs.append(env_vars)

        # Worker init kwargs
        worker_kwargs = {
            "vllm_config": vllm_config,
            "local_rank": local_rank,
            "rank": rank,
            "distributed_init_method": "env://" if world_size > 1 else None,
            "is_driver_worker": is_driver,
            "shared_worker_lock": None,  # shm cache not supported in distributed
        }
        all_kwargs.append(worker_kwargs)

    return all_envs, all_kwargs


class WorkerRegistry(Actor):
    """Rendezvous point for cross-process worker registration.

    Why this exists:
        AsyncLLM spawns EngineCore as a subprocess. MonarchExecutor runs inside
        this subprocess and cannot directly return the workers ActorMesh to Generator.
        This registry bridges the process boundary—MonarchExecutor registers workers
        here, Generator queries it after initialization.

    Why `call` instead of `broadcast`:
        MonarchExecutor uses `register_workers.call_one().get()` (blocking) rather than
        `broadcast` (fire-and-forget) to avoid a race condition. Broadcast doesn't
        guarantee message delivery before returning, so Generator might query
        `get_workers` before the registration completes. The blocking `call` ensures
        workers are registered before MonarchExecutor initialization finishes.

    Spawned by Generator on CPU proc (same host as Generator).
    """

    def __init__(self):
        self._workers = None

    @endpoint
    def register_workers(self, workers_mesh) -> None:
        """Register workers mesh reference."""
        self._workers = workers_mesh
        logger.info(f"[WorkerRegistry] Workers registered: {workers_mesh}")

    @endpoint
    def get_workers(self):
        """Get registered workers mesh reference."""
        return self._workers


class _FutureWrapper:
    """Wrapper to make Monarch Future compatible with vLLM's Future interface.

    vLLM expects a concurrent.futures.Future-like object with result() method
    and subscripting support (e.g., future[0]).
    """

    def __init__(self, monarch_future, timeout):
        self._future = monarch_future
        self._timeout = timeout
        self._result = None

    def result(self, timeout=None):
        """Get results, matching concurrent.futures.Future interface."""
        if self._result is None:
            use_timeout = timeout if timeout is not None else self._timeout
            try:
                result = self._future.get(timeout=use_timeout)
            except TimeoutError as e:
                raise TimeoutError(
                    f"Monarch RPC timed out after {use_timeout}s. "
                    "Workers may be unresponsive or overloaded."
                ) from e
            except Exception as e:
                raise RuntimeError(
                    f"Monarch RPC failed: {e}. " "Check worker connectivity and logs."
                ) from e
            # Extract values from ValueMesh
            outputs = []
            for rank_point, value in result.items():
                outputs.append(value)
            self._result = outputs
        # TODO: this is not compatible with PP
        return self._result[0] if self._result else None

    def __getitem__(self, index):
        """Support subscripting like output[0].

        vLLM expects output[0] to still return a Future-like object with .result()
        For index 0, return self to allow chaining.
        """
        if index == 0:
            return self
        raise IndexError(f"FutureWrapper only supports index 0, got {index}")


class WorkerWrapper(WorkerWrapperBase, Actor):
    """Monarch actor wrapper around vLLM WorkerWrapperBase.

    This class extends both WorkerWrapperBase (for vLLM worker functionality)
    and Actor (for Monarch RPC). It inherits all vLLM worker lifecycle methods
    (init_worker, init_device, load_model, execute_model) and exposes them as
    Monarch endpoints via dynamic dispatch through `execute_method`.

    Subclass this to add custom functionality (e.g., weight sync from external
    stores).
    """

    def __init__(self, vllm_config):
        rank = context().actor_instance.rank.rank
        # rpc_rank: rank within this executor (0 to num_workers-1)
        # global_rank: rank in distributed group (same as rpc_rank for single executor)
        WorkerWrapperBase.__init__(self, vllm_config, rpc_rank=rank, global_rank=rank)
        Actor.__init__(self)

    def init_worker(self, all_kwargs):
        """Initialize worker with global rank consistency check.

        Verifies that Monarch's actor ordering matches our expected loop ordering
        before delegating to the parent init_worker.
        """
        monarch_global_rank = self.rpc_rank
        expected_global_rank = all_kwargs[monarch_global_rank].get("rank")
        assert monarch_global_rank == expected_global_rank, (
            f"Global rank mismatch: Monarch assigned rank {monarch_global_rank}, "
            f"but all_kwargs[{monarch_global_rank}] has rank={expected_global_rank}. "
            "This indicates Monarch's actor ordering differs from our loop order."
        )
        super().init_worker(all_kwargs)

    @endpoint
    def execute_method(self, method: str, *args, **kwargs):
        # For simplicity, we only support string method names for now
        fn = getattr(self, method)
        return fn(*args, **kwargs)

    @endpoint
    def destroy_process_group(self) -> None:
        """Destroy PyTorch distributed process group to allow clean shutdown.

        Must be called on ALL workers before stopping procs to avoid
        NCCL heartbeat monitor errors when TCPStore dies.
        """
        import torch.distributed as dist

        if dist.is_initialized():
            logger.info("[WorkerWrapper] Destroying process group")
            dist.destroy_process_group()
            logger.info("[WorkerWrapper] Process group destroyed")


class MonarchExecutor(Executor):
    """Distributed executor that runs vLLM workers on Monarch's actor system.

    Receives a serialized HostMesh via environment variable, creates ProcMesh,
    spawns GPU workers, and registers them with WorkerRegistry for the caller
    to query.

    Architecture::

        ┌───────────────────────────────────────────────────────────────────────┐
        │                              Host Mesh                                │
        │                                                                       │
        │  ┌─────────────────────────────────────────────────────────────────┐  │
        │  │ Caller process                                                  │  │
        │  │                                                                 │  │
        │  │  ┌─────────────────────┐       ┌─────────────────────────────┐  │  │
        │  │  │ AsyncLLM            │       │ WorkerRegistry (actor)      │  │  │
        │  │  └─────────────────────┘       └─────────────────────────────┘  │  │
        │  │            │                                                    │  │
        │  │            │ serialize host_mesh & registry to env vars         │  │
        │  │            ▼                                                    │  │
        │  │  ┌───────────────────────────────────────────────────────────┐  │  │
        │  │  │ EngineCore subprocess                                     │  │  │
        │  │  │                                                           │  │  │
        │  │  │ MonarchExecutor                                           │  │  │
        │  │  │   ├── deserialize host_mesh                               │  │  │
        │  │  │   ├── create proc_mesh from host_mesh (owns lifecycle) ───│──│──│──┐
        │  │  │   ├── spawn worker actors on proc_mesh                    │  │  │  │
        │  │  │   └── register workers in WorkerRegistry                  │  │  │  │
        │  │  └───────────────────────────────────────────────────────────┘  │  │  │
        │  └─────────────────────────────────────────────────────────────────┘  │  │
        │                                                                       │  │
        │  ┌─────────────────────────────────────────────────────────────────┐  │  │
        │  │ GPU ProcMesh (owned by MonarchExecutor)                         │  │  │
        │  │                                                                 │  │  │
        │  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐         │  │  │
        │  │  │ Worker 0 │  │ Worker 1 │  │ Worker 2 │  │ Worker 3 │  ... ◀──│──│──┘
        │  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘         │  │
        │  │                   ◀──── NCCL (tensor parallel) ────▶            │  │
        │  └─────────────────────────────────────────────────────────────────┘  │
        └───────────────────────────────────────────────────────────────────────┘

    Design:
        Caller owns host_mesh (resource allocation), executor owns proc_mesh +
        workers (execution). This mirrors vLLM's Ray executor pattern.

        Alternative: Caller spawns workers directly and passes reference to executor,
        eliminating WorkerRegistry. We chose executor-owns-workers for vLLM alignment.

    Usage:
        Set environment variables before creating AsyncLLM:
        - VLLM_MONARCH_HOST_MESH: base64-encoded cloudpickle of HostMesh
        - VLLM_MONARCH_WORKER_REGISTRY: base64-encoded cloudpickle of WorkerRegistry

    Limitations:
        - TP supported, PP not supported (would require DAG execution)
        - Shared memory cache (mm_processor_cache_type='shm') not supported
        - Symmetric memory all-reduce disabled

    Attributes:
        worker_class: Worker class to spawn. Override in subclasses for custom workers.
    """

    uses_ray: bool = False  # defined on the base `Executor` class
    supports_pp: bool = False  # defined on the base `Executor` class

    # Worker class to spawn. Override in subclasses for custom workers.
    worker_class = WorkerWrapper

    def _init_executor(self) -> None:
        """Initialize by deserializing HostMesh and creating ProcMesh with workers."""
        # Enable TCP transport for multi-node communication
        # This MUST be called before any Monarch API calls that span multiple hosts
        # Default is Unix sockets which only work locally
        from monarch.actor import enable_transport

        enable_transport("tcp")

        host_mesh_str = os.environ.get("VLLM_MONARCH_HOST_MESH")
        if not host_mesh_str:
            raise RuntimeError(
                "VLLM_MONARCH_HOST_MESH not set. Generator must set this."
            )

        registry_str = os.environ.get("VLLM_MONARCH_WORKER_REGISTRY")
        if not registry_str:
            raise RuntimeError(
                "VLLM_MONARCH_WORKER_REGISTRY not set. Generator must set this."
            )

        world_size = self.vllm_config.parallel_config.world_size

        logger.info("[MonarchExecutor] Deserializing HostMesh...")
        self.host_mesh = cloudpickle.loads(base64.b64decode(host_mesh_str))
        logger.info(f"[MonarchExecutor] HostMesh: {self.host_mesh}")

        logger.info("[MonarchExecutor] Deserializing WorkerRegistry...")
        self.worker_registry = cloudpickle.loads(base64.b64decode(registry_str))
        logger.info(f"[MonarchExecutor] WorkerRegistry: {self.worker_registry}")

        # Derive gpus_per_host from host_mesh extent and vllm_config
        try:
            num_hosts = self.host_mesh.extent["hosts"]
        except (KeyError, AttributeError, TypeError, ValueError):
            num_hosts = 1  # Local host (this_host()) has no extent
        gpus_per_host = world_size // num_hosts

        # Check for unsupported multi-node PP configuration
        pp_size = self.vllm_config.parallel_config.pipeline_parallel_size
        if pp_size > 1 and num_hosts > 1:
            raise NotImplementedError(
                f"Multi-node Pipeline Parallelism (PP={pp_size}, hosts={num_hosts}) "
                "is not supported by MonarchExecutor yet. "
            )

        logger.info(
            f"[MonarchExecutor] Creating ProcMesh with {gpus_per_host} procs per host "
            f"(world_size={world_size}, num_hosts={num_hosts})..."
        )
        self.proc_mesh = self.host_mesh.spawn_procs(
            per_host={"procs": gpus_per_host}, name="vllm_workers"
        )
        self.workers = self.proc_mesh.spawn(
            "vllm_workers", self.worker_class, self.vllm_config
        )

        # Build worker configs and initialize
        head_node_ip = _get_host_ip()
        master_port = _get_free_port()
        logger.info(f"[MonarchExecutor] Head node: {head_node_ip}:{master_port}")

        # Read allocated GPU IDs from environment (set by Generator.launch)
        gpu_ids_str = os.environ.get("VLLM_MONARCH_GPU_IDS")
        gpu_ids = gpu_ids_str.split(",") if gpu_ids_str else None

        if gpu_ids:
            logger.info(f"[MonarchExecutor] Using allocated GPUs: {gpu_ids}")
        else:
            logger.warning(
                "[MonarchExecutor] No VLLM_MONARCH_GPU_IDS set, "
                "using sequential indices (may cause GPU contention)"
            )

        all_envs, all_kwargs = _build_worker_configs(
            self.vllm_config, gpus_per_host, head_node_ip, master_port, gpu_ids
        )

        self.collective_rpc("update_environment_variables", args=(all_envs,))
        self.collective_rpc("init_worker", args=(all_kwargs,))
        self.collective_rpc("init_device")
        self.collective_rpc("load_model")

        logger.info(f"[MonarchExecutor] Initialized {world_size} workers")

        self.worker_registry.register_workers.call_one(self.workers).get()
        logger.info("[MonarchExecutor] Workers registered")

    def collective_rpc(
        self,
        method: Union[str, Callable],
        timeout: Optional[float] = None,
        args: Tuple = (),
        kwargs: Optional[Dict[str, Any]] = None,
        non_block: bool = False,
    ) -> List[Any]:
        """Execute method on all workers via Monarch RPC."""
        future = self.workers.execute_method.call(method, *args, **(kwargs or {}))

        if non_block:
            return _FutureWrapper(future, timeout)

        result = future.get(timeout=timeout)
        return [value for _, value in result.items()]

    def check_health(self):
        """Health check for workers (not yet implemented)."""
        return

    def shutdown(self):
        """Shutdown workers and stop ProcMesh.

        Since MonarchExecutor owns the proc_mesh and workers (created them from
        host_mesh), it is responsible for cleaning them up during shutdown.

        Cleanup order:
        1. Destroy PyTorch process groups on workers (prevents NCCL errors)
        2. Call base class shutdown
        3. Stop proc_mesh
        """
        logger.info("[MonarchExecutor] Shutting down...")

        # Destroy process groups on all workers before stopping proc_mesh
        # This prevents NCCL heartbeat monitor errors when TCPStore dies
        try:
            if hasattr(self, "workers") and self.workers is not None:
                self.workers.destroy_process_group.call().get()
                logger.info("[MonarchExecutor] Destroyed process groups on workers")
        except Exception as e:
            logger.warning(f"[MonarchExecutor] Error destroying process groups: {e}")

        try:
            if hasattr(self, "workers"):
                super().shutdown()
        except Exception as e:
            logger.warning(f"[MonarchExecutor] Error during worker shutdown: {e}")

        try:
            if hasattr(self, "proc_mesh") and self.proc_mesh is not None:
                self.proc_mesh.stop().get()
                logger.info("[MonarchExecutor] Stopped proc_mesh")
        except Exception as e:
            logger.warning(f"[MonarchExecutor] Error stopping proc_mesh: {e}")

        logger.info("[MonarchExecutor] Shutdown complete")
