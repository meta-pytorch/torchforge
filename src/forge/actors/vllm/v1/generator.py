# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import base64
import logging
import os
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Optional

import cloudpickle
import torch
from forge.controller import ForgeActor
from forge.controller.provisioner import _get_provisioner
from forge.data_models.completion import Completion
from forge.data_models.prompt import to_prompt
from monarch.actor import endpoint, this_host
from torchstore.api import _controller as get_torchstore_controller
from vllm.engine.arg_utils import EngineArgs
from vllm.entrypoints.llm import UsageContext
from vllm.outputs import RequestOutput
from vllm.sampling_params import RequestOutputKind, SamplingParams
from vllm.v1.engine.async_llm import AsyncLLM

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Suppress noisy vLLM "Added request" logs
logging.getLogger("vllm.v1.engine.async_llm").setLevel(logging.WARNING)


@dataclass
class Generator(ForgeActor):
    """vLLM-based generator using AsyncLLM with Monarch distributed execution.

    Wraps vLLM's AsyncLLM engine and uses MonarchExecutor for multi-GPU inference.
    See MonarchExecutor docstring for architecture diagram.

    Args:
        engine_args: vLLM EngineArgs for model configuration. Can be EngineArgs or dict.
        sampling_params: Default SamplingParams for generation. Can be SamplingParams or dict.

    Example:
        >>> generator = await Generator.options(procs=1, with_gpus=True).as_service(
        ...     engine_args={"model": "meta-llama/Llama-3-8B", "tensor_parallel_size": 2},
        ...     sampling_params={"max_tokens": 128, "temperature": 0.7},
        ... )
        >>> completions = await generator.generate("Tell me a joke")
        >>> await generator.shutdown()
    """

    engine_args: EngineArgs | Mapping = field(default_factory=EngineArgs)
    sampling_params: SamplingParams | Mapping = field(default_factory=SamplingParams)

    def __post_init__(self):
        super().__init__()
        self.llm: Optional[AsyncLLM] = None
        self.generator_version: int = 0
        self.workers: Any = None  # Workers ActorMesh, registered by MonarchExecutor

        if isinstance(self.engine_args, Mapping):
            self.engine_args = EngineArgs(**self.engine_args)
        self.vllm_config = self.engine_args.create_engine_config(UsageContext.LLM_CLASS)

        if isinstance(self.sampling_params, Mapping):
            self.sampling_params = SamplingParams.from_optional(**self.sampling_params)
            self.sampling_params.output_kind = RequestOutputKind.FINAL_ONLY

    @classmethod
    async def launch(  # pyright: ignore[reportIncompatibleMethodOverride]
        cls: type["Generator"],
        *args,
        **kwargs,
    ) -> "Generator":
        """Custom launch for Generator with proper GPU host provisioning.

        Flow:
        1. Get host mesh directly (via get_host_mesh for remote, this_host for local)
        2. Spawn CPU proc on head host for Generator and WorkerRegistry
        3. Allocate GPUs from provisioner
        4. Pass host_mesh and GPU IDs to setup() - executor creates proc_mesh
        """
        engine_args = kwargs.get("engine_args", {})
        if isinstance(engine_args, Mapping):
            engine_args = EngineArgs(**engine_args)
        vllm_config = engine_args.create_engine_config(UsageContext.LLM_CLASS)

        num_gpus = vllm_config.parallel_config.world_size
        num_hosts = cls.hosts if cls.hosts else None
        gpus_per_host = num_gpus // (num_hosts or 1)
        mesh_name = cls.mesh_name or "generator"

        # Step 1: Get host mesh directly (no bootstrap procs)
        provisioner = await _get_provisioner()
        if provisioner is None:
            raise RuntimeError(
                "Provisioner not initialized. Call init_provisioner() first."
            )

        if num_hosts:
            # Remote allocation - get pre-allocated mesh from launcher
            host_mesh = await provisioner.get_host_mesh(mesh_name)
        else:
            # Local allocation
            host_mesh = this_host()

        # Step 1a: Mount wsfuse on all hosts for local model paths
        # This must happen before any actors are spawned that might access the path
        if num_hosts and provisioner.launcher:
            # Spawn temporary procs on all hosts just for mounting
            mount_procs = host_mesh.spawn_procs(
                per_host={"procs": 1}, name="mount_setup"
            )
            await provisioner.launcher.remote_setup(mount_procs)
            logger.info("Completed remote_setup (mounted wsfuse on all hosts)")

        # Step 1b: Allocate GPUs from provisioner
        # This ensures each Generator gets exclusive GPU IDs
        gpu_ids = await provisioner.allocate_gpu_ids(host_mesh, num_gpus)
        logger.info(f"[Generator.launch] Allocated GPUs: {gpu_ids}")

        logger.info(
            f"[Generator.launch] Provisioned host mesh with {num_hosts or 1} host(s), "
            f"{gpus_per_host} GPUs per host"
        )

        # Step 2: Spawn CPU proc on head host for Generator and WorkerRegistry
        if num_hosts:
            singleton_slice = {k: slice(0, 1) for k in host_mesh.extent.keys()}
            head_host = host_mesh.slice(**singleton_slice)
        else:
            head_host = host_mesh

        generator_proc = head_host.spawn_procs(
            per_host={"procs": 1}, name="generator_proc"
        )
        logger.info("[Generator.launch] Spawned generator_proc on head host")

        # Import WorkerRegistry here to avoid circular import with monarch_executor
        from forge.actors.vllm.v1.monarch_executor import WorkerRegistry

        # Spawn WorkerRegistry on CPU proc (same as Generator)
        worker_registry = generator_proc.spawn("worker_registry", WorkerRegistry)
        logger.info("[Generator.launch] Spawned WorkerRegistry on generator_proc")

        actor_name = kwargs.pop("name", cls.__name__)
        generator = generator_proc.spawn(
            actor_name,
            cls,
            *args,
            **kwargs,
        )

        # Attach for cleanup in Generator.shutdown()
        generator._generator_proc = generator_proc
        generator._worker_registry = worker_registry

        # Step 3: Pass host_mesh and gpu_ids to setup() - executor will create proc_mesh
        await generator.setup.call(host_mesh, worker_registry, gpu_ids)
        return generator

    @endpoint
    async def setup(self, host_mesh, worker_registry, gpu_ids: list[str]):
        """Initialize AsyncLLM with MonarchExecutor.

        Receives a host_mesh from launch(). Serializes it for the EngineCore
        subprocess. MonarchExecutor creates its own proc_mesh from host_mesh,
        spawns workers, and registers them. After AsyncLLM initialization,
        Generator queries the registry for workers.

        Args:
            host_mesh: HostMesh for GPU workers (executor will create proc_mesh from this)
            worker_registry: WorkerRegistry ActorMesh for worker registration
            gpu_ids: List of allocated GPU IDs (e.g., ["0", "1"])
        """
        num_gpus = self.vllm_config.parallel_config.tensor_parallel_size

        logger.info(f"Setting up AsyncLLM with {num_gpus} GPUs, allocated: {gpu_ids}")

        # Set env var for MonarchExecutor subprocess (EngineCore)
        os.environ["VLLM_MONARCH_GPU_IDS"] = ",".join(gpu_ids)

        # Serialize host_mesh reference
        serialized_host_mesh = base64.b64encode(cloudpickle.dumps(host_mesh)).decode(
            "utf-8"
        )
        os.environ["VLLM_MONARCH_HOST_MESH"] = serialized_host_mesh

        # Serialize WorkerRegistry reference
        serialized_registry = base64.b64encode(
            cloudpickle.dumps(worker_registry)
        ).decode("utf-8")
        os.environ["VLLM_MONARCH_WORKER_REGISTRY"] = serialized_registry

        # Serialize TorchStore Controller reference for workers to access torchstore
        torchstore_controller = await get_torchstore_controller()
        serialized_controller = base64.b64encode(
            cloudpickle.dumps(torchstore_controller)
        ).decode("utf-8")
        os.environ["VLLM_TORCHSTORE_CONTROLLER"] = serialized_controller

        # Force 'spawn' multiprocessing method for Monarch actors.
        # This follows vLLM's Ray integration pattern.
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

        # Set the executor backend to ForgeMonarchExecutor via string path
        # This avoids import deadlock when vLLM spawns EngineCore subprocess
        self.vllm_config.parallel_config.distributed_executor_backend = (
            "forge.actors.vllm.v1.forge_executor.ForgeMonarchExecutor"
        )
        from vllm.v1.executor.abstract import Executor

        try:
            self.llm = AsyncLLM(
                vllm_config=self.vllm_config,
                # this resolve to the MonarchExecutor class
                executor_class=Executor.get_class(self.vllm_config),
                log_stats=True,
            )
            logger.info(f"AsyncLLM initialized with {num_gpus} workers")
        except Exception as e:
            logger.error(f"AsyncLLM initialization failed: {e}")
            raise

        # Query the WorkerRegistry for workers that were registered by MonarchExecutor
        # during _init_executor()
        self.workers = await worker_registry.get_workers.call_one()
        if self.workers is None:
            raise RuntimeError(
                "Workers not found in registry. "
                "MonarchExecutor may have failed to register workers."
            )
        logger.info(f"Retrieved workers from registry: {self.workers}")

    @endpoint
    async def generate(
        self,
        prompt: str,
        *,
        priority: int = 0,
        sampling_params: SamplingParams | None = None,
    ) -> list[Completion]:
        """Generate a response for the given prompt.

        Args:
            prompt (str): The prompt to generate a response for.
            priority (int, optional): The priority of the request. Defaults to 0.
            sampling_params (SamplingParams, optional): Sampling parameters to use for this request.
                If not provided, uses self.sampling_params.

        Returns:
            list[Completion]: n completions from vLLM based on your prompt.
        """
        if self.llm is None:
            raise RuntimeError("Generator not initialized. Call setup() first.")

        params = sampling_params or self.sampling_params

        if params.output_kind is None:
            params.output_kind = RequestOutputKind.FINAL_ONLY

        # Use AsyncLLM's generate() method - it returns an async generator
        # that yields RequestOutput objects. We only want the final output.
        request_output = None
        async for output in self.llm.generate(
            prompt=prompt,
            sampling_params=params,
            request_id=str(uuid.uuid4()),
        ):
            request_output = output  # Keep last output (final one)

        completions = self._to_completions(request_output, prompt)

        return completions

    @endpoint
    async def stop(self):
        """Stop the generator and cleanup local resources.

        This method is idempotent and can be called multiple times safely.
        Note: Remote worker cleanup happens in shutdown() which has access to the proxy.
        """
        if self.llm is not None:
            logger.info("Stopping AsyncLLM")
            self.llm.shutdown()
            logger.info("AsyncLLM.shutdown() returned")
            self.llm = None

        logger.info("stop() complete")

    @classmethod
    async def shutdown(cls, actor):
        """Shutdown the generator and cleanup all resources.

        Cleanup order:
        1. Stop AsyncLLM (triggers MonarchExecutor.shutdown() which destroys
           process groups and stops proc_mesh)
        2. Stop generator_proc
        """
        try:
            await actor.stop.call()
        except Exception as e:
            logger.warning(f"Error during actor.stop: {e}")

        try:
            if getattr(actor, "_generator_proc", None):
                await actor._generator_proc.stop()
                logger.info("Stopped generator proc")
        except Exception as e:
            logger.warning(f"Error during generator_proc stop: {e}")

        logger.info("shutdown() complete")

    @endpoint
    async def update_weights(
        self,
        version: Optional[int] = None,
    ) -> None:
        """Update weights on the generator from torchstore.

        This method:
        1. Pauses generation and waits for in-flight requests to complete
        2. Updates weights on workers from torchstore
        3. Resumes generation

        Note: This is NOT the standard vLLM weight update approach. vLLM typically
        uses `collective_rpc` on EngineClient, which internally routes calls to
        workers via the executor. However, `collective_rpc` uses msgspec/msgpack
        serialization which does not support arbitrary Python objects by default
        (only with VLLM_ALLOW_INSECURE_SERIALIZATION=1). This makes it difficult to
        pass complex objects like torchstore storage handles. Instead, we use a
        monarch-native approach where the Generator actor directly calls the worker
        mesh (`self.workers.update_weights`) via Monarch RPC, which uses cloudpickle
        and natively supports Monarch actor references for torchstore integration.

        Args:
            version: Policy version to load from torchstore
        """
        if self.llm is None:
            raise RuntimeError("Generator not initialized. Call setup() first.")

        logger.info(f"Starting weight update to v{version}")

        await self.llm.pause_generation(
            wait_for_inflight_requests=True, clear_cache=True
        )

        try:
            await self.workers.update_weights.call(version)
            self.generator_version = version
            logger.info(f"Updated weights from torchstore v{version}")
        finally:
            await self.llm.resume_generation()

        logger.info(f"Weight update complete, now v{version}")

    @endpoint
    async def save_model_params(self):
        """Save model parameters before weight update, used for testing purposes only."""
        logger.info("save model parameters for testing.")
        await self.workers.save_model_params.call()

    @endpoint
    async def validate_model_params(self, validate_fn):
        """Validate updated model params using validate_fn."""
        logger.info("start validating model parameters.")
        return await self.workers.validate_model_params.call(validate_fn)

    def _to_completions(
        self, request_output: RequestOutput, prompt: str
    ) -> list[Completion]:
        """Convert vLLM RequestOutput to forge Completion objects.

        Args:
            request_output: vLLM request output with completions.
            prompt: Original prompt string.

        Returns:
            List of Completion objects.
        """
        completions = []

        for output in request_output.outputs:
            completion = Completion(
                prompt=to_prompt(prompt),
                text=output.text,
                prompt_ids=torch.tensor(
                    request_output.prompt_token_ids
                    if request_output.prompt_token_ids
                    else []
                ),
                token_ids=torch.tensor(
                    output.token_ids if hasattr(output, "token_ids") else []
                ),
                logprobs=(output.logprobs if hasattr(output, "logprobs") else None),
                stop_reason=output.finish_reason,
                generator_version=self.generator_version,
                metadata=None,
            )
            completions.append(completion)

        return completions
