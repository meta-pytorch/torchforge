# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import os
import sys
import time
from copy import copy
from dataclasses import asdict, dataclass, field

import torch
from monarch.actor import current_rank, endpoint, ProcMesh
from torchstore import MultiProcessStore
from torchstore._state_dict_utils import DELIM, get_state_dict

from vllm.engine.arg_utils import EngineArgs
from vllm.entrypoints.utils import _validate_truncation_size
from vllm.executor.multiproc_worker_utils import set_multiprocessing_worker_envs
from vllm.lora.request import LoRARequest
from vllm.outputs import RequestOutput
from vllm.sampling_params import GuidedDecodingParams, RequestOutputKind, SamplingParams
from vllm.transformers_utils.tokenizer_group import init_tokenizer_from_configs
from vllm.usage.usage_lib import UsageContext
from vllm.utils import get_distributed_init_method
from vllm.v1.core.kv_cache_utils import get_kv_cache_config
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine import EngineCoreOutputs, EngineCoreRequest
from vllm.v1.engine.output_processor import OutputProcessor
from vllm.v1.engine.parallel_sampling import ParentRequest
from vllm.v1.engine.processor import Processor
from vllm.v1.request import Request
from vllm.v1.structured_output import StructuredOutputManager
from vllm.worker.worker_base import WorkerWrapperBase

from forge.controller import ForgeActor, get_proc_mesh, stop_proc_mesh

from forge.data.sharding import VLLMSharding
from forge.interfaces import Policy as PolicyInterface
from forge.types import ProcessConfig


@dataclass
class SamplingOverrides:
    """
    Overrides for vLLMs sampling params.

    Note: We'll want to tie this closer to or directly use vllm's
            SamplingParams. It is currently used to track a supported
            subset

    Args:
        n: Number of samples to generate.
        guided_decoding: Whether to use guided decoding.
        max_tokens: Maximum number of tokens to generate.
    """

    n: int
    guided_decoding: bool = False
    max_tokens: int = 512

    def __post_init__(self):
        gd_params = None
        if self.guided_decoding:
            gd_params = GuidedDecodingParams(choice=["Positive", "Negative"])
        self.guided_decoding = gd_params


@dataclass
class WorkerConfig:
    """
    Config args used for setting up the policy worker.

    Args:
        model: Model name.
        tensor_parallel_size: Number of tensor parallel workers.
        pipeline_parallel_size: Number of pipeline parallel workers.
        enforce_eager: Whether to enforce eager mode.
        vllm_args: vLLM engine args.
    """

    model: str
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    enforce_eager: bool = False
    vllm_args: EngineArgs = None


@dataclass
class PolicyConfig:
    worker_params: WorkerConfig
    sampling_params: SamplingOverrides
    available_devices: str = None


@dataclass
class Policy(PolicyInterface):
    config: PolicyConfig
    # Gets set up by setup
    sampling_params: SamplingParams | None = None
    lora_request: LoRARequest | None = None
    tokenization_kwargs: dict = field(default_factory=dict)
    policy_worker: "PolicyWorker" = None
    store: MultiProcessStore | None = None

    def __post_init__(self):
        self._run_task: asyncio.Task | None = None
        self._policy_proc: ProcMesh | None = None
        self._worker_procs: ProcMesh | None = None
        self.weights_version: int = 0
        self._updating_weights: bool = False
        self._request_queue: list[tuple[str, int, asyncio.Future]] = []
        self.running: bool = False

    @classmethod
    async def launch(  # pyright: ignore[reportIncompatibleMethodOverride]
        cls: type["Policy"],
        *,
        process_config: ProcessConfig,
        config: PolicyConfig,
        store: MultiProcessStore | None = None,
        **kwargs,
    ) -> "Policy":
        # Note - get_proc_mesh will set MASTER_ADDR, MASTER_PORT and CUDA_VISIBLE_DEVICES
        # automatically.
        worker_procs = await get_proc_mesh(process_config=process_config)

        # TODO - we will want to ensure colocation with workers
        policy_proc_config = copy(process_config)
        policy_proc_config.num_procs = 1
        policy_proc_config.with_gpus = False

        policy_proc = await get_proc_mesh(process_config=policy_proc_config)
        workers = await worker_procs.spawn(
            "vllm_worker", PolicyWorker, **asdict(config.worker_params)
        )

        # TODO - expand support so name can stick within kwargs
        actor_name = kwargs.pop("name", cls.__name__)
        policy = await policy_proc.spawn(
            actor_name,
            cls,
            config=config,
            policy_worker=workers,
            store=store,
        )
        policy._policy_proc = policy_proc
        policy._worker_procs = worker_procs
        await policy.setup.call()
        return policy

    @classmethod
    async def shutdown(  # pyright: ignore[reportIncompatibleMethodOverride]
        cls: type["Policy"], actor: "Policy"
    ):
        assert (
            actor._policy_proc is not None
        ), "Tried to shutdown a policy that was not initialized correctly"
        assert (
            actor._worker_procs is not None
        ), "Tried to shutdown a policy that was not initialized correctly"

        # TODO - may want to expand stop to gracefully respond to
        # ongoing requests.
        await actor.stop.call()
        await stop_proc_mesh(actor._worker_procs)
        await stop_proc_mesh(actor._policy_proc)

    @endpoint
    async def setup(self):
        # Set up policy_worker
        assert self.policy_worker is not None, "Policy worker should not be None"
        await self.policy_worker.setup.call(store=self.store)

        self.request_id = 0
        self.requests: dict[str, tuple[None | ParentRequest, asyncio.Future]] = {}
        self.vllm_args = await self.policy_worker.get_vllm_args.choose()

        # Setup sampling params
        self.sampling_params = get_default_sampling_params(
            self.vllm_args, overrides=asdict(self.config.sampling_params)
        )

        # Setup processors
        # TODO: move all processing to the Environment
        # TODO: add support for `log_stats` and `mm_registry`
        tokenizer = init_tokenizer_from_configs(
            model_config=self.vllm_args.model_config,
            scheduler_config=self.vllm_args.scheduler_config,
            lora_config=self.vllm_args.lora_config,
        )
        self.processor = Processor(
            vllm_config=self.vllm_args, tokenizer=tokenizer, mm_registry=None
        )
        self.output_processor = OutputProcessor(tokenizer, log_stats=None)

        # Setup scheduler
        # TODO: Add support for `log_stats`
        kv_cache_configs = await self.policy_worker.setup_kv_cache.call()
        kv_cache_config = kv_cache_configs._values[0]
        self.vllm_args.cache_config.num_gpu_blocks = kv_cache_config.num_blocks
        self.vllm_args.cache_config.num_cpu_blocks = 0

        structured_output_manager = StructuredOutputManager(self.vllm_args)
        self.scheduler = Scheduler(
            vllm_config=self.vllm_args,
            kv_cache_config=kv_cache_config,
            structured_output_manager=structured_output_manager,
            include_finished_set=False,
            log_stats=None,
        )
        self.start_processing()

    def start_processing(self):
        """Start the replica's processing loop if not already running."""
        if self._run_task is None or self._run_task.done():
            self._run_task = asyncio.create_task(self.run())

    @endpoint
    async def generate(self, prompt: str, priority: int = 0) -> RequestOutput | None:
        """Generate a response for the given prompt."""
        if self._updating_weights:
            request_future = asyncio.Future()
            self._request_queue.append((prompt, priority, request_future))
            return await request_future
        return await self._generate(prompt, priority)

    async def _generate(self, prompt: str, priority: int = 0) -> RequestOutput | None:
        """Internal generation method that doesn't check for weight updates."""
        self.request_id += 1 % sys.maxsize
        request_id = str(self.request_id)  # implement from a counter

        # Wraps prompt into a dict
        prompt_dict: dict[str, str] = convert_input(prompt=prompt)

        # truncate prmpt
        tokenization_kwargs = self.tokenization_kwargs or {}
        # TODO: add truncation support https://github.com/vllm-project/vllm/issues/4507
        truncate_prompt_tokens = self.sampling_params.truncate_prompt_tokens
        _validate_truncation_size(
            self.vllm_args.model_config.max_model_len,
            truncate_prompt_tokens,
            tokenization_kwargs,
        )

        # process and tokenize prompt
        prompt_str, request = self.processor.process_inputs(
            request_id=request_id,
            prompt=prompt_dict,
            params=self.sampling_params,
            arrival_time=None,
            lora_request=self.lora_request,
            tokenization_kwargs=tokenization_kwargs,
            trace_headers=None,
            priority=priority,
            data_parallel_rank=None,
        )

        # Explicitly keeping the redundant logic to make it easier to pick up
        # vllm changes
        # TODO: Clean up before release
        if (num_samples := self.sampling_params.n) == 1:
            self.output_processor.add_request(request, prompt_str, None, 0)
            request, _ = self.preprocess_add_request(request)
            request_fut = asyncio.Future()
            self.requests[request_id] = (None, request_fut)

            self.scheduler.add_request(request)
        else:
            parent_req = ParentRequest(request_id, self.sampling_params)
            for idx in range(num_samples):
                # Note: `get_child_info` mutates ParentRequest to track the
                # generated child request
                child_request_id, params = parent_req.get_child_info(idx)
                child_request = request if idx == num_samples - 1 else copy(request)
                child_request.request_id = child_request_id
                child_request.sampling_params = params
                self.output_processor.add_request(
                    child_request, prompt_str, parent_req, idx
                )
                child_request, _ = self.preprocess_add_request(child_request)

                self.scheduler.add_request(child_request)
            request_fut = asyncio.Future()
            self.requests[request_id] = (parent_req, request_fut)

        # Yield control to allow the run() loop to process the scheduled request
        await asyncio.sleep(0)

        try:
            generations = await request_fut
            return generations
        except asyncio.exceptions.CancelledError:
            self.logger.debug(f"Request {request_id} was cancelled")
            return None

    # Abstracted to match vllm
    # https://github.com/vllm-project/vllm/blob/0e3bb543f064eb416bca4f6f3013efa3830b12f7/vllm/v1/engine/core.py#L419
    def preprocess_add_request(self, request: EngineCoreRequest) -> tuple[Request, int]:
        if request.mm_hashes is not None:
            raise NotImplementedError("Support for mm_hash is not implemented yet.")
        request: Request = Request.from_engine_core_request(request)
        if request.use_structured_output:
            self.scheduler.structured_output_manager.grammar_init(request)

        return request, 0  # Unused Arg: Current Wave

    async def run(self):
        # TODO: add support for `iteration_stats`
        # TODO: move postprocessing out of loop to not block
        parallel_config = self.vllm_args.parallel_config
        output_rank = parallel_config.world_size - parallel_config.tensor_parallel_size
        self.running = True
        while self.running:
            scheduler_output = self.scheduler.schedule()
            worker_outputs = await self.policy_worker.execute_model.call(
                scheduler_output
            )
            worker_output = worker_outputs._values[output_rank]
            outputs = self.scheduler.update_from_output(scheduler_output, worker_output)
            outputs = outputs.get(0) or EngineCoreOutputs()
            await asyncio.sleep(0)  # Release control before processing outputs

            processed_outputs = self.output_processor.process_outputs(
                outputs.outputs,
                engine_core_timestamp=outputs.timestamp,
                iteration_stats=None,
            )

            for request_output in processed_outputs.request_outputs:
                if request_output.finished:
                    if request_output.request_id in self.requests:
                        _, fut = self.requests.pop(request_output.request_id)
                        fut.set_result(request_output)

    @endpoint
    async def update_weights(self):
        """Update the policy weights and schedule processing of queued requests."""
        queued_count = len(self._request_queue)
        self.logger.debug(
            f"Starting weight update (v{self.weights_version} -> v{self.weights_version + 1})"
        )
        if queued_count > 0:
            self.logger.debug(
                f"Will process {queued_count} queued requests after update"
            )

        self._updating_weights = True

        # Cancel all current requests and wait for them to finish
        pending_futures = []
        for request_id, (parent_req, fut) in list(self.requests.items()):
            if not fut.done():
                fut.cancel("Received weight update, cancelling request")
                pending_futures.append(fut)

        # Wait for all cancelled requests to finish with a timeout
        if pending_futures:
            self.logger.debug(f"Cancelling {len(pending_futures)} pending requests")
            try:
                await asyncio.wait_for(
                    asyncio.gather(*pending_futures, return_exceptions=True),
                    timeout=5.0,
                )
            except asyncio.TimeoutError:
                self.logger.warning("Some requests did not cancel within timeout")

        self.requests.clear()

        try:
            await self.policy_worker.update.call(version=self.weights_version)
            self.weights_version += 1
            self.logger.info(f"Weight update completed (now v{self.weights_version})")
        except Exception as e:
            self.logger.error(f"Weight update failed: {e}")
            self._updating_weights = False
            raise

        self._updating_weights = False

        # Schedule queue processing as a separate task to avoid blocking the endpoint
        if self._request_queue:
            task = asyncio.create_task(self._process_queued_requests())
            task.add_done_callback(self._queue_processing_callback)

    async def _process_queued_requests(self):
        """Process all queued requests after weight update completes."""
        queued_requests = self._request_queue.copy()
        self._request_queue.clear()

        for i, (prompt, priority, future) in enumerate(queued_requests):
            try:
                # Use the internal method directly to avoid the updating weights check
                result = await self._generate(prompt, priority)
                future.set_result(result)
            except Exception as e:
                self.logger.error(f"Error processing queued request {i+1}: {e}")
                future.set_exception(e)

    def _queue_processing_callback(self, task: asyncio.Task):
        """Callback to handle completion/errors of queue processing task."""
        try:
            if task.exception():
                self.logger.error(f"Queue processing task failed: {task.exception()}")
            else:
                self.logger.debug("Queue processing task completed successfully")
        except Exception as e:
            self.logger.error(f"Error in queue processing callback: {e}")

    @endpoint
    async def _get_model_params(self) -> dict[str, torch.Tensor]:
        """Get the current model parameters. Only for testing purposes."""
        model_params = await self.policy_worker._get_model_params.choose()
        return model_params

    @endpoint
    async def get_version(self) -> int:
        """Get the current policy version."""
        return self.weights_version

    @endpoint
    async def stop(self):
        self.running = False


@dataclass
class PolicyWorker(ForgeActor):
    model: str
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    enforce_eager: bool = False
    vllm_args: EngineArgs = None
    state_dict_key: str = "model_state_dict"

    def __post_init__(self):
        """Build vLLM Arguments

        vLLM specific TODOS
        - output format
        - check_health
        - _aggregate workers output
        - register_failure_callback

        Testing
        - all LLM generate methods, verify against LLM inputs
        - all executor methods verify no changes
        """
        if self.vllm_args is None:
            # Use default vllm EngineArgs
            self.vllm_args = EngineArgs(
                model=self.model,
                tensor_parallel_size=self.tensor_parallel_size,
                pipeline_parallel_size=self.pipeline_parallel_size,
                enforce_eager=self.enforce_eager,
            )
            # Original method returns False when not run in the main thread
            self.vllm_args._is_v1_supported_oracle = lambda *_: True
        else:
            # Check that provided args match Policy args
            cfg = [
                "model",
                "tensor_parallel_size",
                "pipeline_parallel_size",
                "data_parallel_size",
            ]
            for key in cfg:
                value = getattr(self, key) if key != "data_parallel_size" else 1
                if getattr(self.vllm_args, key) != value:
                    self.logger.warning(
                        f"{key} args don't match value in EngineArgs, overriding with {value}"
                    )
                    setattr(self.vllm_args, key, value)
        # Build Config
        self.vllm_args = self.vllm_args.create_engine_config(UsageContext.LLM_CLASS)

    @endpoint
    async def setup(self, store: MultiProcessStore = None):
        self.torchstore = store
        # TODO: remove ["gpus"] when monarch implements a flat rank
        self.rank = current_rank()["gpus"]
        self.worker = self.setup_worker()

    @endpoint
    async def execute_model(self, schedule: SchedulerOutput):
        return self.worker.execute_model(schedule)

    async def _load_tensor_parallel_state_dict(
        self, current_state_dict: dict, version: int
    ):
        """
        Load full state dict from torchstore into tensor parallel model with deterministic sharding.
        """

        updated_count = 0
        # setting explictly to llama3 for now as its our only use case
        sharding = VLLMSharding(self.tensor_parallel_size, self.rank)

        for param_name in current_state_dict.keys():
            current_tensor = current_state_dict[param_name]

            # Load the full tensor from torchstore
            # TODO: only get the part of the tensor that is needed
            stored_tensor = await self.torchstore.get(
                f"{self.state_dict_key}{DELIM}{version}{DELIM}{param_name}"
            )
            sharding.load_from_source_to_target(
                param_name,
                stored_tensor,
                current_tensor,
            )

            updated_count += 1

    @endpoint
    async def update(self, version: int):
        """Update model weights by reading state dict from torchstore"""
        if self.torchstore is None:
            raise Exception("No torchstore configured, skipping model update")

        key = f"{self.state_dict_key}{DELIM}{version}"
        model = self.worker.model_runner.model
        start = time.time()
        new_state_dict = await get_state_dict(
            self.torchstore, f"{self.state_dict_key}{DELIM}{version}"
        )
        # We use the load_weights method from vLLM b/c they perform custom mapping like
        # fusing QKV linear layers
        model.load_weights(list(new_state_dict.items()))
        self.logger.debug(
            f"Loaded state dict from {key} in {time.time() - start} seconds"
        )

    @endpoint
    async def setup_kv_cache(self):
        """Based on vllm/v1/engine/core.py:EngineCore._initialize_kv_caches
        TODO: test that fails if vllm method updates
        """
        kv_cache_spec = self.worker.get_kv_cache_spec()
        if kv_cache_spec is not None:
            available_gpu_memory = self.worker.determine_available_memory()
        else:
            # Attention free models don't need memory for kv cache
            available_gpu_memory = 0

        # Get the kv cache tensor size
        kv_cache_config = get_kv_cache_config(
            self.vllm_args, kv_cache_spec, available_gpu_memory
        )
        # TODO: unify configs across TorchStore
        # unify_kv_cache_configs(kv_cache_configs)
        self.vllm_args.cache_config.num_gpu_blocks = kv_cache_config.num_blocks
        self.vllm_args.cache_config.num_cpu_blocks = 0

        # Initialize kv cache and warmup the execution:
        # from multiproc_executor.py:MultiprocExecutor.initialize_from_config
        kv_cache_configs = [None] * self.vllm_args.parallel_config.world_size
        kv_cache_configs[self.rank] = kv_cache_config
        self.worker.initialize_from_config(kv_cache_configs)
        self.worker.compile_or_warm_up_model()
        self.worker.initialize_cache(kv_cache_config.num_blocks, 0)
        return kv_cache_config

    @endpoint
    async def get_vllm_args(self):
        return self.vllm_args

    @endpoint
    async def _get_model_params(self) -> dict[str, torch.Tensor]:
        model = self.worker.model_runner.model
        state_dict = {}

        for name, param in model.named_parameters():
            if "layers.0" not in name:
                continue
            state_dict[name] = param.cpu().detach()
        return state_dict

    def setup_worker(self):
        """Build and Instantiate vLLM worker"""
        parallel_config = self.vllm_args.parallel_config
        set_multiprocessing_worker_envs(parallel_config)
        ip, port = os.getenv("MASTER_ADDR"), os.getenv("MASTER_PORT")
        distributed_init_method = get_distributed_init_method(ip, port)
        all_kwargs = [{}] * parallel_config.world_size
        local_rank = self.rank % torch.accelerator.device_count()
        is_driver_worker = self.rank % parallel_config.tensor_parallel_size == 0
        all_kwargs[self.rank] = {
            "vllm_config": self.vllm_args,
            "local_rank": local_rank,
            "rank": self.rank,
            "distributed_init_method": distributed_init_method,
            "is_driver_worker": is_driver_worker,
        }
        worker = WorkerWrapperBase(self.vllm_args, self.rank)
        worker.init_worker(all_kwargs)
        worker.init_device()
        worker.load_model()
        return worker


def convert_input(prompt=None, prompt_token_ids=None) -> dict:
    assert (prompt is None) ^ (prompt_token_ids is None)
    if prompt is not None:
        return {"prompt": prompt}
    return {"prompt_token_ids": prompt_token_ids}


def get_default_sampling_params(vllm_config, overrides=None) -> SamplingParams:
    default_params = vllm_config.model_config.get_diff_sampling_param()
    if overrides is not None:
        default_params |= overrides
    if default_params:
        params = SamplingParams.from_optional(**default_params)
    else:
        params = SamplingParams()
    # We only care about the final output
    params.output_kind = RequestOutputKind.FINAL_ONLY
    return params
