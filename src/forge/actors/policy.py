# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import logging
import os
import sys
from dataclasses import dataclass

import torch
from monarch.actor import Actor, current_rank, endpoint, proc_mesh
from torchstore import MultiProcessStore

from torchstore._state_dict_utils import DELIM

from vllm.engine.arg_utils import EngineArgs
from vllm.entrypoints.utils import _validate_truncation_size
from vllm.executor.multiproc_worker_utils import set_multiprocessing_worker_envs
from vllm.lora.request import LoRARequest
from vllm.sampling_params import RequestOutputKind, SamplingParams
from vllm.transformers_utils.tokenizer_group import init_tokenizer_from_configs
from vllm.usage.usage_lib import UsageContext
from vllm.utils import get_distributed_init_method, get_loopback_ip, get_open_port
from vllm.v1.core.kv_cache_utils import get_kv_cache_config
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine import EngineCoreOutputs
from vllm.v1.engine.output_processor import OutputProcessor
from vllm.v1.engine.processor import Processor
from vllm.v1.request import Request
from vllm.v1.structured_output import StructuredOutputManager
from vllm.worker.worker_base import WorkerWrapperBase

from forge.data.llama3_sharding import (
    calculate_tensor_shard,
    get_tensor_parallel_sharding_strategy,
)

logger = logging.getLogger(__name__)


@dataclass
class PolicyRouter(Actor):
    # TODO: Add dp support
    policy: Actor
    sampling_params: SamplingParams = None
    lora_request: LoRARequest = None
    tokenization_kwargs: dict = None

    @endpoint
    async def setup(self):
        self.request_id = 0
        self.requests = {}
        self.vllm_args = await self.policy.get_vllm_args.choose()
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

        # Setup schduuler
        # TODO: Add support for `log_stats`
        kv_cache_configs = await self.policy.setup_kv_cache.call()
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

    @endpoint
    async def generate(self, prompt: str, priority: int = 0):
        self.request_id += 1 % sys.maxsize
        request_id = str(self.request_id)  # implement from a counter

        prompt = convert_input(prompt)
        if self.sampling_params is None:
            self.sampling_params = get_default_sampling_params(self.vllm_args)

        # truncate prmpt
        tokenization_kwargs = self.tokenization_kwargs or {}
        truncate_prompt_tokens = self.sampling_params.truncate_prompt_tokens
        _validate_truncation_size(
            self.vllm_args.model_config.max_model_len,
            truncate_prompt_tokens,
            tokenization_kwargs,
        )

        # process and tokenize prompt
        prompt_str, request = self.processor.process_inputs(
            request_id=request_id,
            prompt=prompt,
            params=self.sampling_params,
            arrival_time=None,
            lora_request=self.lora_request,
            tokenization_kwargs=tokenization_kwargs,
            trace_headers=None,
            priority=priority,
            data_parallel_rank=None,
        )

        if self.sampling_params.n == 1:
            self.output_processor.add_request(request, prompt_str, None, 0)
            request = Request.from_engine_core_request(request)
            # TODO: mm_hash and sturcutured_output
            request_fut = asyncio.Future()
            self.requests[request_id] = request_fut
            self.scheduler.add_request(request)
        else:
            raise NotImplementedError("Multiple samples not supported yet")
            # # Fan out child requests (for n>1).
            # parent_req = ParentRequest(request_id, sampling_params)
            # for idx in range(sampling_params.n):
            #     request_id, params = parent_req.get_child_info(idx)
            #     child_request = request if idx == n - 1 else copy(request)
            #     child_request.request_id = request_id
            #     child_request.sampling_params = sampling_params
            #     # Make a new RequestState and queue.
            #     output_processor.add_request(child_request, prompt_str,
            #                                     parent_req, idx)
            #     child_request = Request.from_engine_core_request(child_request)
            #     self.scheduler.add_request(chile_request)

        return await request_fut

    @endpoint
    async def run(self):
        # TODO: add support for `iteration_stats`
        # TODO: move postprocessing out of loop to not block
        parallel_config = self.vllm_args.parallel_config
        output_rank = parallel_config.world_size - parallel_config.tensor_parallel_size
        self.running = True
        while self.running:
            scheduler_output = self.scheduler.schedule()
            worker_outputs = await self.policy.execute_model.call(scheduler_output)
            worker_output = worker_outputs._values[output_rank]
            outputs = self.scheduler.update_from_output(scheduler_output, worker_output)
            outputs = outputs.get(0) or EngineCoreOutputs()
            await asyncio.sleep(0)  # Release control before processing outputs
            processed_outputs = self.output_processor.process_outputs(
                outputs.outputs,
                engine_core_timestamp=outputs.timestamp,
                iteration_stats=None,
            )
            for output in processed_outputs.request_outputs:
                if output.finished:
                    fut = self.requests.pop(output.request_id)
                    fut.set_result(output)

    @endpoint
    async def shutdown(self):
        self.running = False


@dataclass
class Policy(Actor):
    model: str
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    enforce_eager: bool = False
    vllm_args: EngineArgs = None
    resources: int = 1
    torchstore: MultiProcessStore = None
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
                gpu_memory_utilization=0.4,
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
                    logger.warning(
                        f"{key} args don't match value in EngineArgs, overriding with {value}"
                    )
                    setattr(self.vllm_args, key, value)
        # Build Config
        self.vllm_args = self.vllm_args.create_engine_config(UsageContext.LLM_CLASS)
        assert self.vllm_args.parallel_config.world_size == self.resources

    @endpoint
    async def setup(self):
        # TODO: remove ["gpus"] when monarch implements a flat rank
        self.rank = current_rank()["gpus"]
        self.worker = self.setup_worker()

    @endpoint
    async def execute_model(self, schedule: SchedulerOutput):
        return self.worker.execute_model(schedule)

    async def _load_tensor_parallel_state_dict(self, current_state_dict: dict):
        """
        Load full state dict from torchstore into tensor parallel model with deterministic sharding.
        """

        updated_count = 0

        for param_name in current_state_dict.keys():
            current_tensor = current_state_dict[param_name]

            # Load the full tensor from torchstore
            stored_tensor = await self.torchstore.get(
                f"{self.state_dict_key}{DELIM}{param_name}"
            )

            # Determine sharding strategy for this parameter
            shard_dim, is_sharded = get_tensor_parallel_sharding_strategy(param_name)

            if not is_sharded:
                # Parameter is replicated - shapes should match exactly
                if stored_tensor.shape != current_tensor.shape:
                    raise ValueError(
                        f"Replicated parameter {param_name} has mismatched shapes: "
                        f"{stored_tensor.shape} vs {current_tensor.shape}, skipping"
                    )

                # Direct copy for replicated parameters
                current_state_dict[param_name].copy_(stored_tensor)

            else:
                # Need to shard the full tensor
                sharded_tensor = calculate_tensor_shard(
                    stored_tensor, shard_dim, self.tensor_parallel_size, self.rank
                )

                if sharded_tensor.shape != current_tensor.shape:
                    raise ValueError(
                        f"Calculated shard for {param_name} has wrong shape: "
                        f"{sharded_tensor.shape} vs expected {current_tensor.shape}, skipping"
                    )

                current_state_dict[param_name].copy_(sharded_tensor)

            updated_count += 1

        logger.info(f"Successfully updated {updated_count} parameters")

    @endpoint
    async def update(self):
        """Update model weights by reading state dict from torchstore"""

        if self.torchstore is None:
            raise Exception("No torchstore configured, skipping model update")

        logger.info(
            f"Starting model update from torchstore with key: {self.state_dict_key}"
        )

        # Get the current model from the worker
        model = self.worker.model_runner.model
        current_state_dict = model.state_dict()

        logger.info(f"Current state dict has {len(current_state_dict)} parameters")
        logger.info(f"Tensor parallel size: {self.tensor_parallel_size}")

        # Tensor parallel model - use deterministic sharding strategy
        logger.info("Loading state dict with tensor parallel sharding...")
        await self._load_tensor_parallel_state_dict(current_state_dict)

        # Load the updated state dict into the model
        model.load_state_dict(current_state_dict, strict=True)

        logger.info("Successfully updated model weights from torchstore")

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
    async def get_model_params(self):
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


def convert_input(prompt=None, prompt_token_ids=None):
    assert prompt is None or prompt_token_ids is None
    if prompt is not None:
        return {"prompt": prompt}
    elif prompt_token_ids is not None:
        return {"prompt_token_ids": prompt_token_ids}
    else:
        raise ValueError("Either prompt or prompt_token_ids must be provided.")


def get_default_sampling_params(vllm_config, overrides=None):
    default_params = vllm_config.model_config.get_diff_sampling_param()
    default_params["max_tokens"] = 512
    if overrides is not None:
        default_params |= overrides
    if default_params:
        params = SamplingParams.from_optional(**default_params)
    else:
        params = SamplingParams()
    # We only care about the final output
    params.output_kind = RequestOutputKind.FINAL_ONLY
    return params


async def _test(config):
    # TODO: Create proper test
    router_mesh = await proc_mesh(gpus=1)
    policy_mesh = await proc_mesh(
        gpus=config["resources"],
        env={
            "MASTER_ADDR": str(get_loopback_ip()),
            "MASTER_PORT": str(get_open_port()),
        },
    )

    policy_actor = await policy_mesh.spawn("policy", Policy, **config)
    router = await router_mesh.spawn("policy_router", PolicyRouter, policy=policy_actor)

    await policy_actor.setup.call()
    await router.setup.call()
    print("Model setup")

    router.run.call()
    print("Model running")

    prompt = "Tell me a joke"
    response = await router.generate.call_one(prompt)
    print(f"User: {prompt}\nAssistant: {response.outputs[0].text}")

    await router.shutdown.call()


if __name__ == "__main__":
    config = {
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "tensor_parallel_size": 2,
        "pipeline_parallel_size": 1,
        "enforce_eager": True,
        "resources": 2,
    }
    asyncio.run(_test(config))
