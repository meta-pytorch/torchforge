# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import logging
from dataclasses import dataclass

from monarch.actor import Actor, endpoint, current_rank, proc_mesh

from vllm.engine.arg_utils import EngineArgs
from vllm.usage.usage_lib import UsageContext
from vllm.worker.worker_base import WorkerWrapperBase
from vllm.entrypoints.utils import _validate_truncation_size
from vllm.inputs import TextPrompt, TokensPrompt
from vllm.lora.request import LoRARequest
from vllm.sampling_params import RequestOutputKind, SamplingParams
from vllm.v1.engine.processor import Processor
from vllm.v1.engine.output_processor import OutputProcessor
from vllm.transformers_utils.tokenizer_group import init_tokenizer_from_configs
from vllm.v1.request import Request
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.utils import get_distributed_init_method, get_loopback_ip, get_open_port
from vllm.executor.multiproc_worker_utils import set_multiprocessing_worker_envs


import pdb # philip

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
        self.vllm_args = await self.policy.vllm_args.call_one()

        # Setup processors
        # TODO: move all processing to the Environment
        # TODO: add support for `log_stats` and `mm_registry`
        tokenizer = init_tokenizer_from_configs(
            model_config=self.vllm_args.model_config,
            scheduler_config=self.vllm_args.scheduler_config,
            lora_config=self.vllm_args.lora_config)
        self.processor = Processor(
            vllm_config=self.vllm_args,
            tokenizer=tokenizer,
            mm_registry=None)
        self.output_processor = OutputProcessor(tokenizer, log_stats=None)

        # Setup schduuler
        # TODO: Add support for `log_stats`
        kv_cache_configs = await self.policy.setup_kv_cache.call()
        structured_output_manager = StructuredOutputManager(self.vllm_args)
        self.scheduler = Scheduler(
            vllm_config=self.vllm_args,
            kv_cache_config=kv_cache_configs[0],
            structured_output_manager=structured_output_manager,
            include_finished_set=False,
            log_stats=None,
        )


    @endpoint
    async def generate(self, prompt: str, priority: int = 0):
        self.request_id += 1 % sys.maxsize
        request_id = str(self.request_id) # implement from a counter

        prompt = convert_input(prompt)
        if self.sampling_params is None:
            self.sampling_params = get_default_sampling_params(self.vllm_args)

        # truncate prmpt
        tokenization_kwargs = self.tokenization_kwargs or {}
        truncate_prompt_tokens = sampling_params.truncate_prompt_tokens
        _validate_truncation_size(self.vllm_args.model_config.max_model_len, truncate_prompt_tokens, tokenization_kwargs)

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
            self.requests["request_id"] = asyncio.Future()
            self.scheduer.add_request(request)
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
            #     self.scheduer.add_request(chile_request)
        
        return await self.request[request_id]

    @endpoint
    async def run(self):
        # TODO: add support for `iteration_stats`
        # TODO: move postprocessing out of loop to not block
        self.running = True
        while self.running:
            scheduler_output = self.scheduler.schedule()
            worker_output = await self.policy.execute_model(scheduler_output)
            outputs = self.scheduler.update_from_output(scheduler_output, worker_output)
            outputs = outputs.get(0) or EngineCoreOutputs
            await asyncio.sleep(0) # philip
            processed_outputs = self.output_processor.process_outputs(
                outputs.outputs,
                engine_core_timestamp=outputs.timestamp,
                iteration_stats=None)
            for output in processed_outputs:
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

    def __post_init__(self):
        """ Build vLLM Arguments 

        vLLM specific TODOS
        - output rank
        - output format
        - check_health
        - _aggregate workers output
        - register_failure_callback
        - _get_output_rank

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
        else:
            # Check that provided args match Policy args
            cfg = ["model", "tensor_parallel_size", "pipeline_parallel_size", "data_parallel_size"]
            for key in cfg:
                value = getattr(self, key) if key != "data_parallel_size" else 1
                if getattr(self.vllm_args, key) != value:
                    logger.warning(f"{key} args don't match value in EngineArgs, overriding with {value}")
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

    @endpoint
    async def update(self):
        # TODO: add TorchStore support
        pass

    @endpoint
    async def setup_kv_cache(self):
        """ Based on vllm/v1/engine/core.py:EngineCore._initialize_kv_caches
            TODO: test that fails if vllm method updates
        """
        kv_cache_spec = self.worker.get_kv_cache_spec()
        if kv_cache_spec is not None:
            available_gpu_memory = self.worker.determine_available_memory()
        else:
            # Attention free models don't need memory for kv cache
            available_gpu_memory = 0

        # Get the kv cache tensor size
        kv_cache_config = get_kv_cache_config(vllm_config, kv_cache_spec, available_gpu_memory)

        # TODO: unify configs across TorchStore
        # unify_kv_cache_configs(kv_cache_configs)
        self.vllm_args.cache_config.num_gpu_blocks = kv_cache_config.num_blocks
        self.vllm_args.cache_config.num_cpu_blocks = 0

        # Initialize kv cache and warmup the execution: 
        # from multiproc_executor.py:MultiprocExecutor.initialize_from_config
        kv_cache_configs = [None] * vllm_config.parallel_config.world_size
        kv_cache_configs[self.rank] = kv_cache_config
        self.worker.initialize_from_config(kv_cache_configs)
        self.worker.compile_or_warmup_model()
        self.worker.initialize_cache(kv_cache_config.num_blocks, 0)
        return kv_cache_config

    def setup_worker(self):
        """ Build and Instantiate vLLM worker """
        parallel_config = self.vllm_args.parallel_config
        set_multiprocessing_worker_envs(parallel_config)
        distributed_init_method = get_distributed_init_method(get_loopback_ip(), get_open_port())
        all_kwargs = [{}] * parallel_config.world_size
        is_driver_worker = (self.rank % parallel_config.tensor_parallel_size == 0)
        all_kwargs[self.rank] = {
            "vllm_config": self.vllm_args,
            "local_rank": self.rank,
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
    if overrides is not None:
        default_params |= overrides
    if default_params:
        params = SamplingParams.from_optional(**default_params)
    else:
        params = SamplingParams()
    # We only care about the final output
    params.output_kind = RequestOutputKind.FINAL_ONLY
    return params


async def main(config):
    router_mesh = await proc_mesh(gpus=1)
    policy_mesh = await proc_mesh(gpus=config["resources"], env={
        #"CUDA_VISIBLE_DEVICES": "1", # philip
        },)
    
    policy_actor = await policy_mesh.spawn("policy", Policy, **config)    
    router = await router_mesh.spawn("policy_router", PolicyRouter, policy=policy_actor)

    await policy_actor.setup.call()
    await router.setup.call()
    print("Model setup")

    router.run.call()
    print("Model running")

    prompt = "Tell me a joke"
    response = await router.generate.call(prompt)
    print(f"User: {prompt}\nAssistant: {response}")

    await router.shutdown.call()

if __name__ == "__main__":
    config = {
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "tensor_parallel_size": 2,
        "pipeline_parallel_size": 1,
        "enforce_eager": True,
        "resources": 2,
    }
    asyncio.run(main(config), debug=True)  # philip  
