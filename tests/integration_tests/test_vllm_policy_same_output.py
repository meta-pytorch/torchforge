# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import asyncio

from forge.actors.policy import Policy
from forge.observability.metric_actors import get_or_create_metric_logger
from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.v1.engine.async_llm import AsyncLLM


# Configuration
MODEL_NAME = "facebook/opt-125m"
MAX_MODEL_LEN = 512
GPU_MEMORY_UTILIZATION = 0.1
ENFORCE_EAGER = True
ENABLE_PREFIX_CACHING = True
TENSOR_PARALLEL_SIZE = 1

# Sampling parameters
MAX_TOKENS = 50
TEMPERATURE = 0.0  # Deterministic
TOP_P = 1.0
N_SAMPLES = 1

# Test prompts
TEST_PROMPTS = [
    "Hello, how are you?",
    "What is 2+2?",
    "Tell me a joke.",
    "Explain machine learning briefly.",
    "What color is the sky?",
]


async def main():
    """Compare outputs between vLLM and Policy service"""
    policy = None
    try:
        # Setup vLLM directly
        args = AsyncEngineArgs(
            model=MODEL_NAME,
            max_model_len=MAX_MODEL_LEN,
            gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
            enforce_eager=ENFORCE_EAGER,
            enable_prefix_caching=ENABLE_PREFIX_CACHING,
        )
        vllm_model = AsyncLLM.from_engine_args(args)

        # Setup Policy service
        # TODO: Remove metric logger instantiation after https://github.com/meta-pytorch/forge/pull/303 lands
        mlogger = await get_or_create_metric_logger()
        await mlogger.init_backends.call_one({"console": {"log_per_rank": False}})

        policy = await Policy.options(
            procs=1, num_replicas=1, with_gpus=True
        ).as_service(
            engine_config={
                "model": MODEL_NAME,
                "tensor_parallel_size": TENSOR_PARALLEL_SIZE,
                "enforce_eager": ENFORCE_EAGER,
                "max_model_len": MAX_MODEL_LEN,
                "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,
                "enable_prefix_caching": ENABLE_PREFIX_CACHING,
            },
            sampling_config={
                "n": N_SAMPLES,
                "max_tokens": MAX_TOKENS,
                "temperature": TEMPERATURE,
                "top_p": TOP_P,
            },
        )

        print("Models ready. Generating outputs...\n")
        vllm_outputs = []
        policy_outputs = []
        sampling_params = SamplingParams(
            max_tokens=MAX_TOKENS, temperature=TEMPERATURE, top_p=TOP_P, n=N_SAMPLES
        )

        for i, prompt in enumerate(TEST_PROMPTS, 1):
            # vLLM generation
            vllm_text = None
            async for res in vllm_model.generate(
                prompt, sampling_params, request_id=str(i)
            ):
                vllm_text = res.outputs[0].text
            vllm_outputs.append(vllm_text)

            # Policy generation
            policy_result = await policy.generate.route(prompt)
            policy_text = policy_result[0].text
            policy_outputs.append(policy_text)

        # Final check
        for vllm_output, policy_output in zip(vllm_outputs, policy_outputs):
            if vllm_output != policy_output:
                print(f"❌ Got different results: {vllm_output} vs. {policy_output}")
        print("✅ Outputs are the same!")

    finally:
        if policy is not None:
            await policy.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
