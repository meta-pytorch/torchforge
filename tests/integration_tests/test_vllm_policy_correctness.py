# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import asyncio

from forge.actors.policy import Policy
from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.sampling_params import RequestOutputKind
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


async def test_same_output():
    """Compare outputs between vLLM and Policy service"""
    test_prompts = [
        "Hello, how are you?",
        "What is 2+2?",
        "Tell me a joke.",
        "Explain machine learning briefly.",
        "What color is the sky?",
    ]
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
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            n=N_SAMPLES,
            output_kind=RequestOutputKind.FINAL_ONLY,
        )

        for i, prompt in enumerate(test_prompts, 1):
            # vLLM generation
            async for res in vllm_model.generate(
                prompt, sampling_params, request_id=str(i)
            ):
                vllm_outputs.append(res.outputs[0].text)

            # Policy generation
            policy_result = await policy.generate.route(prompt)
            policy_text = policy_result[0].text
            policy_outputs.append(policy_text)

        # Final check
        for vllm_output, policy_output in zip(vllm_outputs, policy_outputs):
            assert vllm_output != ""
            assert policy_output != ""
            if vllm_output != policy_output:
                print(f"❌ Got different results: {vllm_output} vs. {policy_output}")
        print("✅ Outputs are the same!")

    finally:
        if policy is not None:
            await policy.shutdown()


async def test_cache_usage():
    """Test that KV cache usage is consistent between vLLM and Policy service.

    Namely we want to check two things:
    1. KV cache is populated correctly.
    2. KV cache is cleared correctly.

    Our main tool to inspect the KV cache is the `num_cached_tokens` field in the request output.
    According to the vLLM docs (https://docs.vllm.ai/en/v0.9.0/api/vllm/outputs.html#vllm.outputs.RequestOutput),
    this is the number of tokens with a prefix cache hit. So, the logic is that if we run one generation,
    then run another generation with the same start, we should see the number of cached tokens == the length of the prefix.

    Some important caveats:
    - vLLM does not appear to do partial prefix caching. So if a shared prefix is less than BLOCK_SIZE,
    it won't be cached.
    - This is a limited test. Ideally, it would also be good to check the size of the block pool before and after
    each generation. In addition, it would be interesting to examine the GPU memory freed after
    calling reset_prefix_cache(); however, it is not exactly clear how to access these internal APIs
    via the AsyncLLM interface.
    - We do not test different different block sizes.
    """
    policy = None
    try:
        # Setup vLLM directly
        args = AsyncEngineArgs(
            model=MODEL_NAME,
            max_model_len=MAX_MODEL_LEN,
            gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
            enforce_eager=ENFORCE_EAGER,
            enable_prefix_caching=ENABLE_PREFIX_CACHING,
            block_size=16,
        )
        vllm_model = AsyncLLM.from_engine_args(args)

        # Setup Policy service
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
                "block_size": 16,
            },
            sampling_config={
                "n": N_SAMPLES,
                "max_tokens": MAX_TOKENS,
                "temperature": TEMPERATURE,
                "top_p": TOP_P,
            },
        )

        print("Models ready. Starting KV cache test...")

        sampling_params = SamplingParams(
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            n=N_SAMPLES,
            output_kind=RequestOutputKind.FINAL_ONLY,
        )
        vllm_outputs = []
        policy_outputs = []

        # Exactly 16 tokens to fill up 1 block
        first_prompt = (
            "The paged prefix caching mechanism in vLLM is an interesting approach."
        )
        expected_cached_tokens = 0
        async for res in vllm_model.generate(
            first_prompt, sampling_params, request_id="first_16"
        ):
            vllm_outputs.append(res.outputs[0].text)
            assert res.num_cached_tokens == expected_cached_tokens
        res = await policy.generate.route(first_prompt)
        assert res[0].metadata["num_cached_tokens"] == expected_cached_tokens
        policy_outputs.append(res[0].text)

        # Another 16 tokens to now populate 2 blocks (+ reuse the first block)
        second_prompt = (
            first_prompt
            + " It removes the need to recalculate attention key-values for already processed text."
        )
        expected_cached_tokens = 16
        async for res in vllm_model.generate(
            second_prompt, sampling_params, request_id="second_16_use_first_block"
        ):
            vllm_outputs.append(res.outputs[0].text)
            assert res.num_cached_tokens == expected_cached_tokens
        res = await policy.generate.route(second_prompt)
        assert res[0].metadata["num_cached_tokens"] == expected_cached_tokens
        policy_outputs.append(res[0].text)

        # The first same 32 tokens should now be populated in blocks
        third_prompt = second_prompt
        expected_cached_tokens = 32
        async for res in vllm_model.generate(
            third_prompt, sampling_params, request_id="use_both_blocks"
        ):
            vllm_outputs.append(res.outputs[0].text)
            assert res.num_cached_tokens == expected_cached_tokens
        res = await policy.generate.route(third_prompt)
        assert res[0].metadata["num_cached_tokens"] == expected_cached_tokens
        policy_outputs.append(res[0].text)

        # Now, let's clear the cache
        await vllm_model.reset_prefix_cache()
        await policy._reset_prefix_cache.route()

        # And try the third prompt again (should not use any cached tokens)
        expected_cached_tokens = 0
        async for res in vllm_model.generate(
            third_prompt, sampling_params, request_id="use_no_blocks_bc_cache_cleared"
        ):
            vllm_outputs.append(res.outputs[0].text)
            assert res.num_cached_tokens == expected_cached_tokens
        res = await policy.generate.route(third_prompt)
        assert res[0].metadata["num_cached_tokens"] == expected_cached_tokens
        policy_outputs.append(res[0].text)

        # Sanity check that outputs are still the same
        for vllm_output, policy_output in zip(vllm_outputs, policy_outputs):
            assert vllm_output != ""
            assert policy_output != ""
            if vllm_output != policy_output:
                print(f"❌ Got different results: {vllm_output} vs. {policy_output}")

        print("\n✅ Prefix cache usage is the same!")

    finally:
        if policy is not None:
            await policy.shutdown()


if __name__ == "__main__":
    asyncio.run(test_same_output())
    asyncio.run(test_cache_usage())
